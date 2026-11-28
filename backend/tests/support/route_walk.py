"""The derived route walk of FTDD §16.4 (045 FR-14; CHAIN_PLAN A27, A35, A40; FTASKS 20.1, 22.1).

From each live route's endpoint function, walk the call graph TO A FIXPOINT:
* callees resolve through the calling function's module globals, its function-local imports (relative ones resolved
  against the module's package), `Class.method` and `self.method` in the same class, and `module.function`;
* the walk stays inside `src.api`, `src.services`, `src.workers` and `src.utils` (the miServer helpers), and a
  task's own body is walked when the walk enqueues it (`X.delay`, `X.apply_async`, `gpu_delay(X)`,
  `dispatch_gpu_task(X, …)`, and the token-redacting `_queue_without_leaking_token(X, …)`, bound per call site);
* two categories of primitive: GPU starts (`dispatch_gpu_task`, `gpu_delay`, the enqueue of a task carrying
  `__gpu_job__` or listed in `GPU_TASKS_WITHOUT_MARKER`, `logit_lens_lease`, `ensure_steering_worker`,
  `ensure_model_loaded`) and outside paid calls (`ask_judge`, `_client_for`).

`walk_routes(app)` → {(method, path): {"gpu", "outside"} ∩ categories reached}. The knobs exist so the controls can
cut the walk (stop at the endpoint's module, stop before `src.workers`, drop a primitive) and watch rows drop out.
"""
from __future__ import annotations

import ast
import importlib
import inspect
import textwrap
from typing import Any, Callable, Iterable, Optional

GPU_PRIMITIVES = frozenset({"dispatch_gpu_task", "gpu_delay", "logit_lens_lease", "ensure_steering_worker",
                            "ensure_model_loaded"})
OUTSIDE_PRIMITIVES = frozenset({"ask_judge", "_client_for"})
ENQUEUE_METHODS = frozenset({"delay", "apply_async"})
TASK_BINDERS = frozenset({"gpu_delay", "dispatch_gpu_task", "_queue_without_leaking_token"})
DEFAULT_PACKAGES = ("src.api", "src.services", "src.workers", "src.utils")

#: Tasks that start GPU work but carry no `__gpu_job__` marker the walk can read, each with its reason. A row here
#: must be NEEDED: `test_every_unmarked_row_is_needed` deletes each in turn and requires a route to drop out.
GPU_TASKS_WITHOUT_MARKER: dict[str, str] = {
    "label_features_trial_task": (
        "row 46: a labelling trial is an outside paid call through its own `OpenAILabelingService` client "
        "(`services/labeling_trial_service.py` `run_trial`), not `ask_judge`/`_client_for`, so the walk reaches no "
        "primitive in its body; the task carries no `__gpu_job__` (it runs on `processing`)"),
}


def marked_tasks() -> set[str]:
    """Names of every Celery task under `src.workers` whose function carries the `__gpu_job__` marker."""
    import pkgutil

    import src.workers as workers
    from src.services.gpu_dispatch import gpu_job_spec

    names: set[str] = set()
    for info in pkgutil.iter_modules(workers.__path__):
        try:
            module = importlib.import_module(f"src.workers.{info.name}")
        except Exception:  # noqa: BLE001 - a module that cannot import has no task to mark
            continue
        for name, obj in vars(module).items():
            if hasattr(obj, "delay") and gpu_job_spec(obj) is not None:
                names.add(name)
    return names


class Walker:
    def __init__(self, *, packages: Iterable[str] = DEFAULT_PACKAGES, gpu: Iterable[str] = GPU_PRIMITIVES,
                 outside: Iterable[str] = OUTSIDE_PRIMITIVES, unmarked: Optional[dict] = None,
                 stop_at_module: bool = False):
        self.packages = tuple(packages)
        self.gpu, self.outside = frozenset(gpu), frozenset(outside)
        self.marked = marked_tasks()
        self.unmarked = dict(GPU_TASKS_WITHOUT_MARKER if unmarked is None else unmarked)
        self.stop_at_module = stop_at_module
        self.memo: dict[int, frozenset] = {}
        self.visited: set[tuple[str, str]] = set()
        self.root_module: Optional[str] = None
        #: R1 review fix (OR1-045a0-3), 2026-11-16: every enqueue (`X.delay`, `X.apply_async`, a task binder) whose
        #: task expression the walk cannot resolve to a task, as `(module, qualname, line, expression)`. The walk
        #: FAILS CLOSED on these: `test_the_walk_resolves_every_enqueue` requires none outside its reviewed table.
        self.unresolved: set[tuple[str, str, int, str]] = set()

    # ------------------------------------------------------------------ resolution
    def _inside(self, obj: Any) -> bool:
        module = getattr(inspect.getmodule(obj), "__name__", "") or ""
        if self.stop_at_module:
            return module == self.root_module
        return module.startswith(self.packages)

    @staticmethod
    def _tree(fn: Any) -> Optional[ast.AST]:
        try:
            return ast.parse(textwrap.dedent(inspect.getsource(fn)))
        except (OSError, TypeError, SyntaxError, IndentationError):
            return None

    @staticmethod
    def _local_imports(tree: ast.AST, module_name: str) -> dict[str, Any]:
        out: dict[str, Any] = {}
        package = module_name.rsplit(".", 1)[0]
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    parts = package.split(".")
                    parts = parts[: len(parts) - (node.level - 1)] if node.level > 1 else parts
                    base = ".".join(parts + ([base] if base else []))
                for alias in node.names:
                    try:
                        mod = importlib.import_module(base)
                        out[alias.asname or alias.name] = getattr(mod, alias.name, None) or importlib.import_module(
                            f"{base}.{alias.name}")
                    except Exception:  # noqa: BLE001
                        continue
        return out

    def _resolve(self, func: ast.AST, scope: dict, owner: Any) -> Any:
        if isinstance(func, ast.Name):
            return scope.get(func.id)
        if isinstance(func, ast.Attribute):
            if isinstance(func.value, ast.Name) and func.value.id in ("self", "cls") and owner is not None:
                return getattr(owner, func.attr, None)
            base = self._resolve(func.value, scope, owner)
            if base is not None:
                return getattr(base, func.attr, None)
        return None

    @staticmethod
    def _returns(fn: Any) -> Any:
        import typing

        try:
            return typing.get_type_hints(fn).get("return")
        except Exception:  # noqa: BLE001 - an unresolvable annotation names no class
            return None

    @staticmethod
    def _line(fn: Any, node: ast.AST) -> int:
        try:
            return inspect.getsourcelines(fn)[1] + getattr(node, "lineno", 1) - 1
        except (OSError, TypeError):
            return getattr(node, "lineno", 0)

    @staticmethod
    def _name(func: ast.AST) -> Optional[str]:
        return func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else None

    def _task_hits(self, task: Any, name: Optional[str]) -> set[str]:
        hits: set[str] = set()
        if name in self.unmarked or (name in self.marked):
            hits.add("gpu")
        body = getattr(task, "run", None)
        if body is not None:
            hits |= self.walk(getattr(body, "__wrapped__", body))
        return hits

    # ------------------------------------------------------------------ the walk
    def walk(self, fn: Any) -> frozenset:
        fn = inspect.unwrap(fn) if callable(fn) else fn
        key = id(fn)
        if key in self.memo:
            return self.memo[key]
        self.memo[key] = frozenset()  # cycle guard: the fixpoint is reached by memoised recursion
        self.visited.add((getattr(inspect.getmodule(fn), "__name__", ""), getattr(fn, "__qualname__", "")))
        tree = self._tree(fn)
        if tree is None:
            return self.memo[key]
        module = inspect.getmodule(fn)
        scope = dict(getattr(fn, "__globals__", {}) or vars(module) if module else {})
        scope.update(self._local_imports(tree, getattr(module, "__name__", "")))
        owner = None
        qual = getattr(fn, "__qualname__", "")
        if "." in qual and module is not None:
            owner = getattr(module, qual.split(".")[0], None)
        # Local instances: `service = SomeService(db)` → `service.method(...)` resolves on the class.
        for node in ast.walk(tree):
            if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                    and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)):
                cls = self._resolve(node.value.func, scope, owner)
                if cls is not None and not isinstance(cls, type) and callable(cls) and self._inside(cls):
                    cls = self._returns(cls)  # a factory: `get_x_service() -> XService`
                if isinstance(cls, type) and node.targets[0].id not in scope:
                    scope[node.targets[0].id] = cls
        hits: set[str] = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            # A function handed to another as a callback (`run_in_threadpool(_try_rows, …)`,
            # `background_tasks.add_task(fn, …)`, `asyncio.to_thread(fn, …)`) is walked too.
            for arg in list(node.args) + [k.value for k in node.keywords]:
                if isinstance(arg, (ast.Name, ast.Attribute)):
                    cb = self._resolve(arg, scope, owner)
                    if (cb is not None and callable(cb) and not isinstance(cb, type)
                            and not hasattr(cb, "delay") and self._inside(cb)):
                        hits |= self.walk(cb)
            name = self._name(node.func)
            if name in self.gpu:
                hits.add("gpu")
            if name in self.outside:
                hits.add("outside")
            # enqueue of a task, bound at the call site
            task_expr = None
            if isinstance(node.func, ast.Attribute) and node.func.attr in ENQUEUE_METHODS:
                task_expr = node.func.value
            elif name in TASK_BINDERS and node.args:
                task_expr = node.args[0]
            elif isinstance(node.func, ast.Call) and self._name(node.func.func) == "gpu_delay" and node.func.args:
                task_expr = node.func.args[0]
            if task_expr is not None:
                task = self._resolve(task_expr, scope, owner)
                if task is not None and hasattr(task, "delay"):
                    hits |= self._task_hits(task, self._name(task_expr))
                    continue
                self.unresolved.add((getattr(module, "__name__", ""), qual, self._line(fn, node),
                                     ast.unparse(task_expr)))
            target = self._resolve(node.func, scope, owner)
            if target is None or not callable(target) or isinstance(target, type):
                if isinstance(target, type):  # a constructor: walk __init__ when it is ours
                    init = getattr(target, "__init__", None)
                    if init is not None and self._inside(init):
                        hits |= self.walk(init)
                continue
            if hasattr(target, "delay"):
                continue  # a task object called directly runs in-process; its own name is not an enqueue
            if self._inside(target):
                hits |= self.walk(target)
        self.memo[key] = frozenset(hits)
        return self.memo[key]


def walk_routes(app, walker: Optional[Walker] = None, **knobs) -> dict[tuple[str, str], frozenset]:
    """Every live route (every method) → the primitive categories its endpoint reaches."""
    from src.core.approval_gate import iter_routes

    walker = walker or Walker(**knobs)
    out: dict[tuple[str, str], frozenset] = {}
    for path, route in iter_routes(app):
        endpoint = getattr(route, "endpoint", None)
        if endpoint is None:
            continue
        walker.root_module = getattr(inspect.getmodule(endpoint), "__name__", None)
        if knobs.get("stop_at_module"):
            walker.memo.clear()
        reached = walker.walk(endpoint)
        for method in sorted(route.methods or ()):
            out[(method, path)] = reached
    return out


def enqueue_sites(walker: Walker, subpackages: Iterable[str] = ("api", "services", "workers")):
    """Every enqueue site in `src/<subpackage>` (`X.delay(…)`, `X.apply_async(…)`, `dispatch_gpu_task(X, …)`,
    `_queue_without_leaking_token(X, …)`, `gpu_delay(X)(…)`), with its task expression RESOLVED through the module's
    globals and the enclosing function's own imports: yields `(module, function, line, expression, task)`, where
    `task` is the resolved Celery task or `None` when the expression does not resolve to one (a subscript, a local
    variable, a call's result …). R1 review fix (OR1-045a0-3), 2026-11-16: the scan used to skip such a site."""
    from pathlib import Path

    src = Path(__file__).resolve().parents[2] / "src"
    for sub in subpackages:
        for path in sorted((src / sub).rglob("*.py")):
            module_name = "src." + ".".join(path.relative_to(src).with_suffix("").parts)
            if module_name.endswith(".__init__"):
                module_name = module_name[: -len(".__init__")]
            tree = ast.parse(path.read_text())
            module = None
            for fn in ast.walk(tree):
                if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                for call in ast.walk(fn):
                    if not isinstance(call, ast.Call):
                        continue
                    f, expr = call.func, None
                    if isinstance(f, ast.Attribute) and f.attr in ENQUEUE_METHODS:
                        expr = f.value
                    elif Walker._name(f) in TASK_BINDERS and call.args:  # `gpu_delay(X)(…)` is seen at `gpu_delay(X)`
                        expr = call.args[0]
                    if expr is None:
                        continue
                    if module is None:
                        module = importlib.import_module(module_name)
                    scope = dict(vars(module))
                    scope.update(walker._local_imports(fn, module_name))
                    task = walker._resolve(expr, scope, None)
                    yield module_name, fn.name, call.lineno, ast.unparse(expr), (
                        task if task is not None and hasattr(task, "delay") else None)
