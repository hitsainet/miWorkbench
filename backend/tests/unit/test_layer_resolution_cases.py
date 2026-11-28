"""ENH-011 X-1 (#2): on a multi-tower model, layer discovery must return the TEXT decoder's layers.

`discover_transformer_structure` used to take the first `nn.ModuleList` of transformer-like layers in
the module tree. A composite checkpoint (Gemma 3 and 4 `...ForConditionalGeneration`, LLaVA,
Qwen2.5-VL) registers its vision tower before its language model, so that first match was the vision
encoder. Every caller (steering, extraction, circuits, probes, J-Lens) then hooked a vision layer that a
text forward never runs: a steering edit measured 0.0 against 0.329 on the right layer (R2).

Every case comes from `docs/schemas/layer-resolution-cases.json`, the file miServer vendors, and is
built as a tiny model with random weights from its family's own config class. So the file, these
tests and both repos' resolvers are one source: change a case there first.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from src.ml import layer_discovery
from src.ml.layer_discovery import discover_transformer_structure
from tests.support.layer_cases import (
    case_ids,
    load_cases,
    text_input_ids,
    tiny_config,
    tiny_model,
)

CASES = load_cases()


# The helpers the fix adds are read off the module when a test runs, so this file collects on the base
# and its red there is the defect's own (a wrong path, a hook that never fires), not an ImportError.
def LayerResolutionError():  # noqa: N802 - stands in for the class
    return layer_discovery.LayerResolutionError


def text_hidden_size(config):
    return layer_discovery.text_hidden_size(config)


def transformer_stacks(model):
    return layer_discovery.transformer_stacks(model)
COMPOSITE = [c for c in CASES if c["config_class"] != "LlamaConfig"]


def _logits(model, case, hook=None):
    handle = None
    if hook is not None:
        module, fn = hook
        handle = module.register_forward_hook(fn)
    try:
        with torch.no_grad():
            return model(input_ids=text_input_ids(case)).logits.clone()
    finally:
        if handle is not None:
            handle.remove()


def _add(delta):
    def edit(module, inputs, output):
        if isinstance(output, tuple):
            return (output[0] + delta,) + tuple(output[1:])
        return output + delta
    return edit


class TestTheCasesFileIsWhatTheCodeResolves:
    """AC-4: the file cannot drift from the code."""

    def test_the_file_has_every_family_the_brief_names(self):
        families = {c["model_class"] for c in CASES}
        for expected in ("Gemma4ForConditionalGeneration", "Gemma3ForConditionalGeneration",
                         "Qwen2_5_VLForConditionalGeneration", "LlavaForConditionalGeneration",
                         "LlamaForCausalLM"):
            assert expected in families
        assert all({"family", "config_class", "model_class", "text_layers_path", "text_depth", "notes"}
                   <= set(c) for c in CASES)

    @pytest.mark.parametrize("case", CASES, ids=case_ids)
    def test_the_file_names_every_layer_stack_the_model_has(self, case):
        """`vision_layers_path` is the stack a first-match search took; it must exist, and be the only other."""
        stacks = [path for path, _ in transformer_stacks(tiny_model(case))]
        expected = [case["text_layers_path"]] + (
            [case["vision_layers_path"]] if "vision_layers_path" in case else [])
        assert sorted(stacks) == sorted(expected)

    @pytest.mark.parametrize("case", CASES, ids=case_ids)
    def test_text_depth_is_the_tiny_text_config_depth(self, case):
        text = tiny_config(case).get_text_config(decoder=True)
        assert text.num_hidden_layers == case["text_depth"]


class TestDiscoveryReturnsTheTextTower:
    """AC-1 (control C1: restore first-match-wins)."""

    @pytest.mark.parametrize("case", CASES, ids=case_ids)
    def test_the_path_and_depth_are_the_text_towers(self, case):
        model = tiny_model(case)
        structure = discover_transformer_structure(model)
        assert structure.layers_path == case["text_layers_path"]
        assert structure.num_layers == case["text_depth"]
        assert structure.layers_module is model.get_submodule(case["text_layers_path"])

    @pytest.mark.parametrize("case", CASES, ids=case_ids)
    def test_a_hook_on_layer_1_fires_once_on_a_text_forward(self, case):
        model = tiny_model(case)
        layer = discover_transformer_structure(model).layers_module[1]
        fired = []
        _logits(model, case, hook=(layer, lambda m, i, o: fired.append(m)))
        assert len(fired) == 1, f"layer 1 fired {len(fired)} times on a text-only forward"
        assert fired[0] is layer

    @pytest.mark.parametrize("case", CASES, ids=case_ids)
    def test_a_residual_edit_through_layer_1_moves_the_logits(self, case):
        """The R2 measurement in miniature: an edit at a layer the text forward never runs changes nothing."""
        model = tiny_model(case)
        layer = discover_transformer_structure(model).layers_module[1]
        hidden = model.config.get_text_config(decoder=True).hidden_size
        base = _logits(model, case)
        edited = _logits(model, case, hook=(layer, _add(torch.full((hidden,), 0.5))))
        unedited = _logits(model, case)
        assert torch.equal(base, unedited), "the edit hook must not outlive its forward"
        assert (edited - base).abs().max().item() > 1e-4

    def test_the_single_tower_control_resolves_exactly_as_before(self):
        llama = next(c for c in CASES if c["config_class"] == "LlamaConfig")
        structure = discover_transformer_structure(tiny_model(llama))
        assert (structure.layers_path, structure.num_layers, structure.attention_module,
                structure.mlp_module, structure.residual_norm_module) == (
            "model.layers", 3, "self_attn", "mlp", "post_attention_layernorm")

    @pytest.mark.parametrize("case", COMPOSITE, ids=case_ids)
    def test_the_embedding_owner_and_get_decoder_agree_on_5_20_2(self, case):
        """The brief asked which of the two works on the pinned transformers for each family: both do."""
        model = tiny_model(case)
        embedding = model.get_input_embeddings()
        decoder = model.get_decoder()
        text_stack = model.get_submodule(case["text_layers_path"])
        assert any(m is embedding for m in decoder.modules())
        assert any(m is text_stack for m in decoder.modules())


class TestADepthMismatchIsRefused:
    """AC-2 (control C2: drop the cross-check)."""

    @pytest.mark.parametrize("family", ["gemma3", "llama"])
    def test_a_text_config_that_disagrees_with_the_stack_refuses(self, family):
        case = next(c for c in CASES if c["family"] == family)
        model = copy.deepcopy(tiny_model(case))
        text = model.config.get_text_config(decoder=True)
        text.num_hidden_layers = case["text_depth"] + 1
        with pytest.raises(LayerResolutionError()) as refused:
            discover_transformer_structure(model)
        message = str(refused.value)
        assert case["text_layers_path"] in message
        assert f"{case['text_depth']} layers" in message
        assert f"num_hidden_layers={case['text_depth'] + 1}" in message

    def test_the_refusal_is_a_value_error_so_existing_callers_still_refuse(self):
        assert issubclass(LayerResolutionError(), ValueError)

    def test_a_model_without_an_integer_depth_is_not_refused(self):
        """A config that states no depth cannot disagree; only a stated, different number refuses."""
        case = next(c for c in CASES if c["family"] == "llama")
        model = copy.deepcopy(tiny_model(case))
        # A strict transformers config refuses `num_hidden_layers=None`, so stand in a config that states none.
        model.config = SimpleNamespace(model_type="llama")
        assert discover_transformer_structure(model).num_layers == case["text_depth"]


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = nn.Linear(4, 4)
        self.mlp = nn.Linear(4, 4)


class TestTwoStacksWithNoTextOwnerAreRefused:

    def test_two_stacks_and_no_way_to_tell_the_text_one_refuses(self):
        class TwoTowers(nn.Module):
            def __init__(self):
                super().__init__()
                self.left = nn.ModuleList([_Block(), _Block()])
                self.right = nn.ModuleList([_Block(), _Block(), _Block()])

        with pytest.raises(LayerResolutionError(), match="left.*right|right.*left"):
            discover_transformer_structure(TwoTowers())

    def test_a_stack_nested_inside_a_layer_is_not_a_second_tower(self):
        class Inner(_Block):
            def __init__(self):
                super().__init__()
                self.sub = nn.ModuleList([_Block()])

        class Flat(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([Inner(), Inner()])

        structure = discover_transformer_structure(Flat())
        assert (structure.layers_path, structure.num_layers) == ("layers", 2)


class TestDimensionsComeFromTheTextConfig:
    """AC-3 (control C3: read hidden_size from the top-level config again).

    A composite config on 5.20.2 carries no top-level `hidden_size`, so a test on a stock config agrees
    with a top-level read by construction. Each fixture here PLANTS a top-level width that differs from
    the text tower's, which is what an older or a hand-written composite config.json looks like.
    """

    def _planted(self):
        case = next(c for c in CASES if c["family"] == "gemma3")
        config = tiny_config(case)
        config.hidden_size = 48            # e.g. the vision tower's width, at the top level
        return config, config.get_text_config(decoder=True).hidden_size

    def test_text_hidden_size_reads_the_text_tower(self):
        config, text_width = self._planted()
        assert text_width == 32
        assert text_hidden_size(config) == 32

    def test_text_hidden_size_reads_a_recorded_dict_too(self):
        assert text_hidden_size({"hidden_size": 48, "text_config": {"hidden_size": 32}}) == 32
        assert text_hidden_size({"d_model": 16}) == 16
        assert text_hidden_size({"hidden_size": True}) is None
        assert text_hidden_size(None) is None

    def test_a_flat_config_keeps_its_own_width(self):
        llama = next(c for c in CASES if c["family"] == "llama")
        assert text_hidden_size(tiny_config(llama)) == 32

    def test_the_probe_runs_memory_forecast_uses_the_text_width(self, monkeypatch):
        from types import SimpleNamespace

        from src.services import probe_run_memory as memory

        config, _ = self._planted()
        seen = []
        monkeypatch.setattr(memory, "forecast_memory", lambda **kw: seen.append(kw["d_model"]) or {"checked": True})
        monkeypatch.setattr(memory, "_card_state", lambda model: (None, None))
        monkeypatch.setattr(memory, "other_reserves_bytes", lambda db: 0)
        monkeypatch.setattr(memory, "template_prefix_skipped", lambda rendered, scope: False)
        monkeypatch.setattr(memory, "_with_prefix_skip", lambda answer, skipped: answer)
        import src.services.probe_monitor_capture as capture
        monkeypatch.setattr(capture, "count_scored_tokens", lambda rendered, scope: (8, [0, 4, 8]))
        answer = memory.forecast_for_run(None, SimpleNamespace(config=config), [], [0], [1], scope="all",
                                         swept_layers=1, top_n_layers=1, storage_bytes=2)
        assert answer == {"checked": True}
        assert seen == [32]

    def test_the_sae_training_width_check_uses_the_text_width(self):
        from src.workers.training_tasks import loaded_model_hidden_dim

        config, _ = self._planted()
        assert loaded_model_hidden_dim(config) == 32

    def test_jlens_sizing_reads_the_checkpoints_text_width(self, tmp_path):
        import json
        from types import SimpleNamespace
        from unittest.mock import patch

        from src.services import jlens_model_registry as registry

        (tmp_path / "config.json").write_text(json.dumps({
            "model_type": "gemma3", "hidden_size": 48,
            "text_config": {"hidden_size": 32, "num_hidden_layers": 3, "vocab_size": 128,
                            "num_attention_heads": 2},
        }))
        with patch.object(registry, "locate_weights", return_value=("org/model", str(tmp_path))), \
             patch("src.services.analysis_service.resolve_snapshot_dir", return_value=tmp_path):
            found = registry._dims_from_checkpoint(SimpleNamespace(id="m_x", file_path=str(tmp_path)))
        assert found["hidden_size"] == 32
        assert found["num_hidden_layers"] == 3

    def test_the_sae_training_task_calls_it_on_the_loaded_config(self):
        """The CALLER is pinned, not only the helper (memory `mutation-survivors-cluster-on-callers`)."""
        import ast
        import inspect

        from src.workers import training_tasks

        tree = ast.parse(inspect.getsource(training_tasks))
        task = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "train_sae_task")
        calls = [n for n in ast.walk(task) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name) and n.func.id == "loaded_model_hidden_dim"]
        assert len(calls) == 1
        assert [ast.unparse(a) for a in calls[0].args] == ["model_config"]
        reads = [n for n in ast.walk(task) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                 and n.func.id == "getattr" and len(n.args) >= 2 and ast.unparse(n.args[0]) == "model_config"
                 and isinstance(n.args[1], ast.Constant) and n.args[1].value in ("hidden_size", "d_model")]
        assert reads == [], "the task must not read a width off the top-level config"
