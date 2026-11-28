"""The shared layer-resolution cases (`docs/schemas/layer-resolution-cases.json`) as tiny models.

Each case names a transformers config class, a model class and the keyword arguments of a CPU-sized
copy with random weights (`tiny_config`). Nothing is downloaded: the model is built from the config
class, so a test of the resolver runs the real module tree of each family (ENH-011 X-1, #2).
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

REPO = Path(__file__).resolve().parents[3]
CASES_PATH = REPO / "docs" / "schemas" / "layer-resolution-cases.json"


def load_cases() -> List[Dict[str, Any]]:
    return json.loads(CASES_PATH.read_text())["cases"]


def case_ids(case: Dict[str, Any]) -> str:
    return case["family"]


def tiny_config(case: Dict[str, Any]):
    import transformers

    config_class = getattr(transformers, case["config_class"])
    return config_class(**case["tiny_config"])


@lru_cache(maxsize=None)
def _built(family: str):
    import torch
    import transformers

    case = next(c for c in load_cases() if c["family"] == family)
    torch.manual_seed(0)
    model_class = getattr(transformers, case["model_class"])
    model = model_class(tiny_config(case))
    model.eval()
    return model


def tiny_model(case: Dict[str, Any]):
    """The case's tiny model, built once per process (eval mode, seed 0). Callers must not mutate it."""
    return _built(case["family"])


def text_input_ids(case: Dict[str, Any], length: int = 6):
    """A text-only prompt: ids well inside the tiny vocabulary and clear of every special image token."""
    import torch

    return torch.arange(3, 3 + length).unsqueeze(0)
