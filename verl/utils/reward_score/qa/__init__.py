"""Rule-based reward functions for QA-style RLVR datasets.

Predictions must put the final answer option inside the last ``\boxed{...}``, matching
the math-style RLVR format. Current QA datasets are choice tasks and expect
answers such as ``\boxed{A}``.
"""

from __future__ import annotations

import json
import re
import string
import traceback
from typing import Any

_BOXED_RE = re.compile(r"\\boxed\s*\{")
_CHOICE_DATA_SOURCE_MARKERS = ("logiqa", "prontoqa", "proverqa", "logical_deduction", "lsat_ar", "lsat-ar", "csqa", "openbookqa", "gpqa", "commonsense", "qa")


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        return " ".join(_stringify(v) for v in value)
    if isinstance(value, dict):
        for key in ("answer", "label", "target", "ground_truth", "gold", "correct"):
            if key in value:
                return _stringify(value[key])
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _maybe_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text or text[0] not in "[{\"":
        return value
    try:
        return json.loads(text)
    except Exception:
        return value


def _clean_text(text: Any) -> str:
    text = _stringify(_maybe_json(text)).strip()
    text = text.replace("\u3000", " ")
    return re.sub(r"\s+", " ", text)


def _normalize_freeform(text: Any) -> str:
    text = _clean_text(text).lower()
    text = text.strip().strip(string.punctuation + "，。；：、（）()[]{}<>\"'")
    return re.sub(r"\s+", " ", text)


def _extract_last_boxed(text: Any) -> str | None:
    text = _stringify(text)
    matches = list(_BOXED_RE.finditer(text))
    if not matches:
        return None
    start = matches[-1].end()
    depth = 1
    i = start
    while i < len(text):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i].strip()
        i += 1
    return None



def _normalize_choice(text: Any) -> str | None:
    stripped = _clean_text(text).strip().strip("，。；：、()[]{}<>\"'")
    if len(stripped) == 1 and stripped.upper() in "ABCDEFGH":
        return stripped.upper()
    explicit = re.search(r"^(?:option|choice|answer|答案|选项)?\s*(?:is|:|：)?\s*\(?([A-H])\)?\.?$", stripped, re.IGNORECASE)
    if explicit:
        return explicit.group(1).upper()
    prefixed = re.search(r"^\(?([A-H])\)?(?:\.|:|：)?(?:\s|$)", stripped, re.IGNORECASE)
    if prefixed:
        return prefixed.group(1).upper()
    return None


def _extract_prediction(data_source: str, solution_str: str) -> tuple[str | None, str]:
    boxed = _extract_last_boxed(solution_str)
    source = data_source.lower()
    if boxed is None:
        return None, "boxed_missing"
    if any(name in source for name in _CHOICE_DATA_SOURCE_MARKERS):
        return _normalize_choice(boxed), "choice"
    return _normalize_freeform(boxed), "freeform"


def _normalize_ground_truth(data_source: str, ground_truth: Any) -> tuple[str, str]:
    source = data_source.lower()
    raw = _maybe_json(ground_truth)
    choice_gt = _normalize_choice(raw)
    if choice_gt is not None and any(name in source for name in _CHOICE_DATA_SOURCE_MARKERS):
        return choice_gt, "choice"
    return _normalize_freeform(raw), "freeform"


def compute_score(data_source: str, solution_str: str, ground_truth: Any, extra_info: dict | None = None) -> dict:
    pred, pred_kind = _extract_prediction(data_source, solution_str)
    gt, gt_kind = _normalize_ground_truth(data_source, ground_truth)
    format_score = 1.0 if pred is not None else 0.0
    is_correct = bool(pred is not None and gt and pred == gt)
    return {
        "score": 1.0 if is_correct else 0.0,
        "format_score": format_score,
        "acc": bool(is_correct),
        "pred": "" if pred is None else pred,
        "extracted_gt": gt,
        "pred_kind": pred_kind,
        "gt_kind": gt_kind,
    }


def reward_func(data_source, solution_str, ground_truth, extra_info=None, sandbox_fusion_url=None, concurrent_semaphore=None):
    try:
        return compute_score(str(data_source), solution_str, ground_truth, extra_info=extra_info or {})
    except Exception as exc:
        print(f"[ERROR] QA reward failed for {data_source}: {exc}")
        traceback.print_exc()
        return {
            "score": 0.0,
            "format_score": 0.0,
            "acc": False,
            "pred": "",
            "extracted_gt": _clean_text(ground_truth),
            "error": str(exc),
        }
