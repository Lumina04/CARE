"""Reward functions for knights-and-knaves logic data.

The training score follows the Logic-RL KK reward shape: a format reward plus an
answer reward. Format validation only requires a non-empty final
``<answer>...</answer>`` block because the prompt already supplies ``<think>``.
"""

from __future__ import annotations

import re
import traceback
from typing import Any, Optional

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
_ROLE_RE = re.compile(
    r"\b([A-Za-z][A-Za-z'_-]*)\b\s+(?:is|are)\s+(?:a\s+|an\s+)?\b(knight|knave)\b",
    re.IGNORECASE,
)
_CANONICAL_PART_RE = re.compile(r"^([A-Za-z][A-Za-z'_-]*)=(knight|knave)$", re.IGNORECASE)


def extract_answer_block(solution_str: Any) -> Optional[str]:
    text = "" if solution_str is None else str(solution_str)
    matches = list(_ANSWER_RE.finditer(text))
    if not matches:
        return None
    answer = matches[-1].group(1).strip()
    return answer or None


def _clean_name(name: Any) -> str:
    return re.sub(r"\s+", " ", str(name).strip()).lower()


def _ground_truth_text(ground_truth: Any) -> str:
    if isinstance(ground_truth, dict):
        for key in ("solution_text_format", "answer", "ground_truth", "target"):
            if key in ground_truth:
                return "" if ground_truth[key] is None else str(ground_truth[key])
    return "" if ground_truth is None else str(ground_truth)


def parse_roles(text: Any, expected_names: Optional[list[str]] = None) -> Optional[dict[str, str]]:
    if text is None:
        return None
    raw = str(text).strip()
    if not raw:
        return None

    canonical_roles: dict[str, str] = {}
    canonical_parts = [part.strip() for part in raw.split("|") if part.strip()]
    if canonical_parts:
        for part in canonical_parts:
            match = _CANONICAL_PART_RE.match(part)
            if not match:
                canonical_roles = {}
                break
            canonical_roles[_clean_name(match.group(1))] = match.group(2).lower()
        if canonical_roles:
            if expected_names is not None:
                expected = [_clean_name(name) for name in expected_names]
                if any(name not in canonical_roles for name in expected):
                    return None
                return {name: canonical_roles[name] for name in expected}
            return canonical_roles

    found: dict[str, str] = {}
    for match in _ROLE_RE.finditer(raw):
        found[_clean_name(match.group(1))] = match.group(2).lower()

    if expected_names is None:
        return found or None

    expected = [_clean_name(name) for name in expected_names]
    if any(name not in found for name in expected):
        return None
    return {name: found[name] for name in expected}


def canonicalize_roles(roles: Optional[dict[str, str]]) -> Optional[str]:
    if not roles:
        return None
    return "|".join(f"{name}={roles[name]}" for name in sorted(roles))


def canonicalize_ground_truth(ground_truth: Any) -> Optional[str]:
    return canonicalize_roles(parse_roles(_ground_truth_text(ground_truth)))


def canonicalize_answer(answer_text: Any, ground_truth: Any = None) -> Optional[str]:
    expected_names = None
    if ground_truth is not None:
        gt_roles = parse_roles(_ground_truth_text(ground_truth))
        if gt_roles:
            expected_names = list(gt_roles.keys())
    return canonicalize_roles(parse_roles(answer_text, expected_names=expected_names))


def compute_score(data_source: str, solution_str: str, ground_truth: Any, extra_info: dict | None = None) -> dict:
    answer_text = extract_answer_block(solution_str)
    format_correct = answer_text is not None
    format_score = 1.0 if format_correct else -1.0

    pred = canonicalize_answer(answer_text, ground_truth=ground_truth) if format_correct else None
    gt = canonicalize_ground_truth(ground_truth)
    is_correct = bool(pred is not None and gt is not None and pred == gt)

    if not format_correct:
        answer_score = -2.0
    elif pred is None:
        answer_score = -2.0
    elif is_correct:
        answer_score = 2.0
    else:
        answer_score = -1.5

    total_score = format_score + answer_score
    return {
        "score": total_score,
        "format_score": format_score,
        "answer_score": answer_score,
        "result_score": 1.0 if is_correct else 0.0,
        "acc": bool(is_correct),
        "pred": "" if pred is None else pred,
        "extracted_gt": "" if gt is None else gt,
    }


def reward_func(data_source, solution_str, ground_truth, extra_info=None, sandbox_fusion_url=None, concurrent_semaphore=None):
    try:
        return compute_score(str(data_source), solution_str, ground_truth, extra_info=extra_info or {})
    except Exception as exc:
        print(f"[ERROR] KK reward failed for {data_source}: {exc}")
        traceback.print_exc()
        return {
            "score": -3.0,
            "format_score": -1.0,
            "answer_score": -2.0,
            "result_score": 0.0,
            "acc": False,
            "pred": "",
            "extracted_gt": "",
            "error": str(exc),
        }
