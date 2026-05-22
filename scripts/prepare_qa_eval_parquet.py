#!/usr/bin/env python3
"""Convert local GPQA/StrategyQA files to verl-style QA parquet files.

Expected inputs in the same directory by default:
  - gpqa_main.csv
  - strategyqa_test.json

Outputs:
  - qa_parquet/gpqa_main_test.parquet
  - qa_parquet/strategyqa_test.parquet

Each row follows the format consumed by verl's RL dataset:
  data_source, prompt, ability, reward_model, extra_info
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

import pandas as pd


CHOICE_LETTERS = ("A", "B", "C", "D")


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).replace("\r\n", "\n").replace("\r", "\n").split())


def first_non_empty(row: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = clean_text(row.get(key, ""))
        if value:
            return value
    return ""


def make_chat_prompt(content: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": content}]


def build_gpqa_prompt(question: str, choices: list[tuple[str, str]]) -> str:
    options = "\n".join(f"({letter}) {text}" for letter, text in choices)
    return (
        f"{question}\n\n"
        f"{options}\n\n"
        r"Please put your final answer option (A, B, C, or D) in \boxed{}."
    )


def convert_gpqa_main(csv_path: Path, seed: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rng = random.Random(seed)

    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            question = first_non_empty(row, "Extra Revised Question", "Question", "Pre-Revision Question")
            correct = first_non_empty(row, "Extra Revised Correct Answer", "Correct Answer", "Pre-Revision Correct Answer")
            incorrects = [
                first_non_empty(row, "Extra Revised Incorrect Answer 1", "Incorrect Answer 1", "Pre-Revision Incorrect Answer 1"),
                first_non_empty(row, "Extra Revised Incorrect Answer 2", "Incorrect Answer 2", "Pre-Revision Incorrect Answer 2"),
                first_non_empty(row, "Extra Revised Incorrect Answer 3", "Incorrect Answer 3", "Pre-Revision Incorrect Answer 3"),
            ]
            incorrects = [answer for answer in incorrects if answer]

            if not question or not correct or len(incorrects) != 3:
                continue

            options = [("correct", correct), *[(f"incorrect_{i + 1}", answer) for i, answer in enumerate(incorrects)]]
            rng.shuffle(options)

            labeled_choices = [(CHOICE_LETTERS[i], answer) for i, (_, answer) in enumerate(options)]
            gold_idx = next(i for i, (kind, _) in enumerate(options) if kind == "correct")
            gold_letter = CHOICE_LETTERS[gold_idx]

            prompt = build_gpqa_prompt(question, labeled_choices)
            rows.append(
                {
                    "data_source": "my_data/gpqa_main",
                    "prompt": make_chat_prompt(prompt),
                    "ability": "qa",
                    "reward_model": {"style": "rule", "ground_truth": gold_letter},
                    "extra_info": {
                        "split": "test",
                        "index": idx,
                        "record_id": clean_text(row.get("Record ID", "")),
                        "question": question,
                        "answer": gold_letter,
                        "correct_answer_text": correct,
                        "choices": {letter: text for letter, text in labeled_choices},
                        "subdomain": clean_text(row.get("Subdomain", "")),
                        "high_level_domain": clean_text(row.get("High-level domain", "")),
                    },
                }
            )

    return pd.DataFrame(rows)


def load_strategyqa_items(json_path: Path) -> list[dict[str, Any]]:
    with json_path.open(encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("examples", "data", "test", "validation", "train"):
            value = data.get(key)
            if isinstance(value, list):
                return value
        for value in data.values():
            if isinstance(value, list):
                return value
    raise ValueError(f"Unsupported StrategyQA JSON structure in {json_path}")


def normalize_yes_no(answer: Any) -> str:
    if isinstance(answer, bool):
        return "yes" if answer else "no"
    text = clean_text(answer).lower()
    if text in {"true", "1", "yes", "y"}:
        return "yes"
    if text in {"false", "0", "no", "n"}:
        return "no"
    raise ValueError(f"Cannot normalize StrategyQA answer: {answer!r}")


def build_strategyqa_prompt(question: str) -> str:
    return (
        f"{question}\n\n"
        r"Please put your final yes/no answer in \boxed{}."
    )


def convert_strategyqa(json_path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for idx, item in enumerate(load_strategyqa_items(json_path)):
        question = clean_text(item.get("question", ""))
        if not question:
            continue
        gold = normalize_yes_no(item.get("answer"))
        prompt = build_strategyqa_prompt(question)
        rows.append(
            {
                "data_source": "my_data/strategyqa",
                "prompt": make_chat_prompt(prompt),
                "ability": "qa",
                "reward_model": {"style": "rule", "ground_truth": gold},
                "extra_info": {
                    "split": "test",
                    "index": idx,
                    "qid": clean_text(item.get("qid", "")),
                    "term": clean_text(item.get("term", "")),
                    "description": clean_text(item.get("description", "")),
                    "question": question,
                    "answer": gold,
                    "original_answer": item.get("answer"),
                },
            }
        )
    return pd.DataFrame(rows)


def write_jsonl(df: pd.DataFrame, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in df.to_dict(orient="records"):
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "qa_parquet")
    parser.add_argument("--seed", type=int, default=1, help="Seed used to shuffle GPQA options deterministically.")
    parser.add_argument("--write-jsonl", action="store_true", help="Also write JSONL previews next to parquet files.")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    gpqa_df = convert_gpqa_main(args.input_dir / "gpqa_main.csv", seed=args.seed)
    strategyqa_df = convert_strategyqa(args.input_dir / "strategyqa_test.json")

    outputs = [
        (gpqa_df, args.output_dir / "gpqa_main_test.parquet"),
        (strategyqa_df, args.output_dir / "strategyqa_test.parquet"),
    ]

    for df, path in outputs:
        df.to_parquet(path, index=False)
        if args.write_jsonl:
            write_jsonl(df, path.with_suffix(".jsonl"))
        print(f"wrote {path} rows={len(df)}")
        if len(df):
            sample = df.iloc[0].to_dict()
            print(
                json.dumps(
                    {
                        "data_source": sample["data_source"],
                        "ground_truth": sample["reward_model"]["ground_truth"],
                        "prompt": sample["prompt"][0]["content"][:500],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )


if __name__ == "__main__":
    main()
