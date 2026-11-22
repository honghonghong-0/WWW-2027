#!/usr/bin/env python3
"""Merge UniCRS generated response templates for train/valid only."""

import argparse
import json
from pathlib import Path


def read_generated(path: Path):
    """Read generator JSONL, recovering raw newlines inside model strings."""
    rows, pending = [], ""
    for line in path.read_text(encoding="utf-8").splitlines():
        pending = f"{pending}\\n{line}" if pending else line
        try:
            rows.append(json.loads(pending))
            pending = ""
        except json.JSONDecodeError:
            continue
    if pending:
        raise ValueError(f"unrecoverable generated JSON in {path}: {pending[:120]!r}")
    return rows


def merge_split(raw_path: Path, generated_path: Path, output_path: Path) -> None:
    generated = read_generated(generated_path)
    generated_index = 0

    with raw_path.open(encoding="utf-8") as raw_file, output_path.open("w", encoding="utf-8") as out_file:
        for line in raw_file:
            row = json.loads(line)
            if len(row["context"]) == 1 and row["context"][0] == "":
                row["resp"] = ""
            else:
                if generated_index >= len(generated):
                    raise ValueError(f"generated file ended early: {generated_path}")
                prediction = generated[generated_index]["pred"]
                row["resp"] = prediction.split("System: ")[-1] if "<movie>" in prediction else ""
                generated_index += 1
            out_file.write(json.dumps(row, ensure_ascii=False) + "\n")

    if generated_index != len(generated):
        raise ValueError(
            f"unused generated rows for {generated_path}: {len(generated) - generated_index}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--generated-dir", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "valid"):
        merge_split(
            args.raw_dir / f"{split}_data_processed.jsonl",
            args.generated_dir / f"{args.prefix}_{split}.jsonl",
            args.output_dir / f"{split}_data_processed.jsonl",
        )


if __name__ == "__main__":
    main()
