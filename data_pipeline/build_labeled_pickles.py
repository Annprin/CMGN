#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
import os
import pickle
from typing import List, Dict, Any


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _build_instances(rows: List[Dict[str, Any]], label_key: str) -> List[List[Any]]:
    instances = []
    for row in rows:
        if "id" not in row:
            raise ValueError("Each row must contain 'id'")
        if label_key not in row:
            raise ValueError(f"Each row must contain '{label_key}'")
        # Minimal 4-field format; process_bert_labeled.py can be adjusted accordingly.
        instances.append([str(row["id"]), None, None, row[label_key]])
    return instances


def _write_pickle(instances: List[List[Any]], path: str):
    out_dir = os.path.dirname(path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
    with open(path, "wb") as f:
        pickle.dump([instances], f)


def main():
    parser = argparse.ArgumentParser(
        description="Build train_labeled_0_5000.p and valid_labeled.p from JSONL."
    )
    parser.add_argument("--train_jsonl", required=True, help="Path to train JSONL")
    parser.add_argument("--dev_jsonl", required=True, help="Path to dev JSONL")
    parser.add_argument("--label_key", default="label", help="Key for label field")
    parser.add_argument(
        "--output_dir",
        default="labeled_data/train_data_0_5000",
        help="Output directory for labeled pickles",
    )
    args = parser.parse_args()

    train_rows = _read_jsonl(args.train_jsonl)
    dev_rows = _read_jsonl(args.dev_jsonl)

    train_instances = _build_instances(train_rows, args.label_key)
    dev_instances = _build_instances(dev_rows, args.label_key)

    train_path = os.path.join(args.output_dir, "train_labeled_0_5000.p")
    dev_path = os.path.join(args.output_dir, "valid_labeled.p")

    _write_pickle(train_instances, train_path)
    _write_pickle(dev_instances, dev_path)

    print(f"Wrote {len(train_instances)} train instances to {train_path}")
    print(f"Wrote {len(dev_instances)} dev instances to {dev_path}")


if __name__ == "__main__":
    main()
