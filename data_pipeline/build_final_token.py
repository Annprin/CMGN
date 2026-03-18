#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
import os
import pickle
import random
import re
from typing import List, Dict, Any, Tuple

try:
    import nltk
except Exception:
    nltk = None


def _sent_split(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    if nltk is not None:
        try:
            tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
            return tokenizer.tokenize(text)
        except Exception:
            pass
    # Fallback: simple split
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p for p in parts if p]


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _read_story_file(path: str) -> Tuple[str, str]:
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            lines.append(line.strip())

    # Lowercase to match handle_testing_data.py
    lines = [line.lower() for line in lines]

    # Fix missing periods like in handle_testing_data.py
    end_tokens = {".", "!", "?", "...", "'", "`", '"', ")", "\u2019", "\u201d"}
    fixed = []
    for line in lines:
        if "@highlight" in line or line == "":
            fixed.append(line)
            continue
        if line[-1] in end_tokens:
            fixed.append(line)
        else:
            fixed.append(line + " .")

    article_lines = []
    highlights = []
    next_is_highlight = False
    for line in fixed:
        if line == "":
            continue
        if line.startswith("@highlight"):
            next_is_highlight = True
            continue
        if next_is_highlight:
            highlights.append(line)
        else:
            article_lines.append(line)

    content = " ".join(article_lines)
    abstract = " ".join(highlights)
    return content, abstract


def _read_story_dir(path: str) -> List[Dict[str, Any]]:
    rows = []
    for name in os.listdir(path):
        if not name.endswith(".story"):
            continue
        file_path = os.path.join(path, name)
        id_ = name[: name.find(".story")]
        content, abstract = _read_story_file(file_path)
        rows.append({"id": id_, "content": content, "abstract": abstract})
    return rows


def _normalize_row(row: Dict[str, Any]) -> Tuple[str, str, str]:
    if "id" not in row:
        raise ValueError("Each row must contain 'id'")
    content = row.get("content", "")
    abstract = row.get("abstract", "")
    return str(row["id"]), str(content), str(abstract)


def _build_items(rows: List[Dict[str, Any]]) -> List[List[Any]]:
    items = []
    for row in rows:
        id_, content, abstract = _normalize_row(row)
        content_sents = _sent_split(content)
        abstract_sents = _sent_split(abstract)
        # Keep the same values for *_for_token as expected by process_bert_labeled.py
        items.append([id_, content_sents, abstract_sents, content_sents, abstract_sents])
    return items


def _split_rows(rows: List[Dict[str, Any]], dev_ratio: float, seed: int):
    rng = random.Random(seed)
    rows = list(rows)
    rng.shuffle(rows)
    cut = int(len(rows) * (1 - dev_ratio))
    return rows[:cut], rows[cut:]


def main():
    parser = argparse.ArgumentParser(
        description="Build final_token.p from JSONL or .story files."
    )
    parser.add_argument("--train_jsonl", help="Path to train JSONL")
    parser.add_argument("--dev_jsonl", help="Path to dev JSONL (optional)")
    parser.add_argument("--input_jsonl", help="Single JSONL to split")
    parser.add_argument("--train_story_dir", help="Path to train .story directory")
    parser.add_argument("--dev_story_dir", help="Path to dev .story directory")
    parser.add_argument("--story_dir", help="Single .story directory to split")
    parser.add_argument("--dev_ratio", type=float, default=0.1, help="Dev split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed for split")
    parser.add_argument(
        "--output",
        default="generate_training_data/final_dataset/final_token.p",
        help="Output pickle path",
    )
    args = parser.parse_args()

    if args.story_dir:
        rows = _read_story_dir(args.story_dir)
        train_rows, dev_rows = _split_rows(rows, args.dev_ratio, args.seed)
    elif args.train_story_dir or args.dev_story_dir:
        if not args.train_story_dir or not args.dev_story_dir:
            raise SystemExit("Provide --train_story_dir and --dev_story_dir together")
        train_rows = _read_story_dir(args.train_story_dir)
        dev_rows = _read_story_dir(args.dev_story_dir)
    elif args.input_jsonl:
        rows = _read_jsonl(args.input_jsonl)
        train_rows, dev_rows = _split_rows(rows, args.dev_ratio, args.seed)
    else:
        if not args.train_jsonl or not args.dev_jsonl:
            raise SystemExit(
                "Provide JSONL paths or .story directories for train/dev"
            )
        train_rows = _read_jsonl(args.train_jsonl)
        dev_rows = _read_jsonl(args.dev_jsonl)

    train_items = _build_items(train_rows)
    dev_items = _build_items(dev_rows)

    out_dir = os.path.dirname(args.output)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    with open(args.output, "wb") as f:
        pickle.dump([train_items, dev_items], f)

    print(f"Wrote {len(train_items)} train + {len(dev_items)} dev items to {args.output}")


if __name__ == "__main__":
    main()
