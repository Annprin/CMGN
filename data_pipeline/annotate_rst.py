#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import os
import pickle
import re
import sys
from typing import List, Tuple

try:
    import nltk
except Exception:
    nltk = None

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from data_pipeline.rst_labels import build_labels_with_rst


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
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p for p in parts if p]


def _read_story_file(path: str) -> Tuple[str, str]:
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            lines.append(line.strip())

    lines = [line.lower() for line in lines]

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


def _read_story_dir(path: str):
    rows = []
    for name in os.listdir(path):
        if not name.endswith(".story"):
            continue
        file_path = os.path.join(path, name)
        id_ = name[: name.find(".story")]
        content, abstract = _read_story_file(file_path)
        rows.append((id_, content, abstract))
    return rows


def _build_items_for_rst(rows, max_sent_len: int, max_sents: int):
    items = []
    for id_, content, _abstract in rows:
        sents = _sent_split(content)
        sents = [s for s in sents if len(s.split()) <= max_sent_len]
        if len(sents) == 0:
            continue
        if len(sents) > max_sents:
            sents = sents[:max_sents]
        # build minimal item with sentence list at index 7
        items.append([id_, None, None, None, None, None, None, sents])
    return items


def _write_pickle(instances, path: str):
    out_dir = os.path.dirname(path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
    with open(path, "wb") as f:
        pickle.dump([instances], f)


def main():
    parser = argparse.ArgumentParser(
        description="Annotate sentence relation matrices with RST (binary labels)."
    )
    parser.add_argument("--train_story_dir", required=True, help="Path to train .story directory")
    parser.add_argument("--dev_story_dir", required=True, help="Path to dev .story directory")
    parser.add_argument(
        "--output_dir",
        default="labeled_data/train_data_0_5000",
        help="Output directory for labeled pickles",
    )
    parser.add_argument("--max_sentence_length", type=int, default=50)
    parser.add_argument("--max_content_length", type=int, default=50)
    parser.add_argument(
        "--rst_parser",
        default="isanlp",
        help="RST parser backend (isanlp|auto|rstparser|rst_parser)",
    )
    parser.add_argument(
        "--hf_model_name",
        default="tchewik/isanlp_rst_v3",
        help="IsaNLP HF model name",
    )
    parser.add_argument(
        "--hf_model_version",
        default="gumrrg",
        help="IsaNLP HF model version",
    )
    parser.add_argument(
        "--cuda_device",
        type=int,
        default=-1,
        help="CUDA device for IsaNLP (-1 for CPU)",
    )
    args = parser.parse_args()

    train_rows = _read_story_dir(args.train_story_dir)
    dev_rows = _read_story_dir(args.dev_story_dir)

    train_items = _build_items_for_rst(
        train_rows, args.max_sentence_length, args.max_content_length
    )
    dev_items = _build_items_for_rst(
        dev_rows, args.max_sentence_length, args.max_content_length
    )

    parser_kwargs = {
        "hf_model_name": args.hf_model_name,
        "hf_model_version": args.hf_model_version,
        "cuda_device": args.cuda_device,
    }
    train_labels = build_labels_with_rst(
        train_items, parser_name=args.rst_parser, parser_kwargs=parser_kwargs
    )
    dev_labels = build_labels_with_rst(
        dev_items, parser_name=args.rst_parser, parser_kwargs=parser_kwargs
    )

    train_instances = [[doc_id, None, None, mat] for doc_id, mat in train_labels]
    dev_instances = [[doc_id, None, None, mat] for doc_id, mat in dev_labels]

    train_path = os.path.join(args.output_dir, "train_labeled_0_5000.p")
    dev_path = os.path.join(args.output_dir, "valid_labeled.p")

    _write_pickle(train_instances, train_path)
    _write_pickle(dev_instances, dev_path)

    print(f"Wrote {len(train_instances)} train instances to {train_path}")
    print(f"Wrote {len(dev_instances)} dev instances to {dev_path}")


if __name__ == "__main__":
    main()
