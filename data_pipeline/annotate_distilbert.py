#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import os
import pickle
import re
from typing import List, Tuple

import numpy as np

try:
    import nltk
except Exception:
    nltk = None

try:
    import torch
    from transformers import DistilBertModel, DistilBertTokenizerFast
except Exception:
    torch = None
    DistilBertModel = None
    DistilBertTokenizerFast = None


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


def _embed_sentences(model, tokenizer, sentences: List[str], batch_size: int, device: str):
    embeddings = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i : i + batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
            # mean pool over tokens
            last_hidden = out.last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        embeddings.append(pooled.cpu().numpy())
    return np.vstack(embeddings)


def _cosine_matrix(x: np.ndarray) -> np.ndarray:
    x_norm = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
    sim = np.matmul(x_norm, x_norm.T)
    # map [-1, 1] -> [0, 1]
    return (sim + 1.0) / 2.0


def _build_labels_for_rows(
    rows, model, tokenizer, batch_size: int, device: str, max_sent_len: int, max_sents: int
):
    instances = []
    for id_, content, _abstract in rows:
        sents = _sent_split(content)
        # enforce limits like training
        sents = [s for s in sents if len(s.split()) <= max_sent_len]
        if len(sents) == 0:
            continue
        if len(sents) > max_sents:
            sents = sents[:max_sents]

        emb = _embed_sentences(model, tokenizer, sents, batch_size, device)
        mat = _cosine_matrix(emb)
        np.fill_diagonal(mat, 1.0)

        instances.append([id_, None, None, mat])
    return instances


def _write_pickle(instances, path: str):
    out_dir = os.path.dirname(path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
    with open(path, "wb") as f:
        pickle.dump([instances], f)


def main():
    if DistilBertTokenizerFast is None or DistilBertModel is None or torch is None:
        raise SystemExit("transformers/torch not available in environment")

    parser = argparse.ArgumentParser(
        description="Annotate sentence relation matrices with vanilla DistilBERT."
    )
    parser.add_argument("--train_story_dir", required=True, help="Path to train .story directory")
    parser.add_argument("--dev_story_dir", required=True, help="Path to dev .story directory")
    parser.add_argument(
        "--output_dir",
        default="labeled_data/train_data_0_5000",
        help="Output directory for labeled pickles",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for encoding")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--max_sentence_length", type=int, default=50)
    parser.add_argument("--max_content_length", type=int, default=50)
    args = parser.parse_args()

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    model = DistilBertModel.from_pretrained("distilbert-base-uncased")
    model.eval()
    model.to(args.device)

    train_rows = _read_story_dir(args.train_story_dir)
    dev_rows = _read_story_dir(args.dev_story_dir)

    train_instances = _build_labels_for_rows(
        train_rows,
        model,
        tokenizer,
        args.batch_size,
        args.device,
        args.max_sentence_length,
        args.max_content_length,
    )
    dev_instances = _build_labels_for_rows(
        dev_rows,
        model,
        tokenizer,
        args.batch_size,
        args.device,
        args.max_sentence_length,
        args.max_content_length,
    )

    train_path = os.path.join(args.output_dir, "train_labeled_0_5000.p")
    dev_path = os.path.join(args.output_dir, "valid_labeled.p")

    _write_pickle(train_instances, train_path)
    _write_pickle(dev_instances, dev_path)

    print(f"Wrote {len(train_instances)} train instances to {train_path}")
    print(f"Wrote {len(dev_instances)} dev instances to {dev_path}")


if __name__ == "__main__":
    main()
