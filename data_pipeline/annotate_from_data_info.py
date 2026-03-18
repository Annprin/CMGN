#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import os
import pickle
from typing import List

import numpy as np

try:
    import torch
    from transformers import DistilBertModel, DistilBertTokenizerFast
except Exception:
    torch = None
    DistilBertModel = None
    DistilBertTokenizerFast = None


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
            last_hidden = out.last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        embeddings.append(pooled.cpu().numpy())
    return np.vstack(embeddings)


def _cosine_matrix(x: np.ndarray) -> np.ndarray:
    x_norm = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
    sim = np.matmul(x_norm, x_norm.T)
    return (sim + 1.0) / 2.0


def _annotate_instances(instances, model, tokenizer, batch_size: int, device: str):
    updated = []
    for inst in instances:
        # Expected format:
        # [id, contents_padding, contents_mask, abstracts_id, label,
        #  contents_id_length, seq_mask, content_sents, abstract_sents]
        if len(inst) < 9:
            raise ValueError("Unexpected instance format in data_info.p")

        content_sents = inst[7]
        content_len = inst[5]
        if content_len is None:
            content_len = len(content_sents)
        content_sents = content_sents[:content_len]

        if len(content_sents) == 0:
            updated.append(inst)
            continue

        emb = _embed_sentences(model, tokenizer, content_sents, batch_size, device)
        mat = _cosine_matrix(emb)
        np.fill_diagonal(mat, 1.0)

        new_inst = list(inst)
        new_inst[4] = mat
        updated.append(new_inst)
    return updated


def main():
    if DistilBertTokenizerFast is None or DistilBertModel is None or torch is None:
        raise SystemExit("transformers/torch not available in environment")

    parser = argparse.ArgumentParser(
        description="Build labels with DistilBERT for instances in data_info.p"
    )
    parser.add_argument(
        "--input",
        default="labeled_data/train_data_0_5000/data_info.p",
        help="Input data_info.p path",
    )
    parser.add_argument(
        "--output",
        default="labeled_data/train_data_0_5000/data_info_distilbert.p",
        help="Output data_info.p path with updated labels",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    args = parser.parse_args()

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    model = DistilBertModel.from_pretrained("distilbert-base-uncased")
    model.eval()
    model.to(args.device)

    with open(args.input, "rb") as f:
        train_data, valid_data = pickle.load(f)

    train_updated = _annotate_instances(train_data, model, tokenizer, args.batch_size, args.device)
    valid_updated = _annotate_instances(valid_data, model, tokenizer, args.batch_size, args.device)

    out_dir = os.path.dirname(args.output)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    with open(args.output, "wb") as f:
        pickle.dump([train_updated, valid_updated], f)

    print(f"Wrote updated data to {args.output}")


if __name__ == "__main__":
    main()
