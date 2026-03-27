#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import os
import pickle
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from data_pipeline.rst_labels import build_labels_with_rst


def _annotate_instances(instances, labels_by_id):
    label_map = {doc_id: mat for doc_id, mat in labels_by_id}
    updated = []
    for inst in instances:
        if len(inst) < 9:
            raise ValueError("Unexpected instance format in data_info.p")
        doc_id = inst[0]
        mat = label_map.get(doc_id)
        if mat is None:
            updated.append(inst)
            continue
        new_inst = list(inst)
        new_inst[4] = mat
        updated.append(new_inst)
    return updated


def main():
    parser = argparse.ArgumentParser(
        description="Build labels with RST parser for instances in data_info.p"
    )
    parser.add_argument(
        "--input",
        default="labeled_data/train_data_0_5000/data_info.p",
        help="Input data_info.p path",
    )
    parser.add_argument(
        "--output",
        default="labeled_data/train_data_0_5000/data_info_rst.p",
        help="Output data_info.p path with RST labels",
    )
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

    with open(args.input, "rb") as f:
        loaded = pickle.load(f)

    if isinstance(loaded, (list, tuple)) and len(loaded) == 2:
        train_data, valid_data = loaded
        single_split = False
    elif isinstance(loaded, (list, tuple)) and len(loaded) == 1:
        train_data = loaded[0]
        valid_data = None
        single_split = True
    else:
        train_data = loaded
        valid_data = None
        single_split = True

    # Build labels using sentence list at index 7
    parser_kwargs = {
        "hf_model_name": args.hf_model_name,
        "hf_model_version": args.hf_model_version,
        "cuda_device": args.cuda_device,
    }
    train_labels = build_labels_with_rst(
        train_data, parser_name=args.rst_parser, parser_kwargs=parser_kwargs
    )
    valid_labels = None
    if valid_data is not None:
        valid_labels = build_labels_with_rst(
            valid_data, parser_name=args.rst_parser, parser_kwargs=parser_kwargs
        )

    train_updated = _annotate_instances(train_data, train_labels)
    valid_updated = None
    if valid_data is not None and valid_labels is not None:
        valid_updated = _annotate_instances(valid_data, valid_labels)

    out_dir = os.path.dirname(args.output)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    with open(args.output, "wb") as f:
        if single_split:
            pickle.dump([train_updated], f)
        else:
            pickle.dump([train_updated, valid_updated], f)

    print(f"Wrote updated data to {args.output}")


if __name__ == "__main__":
    main()
