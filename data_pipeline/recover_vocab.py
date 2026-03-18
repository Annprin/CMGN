#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import os
import pickle


def main():
    parser = argparse.ArgumentParser(description="Recover my_vocabulary_add_padd.pickle from id_info.p")
    parser.add_argument(
        "--id_info",
        default="labeled_data/embedding/id_info.p",
        help="Path to id_info.p (word_to_index, index_to_word)",
    )
    parser.add_argument(
        "--output",
        default="labeling/model/my_vocabulary_add_padd.pickle",
        help="Output path for recovered vocab pickle",
    )
    args = parser.parse_args()

    with open(args.id_info, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, list) or len(obj) != 2:
        raise ValueError("Unexpected id_info.p format; expected [word_to_index, index_to_word]")

    word_to_index, index_to_word = obj
    if not isinstance(word_to_index, dict) or not isinstance(index_to_word, list):
        raise ValueError("Unexpected id_info.p format; expected dict + list")

    # The original my_vocabulary_add_padd.pickle usually stores (word_to_index, index_to_word, extra)
    out_dir = os.path.dirname(args.output)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    with open(args.output, "wb") as f:
        pickle.dump((word_to_index, index_to_word, None), f)

    print(f"Recovered vocab with {len(word_to_index)} entries to {args.output}")


if __name__ == "__main__":
    main()
