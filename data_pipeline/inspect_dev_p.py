#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import pickle


def main():
    parser = argparse.ArgumentParser(description="Inspect dev_full.p or similar pickle")
    parser.add_argument("--input", required=True, help="Path to .p file")
    parser.add_argument("--index", type=int, default=0, help="Item index to print")
    args = parser.parse_args()

    with open(args.input, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, list) and len(data) == 1:
        data = data[0]

    item = data[args.index]
    print(f"Index: {args.index}")
    if isinstance(item, (list, tuple)):
        print(f"Fields: {len(item)}")
        for i, part in enumerate(item):
            print(f"\n--- Field {i} ({type(part)}) ---")
            if isinstance(part, list):
                if len(part) > 0 and isinstance(part[0], list):
                    print(f"List of lists, len={len(part)}. First element:")
                    print(part[0])
                else:
                    print(f"List len={len(part)}. Sample:")
                    print(part[:5])
            else:
                print(part)
    else:
        print(item)


if __name__ == "__main__":
    main()
