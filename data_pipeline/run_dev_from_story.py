#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import os
import subprocess
import sys
import shutil


def run(cmd, cwd=None):
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd, cwd=cwd)


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end dev inference from .story files"
    )
    parser.add_argument(
        "--story_dir",
        required=True,
        help="Directory with .story files",
    )
    parser.add_argument(
        "--dev_full",
        default="testing_final_20201012/processed_new/dev_full.p",
        help="Output dev_full.p path",
    )
    parser.add_argument(
        "--dev_full_mode",
        choices=["build", "read"],
        default="build",
        help="build: create dev_full.p from .story; read: use existing dev_full.p",
    )
    parser.add_argument(
        "--seq2graph_dir",
        default="testing_final_20201012/processed_for_seq2graph_dev_new/",
        help="Output dir for dev_for_mymodel_info.p and max_info.p",
    )
    parser.add_argument(
        "--seq2graph_mode",
        choices=["build", "read"],
        default="build",
        help="build: create dev_for_mymodel_info.p; read: use existing files",
    )
    parser.add_argument(
        "--label_mode",
        choices=["distilbert", "zeros", "file", "gold"],
        default="distilbert",
        help="How to build labels for dev",
    )
    parser.add_argument(
        "--labels",
        default=None,
        help="Labels pickle path when label_mode=file",
    )
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--gold_dir",
        default=None,
        help="Directory with gold .story files (a_labeling_dev)",
    )
    parser.add_argument(
        "--coref_out",
        default="directed_maps_simple",
        help="Output dir for coreference maps",
    )
    parser.add_argument(
        "--coref_mode",
        choices=["build", "read", "skip"],
        default="build",
        help="build: create coref maps; read: use existing maps; skip: no coref",
    )
    parser.add_argument(
        "--skip_coref",
        action="store_true",
        help="Skip coreference map generation",
    )
    parser.add_argument(
        "--skip_infer",
        action="store_true",
        help="Skip model inference",
    )
    parser.add_argument(
        "--prep_only",
        action="store_true",
        help="Only build dev_full.p and coreference maps (skip seq2graph/infer/train)",
    )
    parser.add_argument(
        "--clear_generated",
        action="store_true",
        help="Clear a_generated_dev before inference",
    )
    parser.add_argument(
        "--run_train",
        action="store_true",
        help="Run train.py instead of test_seq2graph.py",
    )
    parser.add_argument("--train_data_path", default=None, help="Path for train data_dir")
    parser.add_argument("--train_data_info", default=None, help="data_info filename")
    parser.add_argument("--train_max_info", default=None, help="max_info filename")
    parser.add_argument("--train_coref_train_maps", default=None, help="Coref train maps path")
    parser.add_argument("--train_coref_valid_maps", default=None, help="Coref valid maps path")
    parser.add_argument("--train_coref_dev_maps", default=None, help="Coref dev maps path")
    parser.add_argument("--train_model_name", default=None, help="Model name for train.py")
    parser.add_argument("--train_dir_output", default=None, help="Output dir for train.py")
    parser.add_argument(
        "--coref_maps",
        default=None,
        help="Path to coref DGL map for inference (overrides auto)",
    )
    parser.add_argument(
        "--infer_model_path",
        default=None,
        help="Model checkpoint path for inference (test_seq2graph.py)",
    )
    parser.add_argument(
        "--generated_dir",
        default=None,
        help="Output dir for generated .story files during dev inference",
    )
    parser.add_argument(
        "--benchmarks",
        default=None,
        help="Gold .story directory for evaluation (a_labeling_dev)",
    )
    args = parser.parse_args()

    # Step 1: dev_full.p
    if args.dev_full_mode == "build":
        run(
            [
                sys.executable,
                "testing_final_20201012/handle_testing_data.py",
                "--input_dir",
                args.story_dir,
                "--output",
                args.dev_full,
            ]
        )

    if not args.prep_only and args.seq2graph_mode == "build":
        # Step 2: processed_for_seq2graph
        cmd = [
            sys.executable,
            "testing_final_20201012/process_for_seq2graph.py",
            "--input",
            args.dev_full,
            "--output_dir",
            args.seq2graph_dir,
            "--label_mode",
            args.label_mode,
            "--device",
            args.device,
            "--batch_size",
            str(args.batch_size),
        ]
        if args.label_mode == "file":
            if not args.labels:
                raise SystemExit("--labels is required when label_mode=file")
            cmd += ["--labels", args.labels]
        if args.label_mode == "gold":
            if not args.gold_dir:
                raise SystemExit("--gold_dir is required when label_mode=gold")
            cmd += ["--gold_dir", args.gold_dir]
        run(cmd)

    # Step 3: coreference maps
    coref_maps = args.coref_maps
    if args.coref_mode == "build" and not args.skip_coref:
        run(
            [
                sys.executable,
                "create_coreference_maps.py",
                "--input",
                args.dev_full,
                "--output_dir",
                args.coref_out,
            ]
        )
        base = os.path.basename(args.dev_full).replace("_full", "").replace(".p", "")
        coref_maps = os.path.join(args.coref_out, "DGLgraph", f"{base}.p")
    elif args.coref_mode == "read":
        if not coref_maps:
            base = os.path.basename(args.dev_full).replace("_full", "").replace(".p", "")
            coref_maps = os.path.join(args.coref_out, "DGLgraph", f"{base}.p")
    elif args.coref_mode == "skip":
        coref_maps = None

    # Step 4: inference or training
    if not args.skip_infer and not args.prep_only:
        if args.run_train:
            cmd = [sys.executable, "train.py"]
            if args.train_data_path:
                cmd += ["--data_path", args.train_data_path]
            if args.train_data_info:
                cmd += ["--data_info", args.train_data_info]
            if args.train_max_info:
                cmd += ["--max_info", args.train_max_info]
            if args.train_coref_train_maps:
                cmd += ["--coref_train_maps", args.train_coref_train_maps]
            if args.train_coref_valid_maps:
                cmd += ["--coref_valid_maps", args.train_coref_valid_maps]
            if args.train_coref_dev_maps:
                cmd += ["--coref_dev_maps", args.train_coref_dev_maps]
            if args.train_model_name:
                cmd += ["--model_name", args.train_model_name]
            if args.train_dir_output:
                cmd += ["--dir_output", args.train_dir_output]
            run(cmd)
        else:
            if not coref_maps:
                raise SystemExit("coref maps path is required for inference")
            if args.clear_generated:
                if args.generated_dir:
                    gen_dir = os.path.abspath(args.generated_dir)
                else:
                    gen_dir = os.path.join(
                        "testing_final_20201012",
                        "testing_data_hmt_20201012_final",
                        "a_generated_dev",
                    )
                if os.path.isdir(gen_dir):
                    shutil.rmtree(gen_dir)
            if not os.path.isabs(coref_maps):
                coref_maps = os.path.abspath(coref_maps)
            run(
                [
                    sys.executable,
                    "dev_seq2graph.py",
                    "--coref_maps",
                    coref_maps,
                    "--input",
                    os.path.abspath(os.path.join(args.seq2graph_dir, "dev_for_mymodel_info.p")),
                    "--full",
                    os.path.abspath(args.dev_full),
                    "--max_info",
                    os.path.abspath(os.path.join(args.seq2graph_dir, "max_info.p")),
                    *(
                        ["--generated_dir", os.path.abspath(args.generated_dir)]
                        if args.generated_dir
                        else []
                    ),
                    *(
                        ["--benchmarks", os.path.abspath(args.benchmarks)]
                        if args.benchmarks
                        else []
                    ),
                    *(["--model_path", args.infer_model_path] if args.infer_model_path else []),
                ],
                cwd="testing_final_20201012",
            )


if __name__ == "__main__":
    main()
