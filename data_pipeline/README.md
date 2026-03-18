# Data Pipeline: Raw Text + Labels -> data_info.p / max_info.p

This repository trains from `labeled_data/train_data_0_5000/data_info.p` and
`labeled_data/train_data_0_5000/max_info.p`. These pickles are produced by
`labeled_data/process_bert_labeled.py`. That script expects two inputs that are
not stored in this repo:

1) `final_token.p` with tokenized sentences (train/dev)
2) labeled instances (`train_labeled_0_5000.p` and `valid_labeled.p`)

This document explains how to reproduce those inputs from raw text and labels
and then generate `data_info.p` / `max_info.p`.

## Target Formats

### A) final_token.p

This file is expected at:
`generate_training_data/final_dataset/final_token.p`

Structure (pickled):

```
[train_items, dev_items]

train_items[i] = [
    id,                    # str
    contents,              # list[str]      # content sentences
    abstracts,             # list[str]      # abstract sentences
    contents_for_token,    # list[str]      # same as contents; kept for tokenization
    abstracts_for_token,   # list[str]      # same as abstracts; kept for tokenization
]
```

The code that reads this is in `labeled_data/process_bert_labeled.py`:
`get_info()` expects each item to have the 5 fields above.

### B) labeled data

Expected files:

```
labeled_data/train_data_0_5000/train_labeled_0_5000.p
labeled_data/train_data_0_5000/valid_labeled.p
```

Structure (pickled):

```
[instances]

instances[i] = [
    id,        # str
    _,         # unused in process_bert_labeled.py
    _,         # unused
    label,     # label array or list (this becomes each[5] later)
]
```

Note: in `process_bert_labeled.py`, `label = each[5]` is used, so your
instance list must align to that indexing. If you keep 4 fields, adjust the
script accordingly or pad with placeholders.

## Step 1: Build final_token.p from raw text

If your data is in `.story` format (CNN/DailyMail style), reuse the sentence
splitter in `testing_final_20201012/handle_testing_data.py`.

Use the provided script:

```
python data_pipeline/build_final_token.py \
  --train_jsonl path/to/train.jsonl \
  --dev_jsonl path/to/dev.jsonl
```

Or from `.story` directories (same format as test data):

```
python data_pipeline/build_final_token.py \
  --train_story_dir path/to/train_story/ \
  --dev_story_dir path/to/dev_story/
```

Or split a single `.story` directory:

```
python data_pipeline/build_final_token.py \
  --story_dir path/to/all_story/ --dev_ratio 0.1
```

Or split a single JSONL:

```
python data_pipeline/build_final_token.py \
  --input_jsonl path/to/all.jsonl --dev_ratio 0.1
```

JSONL format (one object per line):

```
{"id": "...", "content": "...", "abstract": "..."}
```

The key requirement is that `id` matches the ids in your labeled data.

## Step 2: Build labeled pickles

`process_bert_labeled.py` joins labels with tokens by `id`. Prepare
`train_labeled_0_5000.p` and `valid_labeled.p` with the same ids.

Use the provided script:

```
python data_pipeline/build_labeled_pickles.py \
  --train_jsonl path/to/train_labels.jsonl \
  --dev_jsonl path/to/dev_labels.jsonl \
  --label_key label
```

JSONL format (one object per line):

```
{"id": "...", "label": ...}
```

This script writes a minimal 4-field format:

```
[ [id, None, None, label] , ... ]
```
This 4-field format is already compatible with `process_bert_labeled.py`.

## Baseline labels with vanilla DistilBERT (no fine-tuning)

If you want a quick baseline without fine-tuning, you can generate N×N
relation matrices using DistilBERT sentence embeddings and cosine similarity.

```
python data_pipeline/annotate_distilbert.py \
  --train_story_dir path/to/train_story/ \
  --dev_story_dir path/to/dev_story/ \
  --device cpu
```

This creates:

```
labeled_data/train_data_0_5000/train_labeled_0_5000.p
labeled_data/train_data_0_5000/valid_labeled.p
```

Then run:

```
python labeled_data/process_bert_labeled.py
```

## Step 3: Run process_bert_labeled.py

```
python labeled_data/process_bert_labeled.py
```

It will create:

```
labeled_data/train_data_0_5000/data_info.p
labeled_data/train_data_0_5000/max_info.p
```

## Checklist

1) `final_token.p` exists and matches ids
2) labeled pickles exist and match ids
3) `process_bert_labeled.py` points to correct paths
4) run `process_bert_labeled.py`

## Coreference maps for train/valid

If you want coreference maps aligned with training data, run
`create_coreference_maps.py` on `data_info.p`:

```
python create_coreference_maps.py \
  --input labeled_data/train_data_0_5000/data_info.p \
  --output_dir directed_maps_simple
```

This will write:

```
directed_maps_simple/data_info_train_maps.p
directed_maps_simple/data_info_valid_maps.p
directed_maps_simple/data_info_train_clusters.p
directed_maps_simple/data_info_valid_clusters.p
```

If DGL is available, it also writes:

```
directed_maps_simple/DGLgraph/data_info_train.p
directed_maps_simple/DGLgraph/data_info_valid.p
```

## One-command dev inference from .story

```
python data_pipeline/run_dev_from_story.py \
  --story_dir testing_final_20201012/testing_data_hmt_20201012_final/dev_full_original_new/
```

Optional flags:
```
--label_mode distilbert|zeros|file
--labels path/to/dev_bert.p
--gold_dir testing_final_20201012/testing_data_hmt_20201012_final/a_labeling_dev
--device cpu|cuda
--batch_size 32
--skip_coref
--skip_infer
--coref_out directed_maps_simple
--run_train
--train_data_path labeled_data/train_data_0_5000/
--train_data_info data_info.p
--train_max_info max_info.p
--train_coref_train_maps directed_maps/DGLgraph/train.p
--train_coref_valid_maps directed_maps/DGLgraph/valid.p
--train_coref_dev_maps directed_maps/DGLgraph/dev.p
--train_model_name my_model
--train_dir_output results_my/
```

### Dev inference with custom model checkpoint

```
python data_pipeline/run_dev_from_story.py \
  --story_dir testing_final_20201012/testing_data_hmt_20201012_final/dev_full_original_test \
  --dev_full testing_final_20201012/processed_new/dev_full.p \
  --seq2graph_dir testing_final_20201012/processed_for_seq2graph_dev_new/ \
  --label_mode gold \
  --gold_dir testing_final_20201012/testing_data_hmt_20201012_final/a_labeling_dev \
  --coref_out directed_maps_simple/new \
  --infer_model_path /abs/path/to/your_model.pkl \
  --clear_generated
```

This runs `testing_final_20201012/dev_seq2graph.py` with the dev-specific inputs.
