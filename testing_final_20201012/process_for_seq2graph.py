# -*- coding: utf8 -*-
# from collections import defaultdict
import pickle
import os
import numpy as np
import collections
import nltk
import re
from tqdm import tqdm

try:
    import torch
    from transformers import DistilBertModel, DistilBertTokenizerFast
except Exception:
    torch = None
    DistilBertModel = None
    DistilBertTokenizerFast = None

import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
try:
    from mind_map_generation import parse_docs
except Exception:
    parse_docs = None
try:
    from data_pipeline.rst_labels import build_labels_with_rst
except Exception:
    build_labels_with_rst = None

def get_label(id_, bert_labels):
    label = []
    for each in bert_labels:
        cur_id_ = each[0]
        if id_ == cur_id_:
            label = each[-1]
            # print(type(label))
            break
    return label


def _embed_sentences(model, tokenizer, sentences, batch_size, device):
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


def _cosine_matrix(x):
    x_norm = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
    sim = np.matmul(x_norm, x_norm.T)
    return (sim + 1.0) / 2.0


def build_labels_with_distilbert(data, batch_size, device):
    if DistilBertTokenizerFast is None or DistilBertModel is None or torch is None:
        raise RuntimeError("transformers/torch not available for distilbert labels")

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    model = DistilBertModel.from_pretrained("distilbert-base-uncased")
    model.eval()
    model.to(device)

    labels = []
    for each in tqdm(data, total=len(data)):
        id_ = each[0]
        content_sents = each[3]
        if len(content_sents) == 0:
            labels.append([id_, np.zeros((0, 0), dtype=np.float32)])
            continue
        emb = _embed_sentences(model, tokenizer, content_sents, batch_size, device)
        mat = _cosine_matrix(emb)
        np.fill_diagonal(mat, 1.0)
        labels.append([id_, mat])
    return labels


def _normalize_sent(s):
    return re.sub(r"\s+", " ", s.strip().lower())


def build_labels_from_gold(data, gold_dir):
    if parse_docs is None:
        raise RuntimeError("mind_map_generation.parse_docs not available")

    labels = []
    for each in tqdm(data, total=len(data)):
        id_ = each[0]
        content_sents = each[3]
        norm_to_idxs = collections.defaultdict(list)
        for idx, s in enumerate(content_sents):
            norm_to_idxs[_normalize_sent(s)].append(idx)

        story_path = os.path.join(gold_dir, f"{id_}.story")
        if not os.path.exists(story_path):
            labels.append([id_, np.zeros((len(content_sents), len(content_sents)), dtype=np.float32)])
            continue

        pairs, _word_pairs, _len_contents = parse_docs(story_path)
        mat = np.zeros((len(content_sents), len(content_sents)), dtype=np.float32)
        np.fill_diagonal(mat, 1.0)
        for parent, child in pairs:
            if not parent or not child:
                continue
            p_idxs = norm_to_idxs.get(_normalize_sent(parent), [])
            c_idxs = norm_to_idxs.get(_normalize_sent(child), [])
            for pi in p_idxs:
                for ci in c_idxs:
                    mat[pi, ci] = 1.0
        labels.append([id_, mat])
    return labels

def handle_data(word_to_id, data, hmt_max_sentence_length, hmt_max_content_length, bert_labels):
    new_data = []
    for each in tqdm(data, total=len(data)):
        id_ = each[0]
        label = get_label(id_, bert_labels)
        contents_token = each[5]
        abstracts_token = each[6]

        cur_max = 0
        for s in contents_token:
            if len(s) > cur_max:
                cur_max = len(s)

        if cur_max <= hmt_max_sentence_length and len(contents_token) <= hmt_max_content_length:
            contents_id = []
            ########### handle ids ############
            for s in contents_token:
                cur_s_id = []
                for each_w in s:
                    if each_w in word_to_id:
                        cur_s_id.append(word_to_id[each_w])
                    else:
                        cur_s_id.append(word_to_id['<unk>'])
                contents_id.append(cur_s_id)
            contents_id_length = len(contents_id)

            abstracts_id = []
            for s in abstracts_token:
                cur_s_id = []
                for each_w in s:
                    if each_w in word_to_id:
                        cur_s_id.append(word_to_id[each_w])
                    else:
                        cur_s_id.append(word_to_id['<unk>'])
                abstracts_id.append(cur_s_id)

            contents_padding = []
            contents_mask = []
            ######## padding part ##########
            for s in contents_id:
                sentence_padding = [0] * (hmt_max_sentence_length - len(s))
                sentence_mask = [1] * len(s)

                s += sentence_padding
                sentence_mask += sentence_padding

                contents_padding.append(s)
                contents_mask.append(sentence_mask)

            seq_mask = [1] * contents_id_length + \
                       [0] * (hmt_max_content_length - contents_id_length)
            for index in range(hmt_max_content_length - contents_id_length):
                contents_padding.append([0] * hmt_max_sentence_length)
                contents_mask.append([0] * hmt_max_sentence_length)

            ###### saving data ########
            cur_instance = []
            cur_instance.append(id_)
            cur_instance.append(contents_padding)
            cur_instance.append(contents_mask)

            cur_instance.append(abstracts_id)
            cur_instance.append(label)
            cur_instance.append(contents_id_length)
            cur_instance.append(seq_mask)
            cur_instance.append(each[3])
            cur_instance.append(each[4])
            new_data.append(cur_instance)
    return new_data

def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def get_max_length(data):
    max_sentence_length = 0
    max_content_length = 0
    for each in data:
        id_, _, _, _, _, contents_token, _ = each
        for s in contents_token:
            if len(s) > max_sentence_length:
                max_sentence_length = len(s)

        cur_content_length = len(contents_token)
        if cur_content_length > max_content_length:
            max_content_length = cur_content_length

    return max_sentence_length, max_content_length

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Process dev_full.p into seq2graph input")
    parser.add_argument(
        "--input",
        default="processed/dev_full.p",
        help="Input pickle path (dev_full.p format)",
    )
    parser.add_argument(
        "--output_dir",
        default="processed_for_seq2graph_dev/",
        help="Output directory for dev_for_mymodel_info.p and max_info.p",
    )
    parser.add_argument(
        "--labels",
        default=None,
        help="BERT labels pickle path (optional)",
    )
    parser.add_argument(
        "--label_mode",
        choices=["file", "distilbert", "zeros", "gold", "rst"],
        default="file",
        help="How to obtain labels when --labels is missing",
    )
    parser.add_argument(
        "--rst_parser",
        default="auto",
        help="RST parser backend (auto|rstparser|rst_parser)",
    )
    parser.add_argument(
        "--gold_dir",
        default=None,
        help="Directory with gold .story files (a_labeling_dev)",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for distilbert")
    parser.add_argument("--device", default="cpu", help="cpu or cuda for distilbert")
    args = parser.parse_args()

    data_path = args.input
    writing_path = args.output_dir
    os_exists = os.path.exists(writing_path)
    if not os_exists:
        os.mkdir(writing_path)

    data = pickle.load(open(data_path, "rb"))[0]

    if args.labels:
        labels_by_bert = args.labels
        bert_labels = pickle.load(open(labels_by_bert, "rb"))[0]
    else:
        if args.label_mode == "file":
            raise FileNotFoundError("labels file is required when label_mode=file")
        if args.label_mode == "distilbert":
            bert_labels = build_labels_with_distilbert(data, args.batch_size, args.device)
        elif args.label_mode == "gold":
            if not args.gold_dir:
                raise FileNotFoundError("gold_dir is required when label_mode=gold")
            bert_labels = build_labels_from_gold(data, args.gold_dir)
        elif args.label_mode == "rst":
            if build_labels_with_rst is None:
                raise RuntimeError("RST labels require data_pipeline/rst_labels.py and a supported parser")
            bert_labels = build_labels_with_rst(data, parser_name=args.rst_parser)
        else:
            bert_labels = [[each[0], np.zeros((len(each[3]), len(each[3])), dtype=np.float32)] for each in data]

    max_sentence_1, max_list_1 = get_max_length(data)

    max_sentence_length = max_sentence_1
    max_content_length = max_list_1

    print("max sentence length: ", max_sentence_length)
    print("max content length: ", max_content_length)

    ####### build vocabulary #######

    word_to_index, index_to_word, _ = pickle.load(open("labeling/model/my_vocabulary_add_padd.pickle", "rb"))

    ###### handle training data and testing data ########
    data_id = handle_data(word_to_index, data, max_sentence_length, max_content_length, bert_labels)

    pickle.dump([data_id], open(os.path.join(writing_path, "dev_for_mymodel_info.p"), "wb"))
    pickle.dump([max_sentence_length, max_content_length], open(os.path.join(writing_path, "max_info.p"), "wb"))
    ### word_id_info ../model/my_vocabulary_add_padd.pickle
    ### embedding ../word_embedding 还没有 <pad>

if __name__ == '__main__':
    main()
