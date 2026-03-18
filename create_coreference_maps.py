import os
import pickle
import re
from collections import defaultdict

import numpy as np
import spacy


ARTICLE_DETERMINERS = {
    "a", "an", "the", "this", "that", "these", "those",
    "my", "your", "his", "her", "its", "our", "their",
}


def _normalize_mention(text):
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9\s'-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    parts = text.split()
    while parts and parts[0] in ARTICLE_DETERMINERS:
        parts = parts[1:]
    if not parts:
        return ""
    if parts[-1] == "'s":
        parts = parts[:-1]
    return " ".join(parts).strip()


def _extract_mentions(sent, nlp, stop_words):
    if nlp is None:
        # Fallback: simple noun-like chunks based on tokens
        tokens = re.findall(r"[a-zA-Z0-9']+", sent.lower())
        mentions = []
        for t in tokens:
            if t in stop_words:
                continue
            if len(t) < 3:
                continue
            mentions.append(t)
        return mentions

    doc = nlp(sent)
    mentions = []

    # Named entities
    for ent in doc.ents:
        key = _normalize_mention(ent.text)
        if key and len(key) >= 3:
            mentions.append(key)

    # Noun chunks as fallback for core mentions
    for chunk in doc.noun_chunks:
        if chunk.root.pos_ == "PRON":
            continue
        key = _normalize_mention(chunk.text)
        if not key or len(key) < 3:
            continue
        # Skip stopword-only chunks
        if all(t.is_stop for t in chunk):
            continue
        mentions.append(key)

    return mentions


def build_clusters(sentences, nlp=None):
    clusters = defaultdict(list)
    stop_words = set()
    if nlp is not None:
        stop_words = nlp.Defaults.stop_words

    for sent_idx, sent in enumerate(sentences):
        mentions = _extract_mentions(sent, nlp, stop_words)
        seen_in_sentence = set()
        for m in mentions:
            if m in seen_in_sentence:
                continue
            clusters[m].append(sent_idx)
            seen_in_sentence.add(m)

    # Keep only clusters with at least 2 mentions across sentences
    filtered = []
    for _, sents in clusters.items():
        uniq = sorted(set(sents))
        if len(uniq) >= 2:
            filtered.append(uniq)
    return filtered


def build_adjacency(num_nodes, clusters):
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float64)
    np.fill_diagonal(adj, 1.0)

    for cluster in clusters:
        if not cluster:
            continue
        root = min(cluster)
        for idx in cluster:
            adj[root, idx] = 1.0
    return adj


def build_dgl_graphs(adjs):
    try:
        import dgl
    except Exception:
        return None

    graphs = []
    for adj in adjs:
        src, dst = np.where(adj > 0)
        g = dgl.graph((src, dst), num_nodes=adj.shape[0])
        graphs.append(g)
    return graphs


def load_sentences_from_pickle(path):
    with open(path, "rb") as f:
        data = pickle.load(f)

    # data_info.p format: [train_list, valid_list]
    if isinstance(data, list) and len(data) == 2 and all(isinstance(x, list) for x in data):
        if data and data[0] and isinstance(data[0][0], (list, tuple)) and len(data[0][0]) >= 9:
            return {
                "train": [item[7] for item in data[0]],
                "valid": [item[7] for item in data[1]],
            }

    if isinstance(data, list) and len(data) == 1:
        data = data[0]

    sentences = []
    for item in data:
        if isinstance(item, dict) and "content_sentences" in item:
            sentences.append(item["content_sentences"])
        elif isinstance(item, (list, tuple)) and len(item) >= 9:
            sentences.append(item[7])
        elif isinstance(item, (list, tuple)) and len(item) >= 4:
            sentences.append(item[3])
        else:
            raise ValueError("Unrecognized item format in data file")
    return sentences


def generate_maps(input_path, output_dir):
    if spacy is None:
        nlp = None
    else:
        try:
            nlp = spacy.load("en_core_web_sm")
        except Exception:
            nlp = None

    sentence_sets = load_sentences_from_pickle(input_path)

    if isinstance(sentence_sets, dict):
        outputs = {}
        for split_name, split_sents in sentence_sets.items():
            all_maps = []
            all_clusters = []
            for sents in split_sents:
                clusters = build_clusters(sents, nlp)
                adj = build_adjacency(len(sents), clusters)
                all_maps.append(adj)
                all_clusters.append(clusters)
            outputs[split_name] = (all_maps, all_clusters)
    else:
        all_maps = []
        all_clusters = []
        for sents in sentence_sets:
            clusters = build_clusters(sents, nlp)
            adj = build_adjacency(len(sents), clusters)
            all_maps.append(adj)
            all_clusters.append(clusters)
        outputs = {"all": (all_maps, all_clusters)}

    os.makedirs(output_dir, exist_ok=True)
    name = os.path.basename(input_path).replace("_full", "").replace(".p", "")
    results = {}
    for split_name, (maps, clusters) in outputs.items():
        suffix = "" if split_name == "all" else f"_{split_name}"
        maps_path = os.path.join(output_dir, f"{name}{suffix}_maps.p")
        clusters_path = os.path.join(output_dir, f"{name}{suffix}_clusters.p")
        with open(maps_path, "wb") as f:
            pickle.dump(maps, f)
        with open(clusters_path, "wb") as f:
            pickle.dump(clusters, f)

        graphs = build_dgl_graphs(maps)
        if graphs is not None:
            dgl_dir = os.path.join(output_dir, "DGLgraph")
            os.makedirs(dgl_dir, exist_ok=True)
            dgl_path = os.path.join(dgl_dir, f"{name}{suffix}.p")
            with open(dgl_path, "wb") as f:
                pickle.dump(graphs, f)

        results[split_name] = (maps_path, clusters_path)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Create coreference maps without AllenNLP by simple mention clustering."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to processed *.p file with content sentences (e.g., dev_full.p)",
    )
    parser.add_argument(
        "--output_dir",
        default="directed_maps_simple",
        help="Output directory for maps and clusters",
    )
    args = parser.parse_args()

    generate_maps(args.input, args.output_dir)
