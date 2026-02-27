from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from os import listdir
from os.path import isfile, join
import re
import rake
import pickle
from mind_map_generation import *
import rouge
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "TTED", "text-tree-distance", "code"))

TTED_AVAILABLE = True
TTED_IMPORT_ERROR = None
try:
    from tted.tree_format import TextTree
    from tted.computation import tted, avg_tted
    from tted.baseline import baseline_distance, baseline_similarity
except Exception as exc:
    TTED_AVAILABLE = False
    TTED_IMPORT_ERROR = exc
    TextTree = None
    tted = None
    avg_tted = None
    baseline_distance = None
    baseline_similarity = None

def _to_sentence(node):
    if node is None:
        return ""
    if isinstance(node, str):
        return node.strip()
    if isinstance(node, (list, tuple)):
        if len(node) == 0:
            return ""
        if all(isinstance(tok, str) for tok in node):
            return " ".join(node).strip()
    return str(node).strip()


def _normalize_tree(nodes, adj, root_idx=None):
    if TextTree is None:
        return None

    if not nodes:
        return None

    normalized_adj = [children[:] for children in adj]
    indegree = [0] * len(nodes)
    for parent, children in enumerate(normalized_adj):
        for child in children:
            if 0 <= child < len(nodes):
                indegree[child] += 1

    roots = [i for i, deg in enumerate(indegree) if deg == 0]
    if root_idx is None or root_idx < 0 or root_idx >= len(nodes):
        root_idx = roots[0] if roots else 0

    for orphan_root in roots:
        if orphan_root != root_idx and orphan_root not in normalized_adj[root_idx]:
            normalized_adj[root_idx].append(orphan_root)

    order = []
    visited = set()
    stack = [root_idx]
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        order.append(node)
        for child in reversed(normalized_adj[node]):
            if child not in visited:
                stack.append(child)

    for node in range(len(nodes)):
        if node not in visited:
            order.append(node)

    remap = {old_idx: new_idx for new_idx, old_idx in enumerate(order)}
    new_nodes = [nodes[i] for i in order]
    new_adj = []
    for old_idx in order:
        new_adj.append([remap[child] for child in normalized_adj[old_idx] if child in remap])

    return TextTree(new_nodes, new_adj)


def _story_to_text_tree(filename):
    with open(filename, "r", encoding="utf-8") as f:
        content = f.read()

    highlight_blocks = re.findall(re.compile("<highlight>.*?</highlight>", re.DOTALL), content, flags=0)
    if not highlight_blocks:
        return None

    block = highlight_blocks[0].replace("<highlight>", "").replace("</hightlight>", "").replace("</highlight>", "")
    raw_nodes = re.findall(re.compile("<T.*?</T", re.DOTALL), block, flags=0)
    if not raw_nodes:
        return None

    tags = []
    nodes = []
    for raw in raw_nodes:
        parts = re.split(">", raw, maxsplit=1)
        tag = parts[0].replace("<T.", "").replace("<T", "").replace(">", "").strip()
        sentence = parts[1].replace("</T", "").strip() if len(parts) > 1 else ""
        tags.append(tag)
        nodes.append(sentence)

    tag_to_idx = {tag: i for i, tag in enumerate(tags)}
    adj = [[] for _ in nodes]
    root_idx = None

    for idx, tag in enumerate(tags):
        if "." not in tag:
            if root_idx is None:
                root_idx = idx
            continue

        parent_tag = tag[:-2]
        while parent_tag and parent_tag not in tag_to_idx:
            parent_tag = parent_tag[:-2]
        if parent_tag in tag_to_idx:
            parent_idx = tag_to_idx[parent_tag]
            adj[parent_idx].append(idx)

    return _normalize_tree(nodes, adj, root_idx=root_idx)


def _pairs_to_text_tree(pairs):
    if not pairs:
        return None

    nodes = []
    adj = []
    node_to_idx = {}
    parent_of = {}
    edges = []
    root_idx = None

    def get_or_create(text):
        if text in node_to_idx:
            return node_to_idx[text]
        node_to_idx[text] = len(nodes)
        nodes.append(text)
        adj.append([])
        return node_to_idx[text]

    for pair in pairs:
        if not isinstance(pair, (list, tuple)) or len(pair) < 2:
            continue

        parent_text = _to_sentence(pair[0])
        child_text = _to_sentence(pair[1])
        if not child_text:
            continue

        child_idx = get_or_create(child_text)
        if not parent_text:
            if root_idx is None:
                root_idx = child_idx
            continue

        parent_idx = get_or_create(parent_text)
        if parent_idx == child_idx:
            continue
        if child_idx in parent_of:
            continue

        parent_of[child_idx] = parent_idx
        edges.append((parent_idx, child_idx))

    for parent_idx, child_idx in edges:
        if child_idx not in adj[parent_idx]:
            adj[parent_idx].append(child_idx)

    if not nodes:
        return None

    if root_idx is None:
        indegree = [0] * len(nodes)
        for children in adj:
            for child in children:
                indegree[child] += 1
        candidates = [i for i, deg in enumerate(indegree) if deg == 0]
        root_idx = candidates[0] if candidates else 0

    return _normalize_tree(nodes, adj, root_idx=root_idx)


def _cosine_distance(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 1.0
    return 1.0 - float(np.dot(a, b) / denom)


def _get_sentence_transformer_encoder(model_name):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)

    def encoder(texts):
        return model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    return encoder


def process_wordPairs(data):
    new_data = []
    for each in data:
        new_each = []
        one, two = each
        if len(one) == 0:
            new_each.append("")
        else:
            new_each.append(" ".join(one))

        if len(two) == 0:
            new_each.append("")
        else:
            new_each.append(" ".join(two))

        new_data.append(new_each)
    return new_data


def compare_method(pairs, pairs2):
    sim = 0.0
    for i in range(len(pairs2)):
        # msp = ['', '']
        # old code
        msp = pairs[1]
        if i == 0:
            continue
        found = False
        for j in range(len(pairs)):
            # modified
            if j == 0:
                continue
            first = rouge_sim2(pairs2[i][0], msp[0]) + rouge_sim2(pairs2[i][1], msp[1])
            second = rouge_sim2(pairs2[i][0], pairs[j][0]) + rouge_sim2(pairs2[i][1], pairs[j][1])
            if first < second:
                msp = pairs[j]
                max_index = j
                found = True
        cur_sim = rouge_sim2(pairs2[i][0], msp[0])
        cur_sim += rouge_sim2(pairs2[i][1], msp[1])

        sim += cur_sim / 2

        if found:
            del pairs[max_index]
    return sim

def main(
    benchmarks,
    my_results,
    sim_threshold,
    enable_tted=False,
    tted_model_name="sentence-transformers/paraphrase-distilroberta-base-v2",
    return_tted=False,
):
    cc = listdir(benchmarks)
    totalSim = 0
    totalSim_word = 00
    total_tted = 0.0
    tted_samples = 0
    evaluator_number = 0
    encoder = None
    cc.sort()

    if enable_tted:
        if not TTED_AVAILABLE:
            print(f"TTED disabled (dependency import failed): {TTED_IMPORT_ERROR}")
            enable_tted = False
        else:
            try:
                encoder = _get_sentence_transformer_encoder(tted_model_name)
                print(f"TTED enabled with model: {tted_model_name}")
            except Exception as exc:
                print(f"TTED disabled (failed to load model): {exc}")
                enable_tted = False

    for idx, target in enumerate(cc):
        if target.find('.story') >= 0:
            print(target)
            # if target == '12.story':
            #     continue
            pairs2, wordPairs2, length_threshold = parse_docs(join(benchmarks, target))
            # print(pairs2)
            # print(wordPairs2)

            target_id = target[0: target.find('.story')]
            sents, prob_matrix = my_results[target_id]

            pairs, wordPairs = my_generate_mindmap(sents, prob_matrix, len(pairs2), sim_threshold, length_threshold)
            # print(f'pairs = {pairs}')
            tmp_pairs = pairs[:]
            # print(f'tmp_pairs = {tmp_pairs}')
            sim = compare_method(tmp_pairs, pairs2)


            wordPairs = process_wordPairs(wordPairs)
            wordPairs2 = process_wordPairs(wordPairs2)
            sim_word = compare_method(wordPairs, wordPairs2)
            print(sim/len(pairs2), sim_word/len(wordPairs2))


            if enable_tted and encoder is not None:
                reference_tree = _story_to_text_tree(join(benchmarks, target))
                generated_tree = _pairs_to_text_tree(pairs)
                print("---------||||||-------")
                print(pairs)
                print(generated_tree)
                print("----------------")
                print(reference_tree)
                print("---------||||||-------")
                if reference_tree is not None and generated_tree is not None:
                    current_tted = avg_tted(
                        generated_tree,
                        reference_tree,
                        encoder=encoder,
                        embedding_dist=_cosine_distance,
                        unordered=False,
                        use_context=False,
                    )
                    total_tted += current_tted
                    tted_samples += 1

            totalSim += sim / len(pairs2)
            totalSim_word += sim_word / len(wordPairs2)
            evaluator_number += 1
            print("TTED:", current_tted)

    print("final result for", evaluator_number, " files")
    print(str(totalSim / evaluator_number))
    print("key word: ", str(totalSim_word / evaluator_number))

    avg_sentence_score = totalSim / evaluator_number
    avg_keyword_score = totalSim_word / evaluator_number

    if enable_tted:
        if tted_samples > 0:
            avg_tted_score = total_tted / tted_samples
            print("TTED average:", avg_tted_score)
        else:
            avg_tted_score = None
            print("TTED average: n/a (no valid trees)")

        if return_tted:
            return avg_sentence_score, avg_keyword_score, avg_tted_score

    return avg_sentence_score, avg_keyword_score







