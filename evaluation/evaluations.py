from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from os import listdir, makedirs
from os.path import isfile, join
import os
import sys
import re
import rake
import pickle
import hashlib
from mind_map_generation import *
import rouge
import time
import datetime
import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
_TTED_ROOT_CANDIDATES = [
    os.path.join(_REPO_ROOT, "TTED", "text-tree-distance", "code"),
    os.path.join(os.path.dirname(__file__), "TTED", "text-tree-distance", "code"),
]
for _tt_path in _TTED_ROOT_CANDIDATES:
    if os.path.isdir(_tt_path) and _tt_path not in sys.path:
        sys.path.insert(0, _tt_path)
        break

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
    for tag in tags:
        if "." not in tag:
            root_idx = tag_to_idx[tag]
            continue
        father = tag.rsplit(".", 1)[0]
        while father not in tag_to_idx and "." in father:
            father = father.rsplit(".", 1)[0]
        if father in tag_to_idx:
            adj[tag_to_idx[father]].append(tag_to_idx[tag])
        else:
            if root_idx is None:
                root_idx = 0
            adj[root_idx].append(tag_to_idx[tag])

    return _normalize_tree(nodes, adj, root_idx)

def _pairs_to_text_tree(pairs):
    nodes = []
    node_to_idx = {}
    adj = []

    def _idx(node):
        sent = _to_sentence(node)
        if sent not in node_to_idx:
            node_to_idx[sent] = len(nodes)
            nodes.append(sent)
            adj.append([])
        return node_to_idx[sent]

    for parent, child in pairs:
        if len(parent) == 0:
            _idx(child)
            continue
        p_idx = _idx(parent)
        c_idx = _idx(child)
        if c_idx not in adj[p_idx]:
            adj[p_idx].append(c_idx)

    return _normalize_tree(nodes, adj, 0)

_ST_ENCODER = None

def _get_sentence_transformer_encoder(model_name):
    global _ST_ENCODER
    if _ST_ENCODER is None:
        from sentence_transformers import SentenceTransformer
        _ST_ENCODER = SentenceTransformer(model_name)
    return _ST_ENCODER

def _hash_sentence_embedding(text, dim=256):
    vec = np.zeros(dim, dtype=np.float32)
    tokens = re.findall(r"\w+", str(text).lower())
    for tok in tokens:
        h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
        idx = h % dim
        sign = 1.0 if ((h >> 1) & 1) else -1.0
        vec[idx] += sign
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec

def _cosine_dist(a, b):
    if a is None or b is None:
        return 1.0
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 1.0
    sim = float(np.dot(a, b) / denom)
    return 1.0 - sim

def _tted_encoder(text_or_list, model=None):
    if isinstance(text_or_list, (list, tuple)):
        return [_tted_encoder(x, model=model) for x in text_or_list]
    text = _to_sentence(text_or_list)
    if model is None:
        return _hash_sentence_embedding(text)
    emb = model.encode([text], show_progress_bar=False, normalize_embeddings=True)
    return emb[0]

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
    if not pairs or not pairs2:
        return 0.0

    sim = 0.0
    used = set()
    base_index = 1 if len(pairs) > 1 else 0
    base_pair = pairs[base_index]

    for i in range(len(pairs2)):
        if i == 0:
            continue

        best_pair = base_pair
        best_idx = base_index
        best_score = (
            rouge_sim2(pairs2[i][0], base_pair[0]) +
            rouge_sim2(pairs2[i][1], base_pair[1])
        )

        for j in range(len(pairs)):
            if j == 0 or j in used:
                continue
            candidate = pairs[j]
            score = rouge_sim2(pairs2[i][0], candidate[0]) + rouge_sim2(pairs2[i][1], candidate[1])
            if score > best_score:
                best_score = score
                best_pair = candidate
                best_idx = j

        cur_sim = rouge_sim2(pairs2[i][0], best_pair[0])
        cur_sim += rouge_sim2(pairs2[i][1], best_pair[1])
        sim += cur_sim / 2

        used.add(best_idx)

    return sim

def main(
    benchmarks,
    my_results,
    sim_threshold,
    enable_tted=False,
    tted_model_name="sentence-transformers/paraphrase-distilroberta-base-v2",
    return_tted=False,
    output_dir=None,
):
    if output_dir:
        makedirs(output_dir, exist_ok=True)
    cc = listdir(benchmarks)
    totalSim = 0
    totalSim_word = 0
    total_tted = 0.0
    tted_samples = 0
    evaluator_number = 0
    cc.sort()
    total_second_phase_time = 0
    encoder = None
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
            start = time.time()
            pairs2, wordPairs2, length_threshold = parse_docs(join(benchmarks, target))

            target_id = target[0: target.find('.story')]
            if target_id not in my_results.keys():
                continue
            else:
                sents, prob_matrix = my_results[target_id]

            pairs, wordPairs = my_generate_mindmap(sents, prob_matrix, len(pairs2), sim_threshold, length_threshold)
            end = time.time()
            second_time = end - start
            total_second_phase_time += second_time

            tmp_pairs = pairs[:]
            sim = compare_method(tmp_pairs, pairs2)

            wordPairs = process_wordPairs(wordPairs)
            wordPairs2 = process_wordPairs(wordPairs2)
            sim_word = compare_method(wordPairs, wordPairs2)

            # print(sim/len(pairs2), sim_word/len(wordPairs2))
            print(sim / len(pairs2), sim_word / len(wordPairs2))

            if enable_tted and encoder is not None:
                try:
                    reference_tree = _story_to_text_tree(join(benchmarks, target))
                    generated_tree = _pairs_to_text_tree(pairs)
                    if reference_tree is not None and generated_tree is not None:
                        current_tted = avg_tted(
                            generated_tree,
                            reference_tree,
                            lambda x: _tted_encoder(x, model=encoder),
                            _cosine_dist,
                            unordered=True,
                            use_context=False,
                        )
                    else:
                        current_tted = None
                except Exception:
                    current_tted = None
            else:
                current_tted = None

            totalSim += sim / len(pairs2)
            totalSim_word += sim_word / len(wordPairs2)
            evaluator_number += 1
            if enable_tted:
                print("TTED:", current_tted)
                if current_tted is not None:
                    total_tted += current_tted
                    tted_samples += 1

    print("total second phase time ", total_second_phase_time)
    # test 不加parse_docs 时间为 19.41512179 s
    # test 加上parse_docs 时间为 23.54273533821
    # dev  加上   按比例计算的结果 2.9428

    # print("final reulst for", evaluator_number, " files")
    # print(str(totalSim / evaluator_number))
    # print("key word: ", str(totalSim_word / evaluator_number))
    #
    # with open('log.txt', 'a+', encoding='utf-8') as f:
    #     f.write('\n')
    #     f.write(datetime.datetime.now().strftime('%Y-%m-%d  %H:%M:%S'))
    #     f.write(f"\nsen: {totalSim / evaluator_number}\n")
    #     f.write(f"key word: {totalSim_word / evaluator_number}")

    print("final reulst for", evaluator_number, " files")
    print('sentence:')
    # print(str(totalSim / evaluator_number))
    print('average: ' + str(totalSim / evaluator_number))

    # print("key word: ", str(totalSim_word / evaluator_number))
    print('key word: ')
    # print(str(totalSim / evaluator_number))
    print('average: ' + str(totalSim_word / evaluator_number))

    avg_tted_score = None
    if enable_tted:
        if tted_samples > 0:
            avg_tted_score = total_tted / tted_samples
            print("TTED average:", avg_tted_score)
        else:
            print("TTED average: n/a (no valid trees)")
        if return_tted:
            return totalSim / evaluator_number, totalSim_word / evaluator_number, avg_tted_score

    return totalSim / evaluator_number, totalSim_word / evaluator_number

    with open('log.txt', 'a+', encoding='utf-8') as f:
        f.write('\n')
        f.write(datetime.datetime.now().strftime('%Y-%m-%d  %H:%M:%S'))
        # f.write(f"\nsen: {totalSim / evaluator_number}\n")
        f.write("\nsentence:")
        f.write('\naverage: ' + str(totalSim / evaluator_number))
        # f.write(f"key word: {totalSim_word / evaluator_number}")
        f.write("\nkey word:")
        f.write('\naverage: ' + str(totalSim_word / evaluator_number))


def compare_generated_maps(benchmarks, generated_maps_dir):
    """
    Compare pre-generated .story maps with target .story maps using
    the same metrics as the main evaluation pipeline.
    """
    targets = [f for f in sorted(listdir(benchmarks)) if f.endswith(".story")]

    totalSim = 0.0
    totalSim_word = 0.0
    evaluator_number = 0
    missing = []

    for target in targets:
        target_path = join(benchmarks, target)
        generated_path = join(generated_maps_dir, target)

        if not isfile(generated_path):
            missing.append(target)
            continue

        target_pairs, target_word_pairs, _ = parse_docs(target_path)
        generated_pairs, generated_word_pairs, _ = parse_docs(generated_path)

        if len(target_pairs) <= 1 or len(generated_pairs) <= 1:
            print(f"skip {target}: not enough nodes for sentence-level comparison")
            continue

        if len(target_word_pairs) <= 1 or len(generated_word_pairs) <= 1:
            print(f"skip {target}: not enough nodes for keyword-level comparison")
            continue

        sim = compare_method(generated_pairs[:], target_pairs) / len(target_pairs)
        sim_word = compare_method(
            process_wordPairs(generated_word_pairs),
            process_wordPairs(target_word_pairs),
        ) / len(target_word_pairs)

        print(target, sim, sim_word)
        totalSim += sim
        totalSim_word += sim_word
        evaluator_number += 1

    print("evaluated files:", evaluator_number)
    if missing:
        print("missing generated files:", len(missing))

    if evaluator_number == 0:
        raise ValueError("No files were evaluated. Check folder paths and file formats.")

    sentence_avg = totalSim / evaluator_number
    keyword_avg = totalSim_word / evaluator_number
    print("sentence average:", sentence_avg)
    print("key word average:", keyword_avg)
    return sentence_avg, keyword_avg, missing
