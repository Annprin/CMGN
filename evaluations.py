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
import numpy as np
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

def _default_output_dir(benchmarks):
    bench = benchmarks.rstrip("/")
    if "a_labeling_" in bench:
        return bench.replace("a_labeling_", "a_generated_", 1)
    return bench + "_generated"


def main(
    benchmarks,
    my_results,
    sim_threshold,
    enable_tted=False,
    tted_model_name="sentence-transformers/paraphrase-distilroberta-base-v2",
    return_tted=False,
    output_dir=None
):
    cc = listdir(benchmarks)
    totalSim = 0
    totalSim_word = 00
    total_tted = 0.0
    tted_samples = 0
    evaluator_number = 0
    encoder = None
    cc.sort()
    output_dir = output_dir or _default_output_dir(benchmarks)
    makedirs(output_dir, exist_ok=True)

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
            # print(target)
            # if target == '12.story':
            #     continue
            pairs2, wordPairs2, length_threshold = parse_docs(join(benchmarks, target))
            # print(pairs2)
            # print(wordPairs2)

            target_id = target[0: target.find('.story')]
            sents, prob_matrix = my_results[target_id]

            pairs, wordPairs, story_content = my_generate_mindmap(
                sents, prob_matrix, len(pairs2), sim_threshold, length_threshold, return_story=True
            )
            with open(join(output_dir, target), "w", encoding="utf-8") as f:
                f.write(story_content)
            # print(f'pairs = {pairs}')
            tmp_pairs = pairs[:]
            # print(f'tmp_pairs = {tmp_pairs}')
            sim = compare_method(tmp_pairs, pairs2)


            wordPairs = process_wordPairs(wordPairs)
            wordPairs2 = process_wordPairs(wordPairs2)
            sim_word = compare_method(wordPairs, wordPairs2)
            print(target_id, sim/len(pairs2), sim_word/len(wordPairs2))


            if enable_tted and encoder is not None:
                reference_tree = _story_to_text_tree(join(benchmarks, target))
                generated_tree = _pairs_to_text_tree(pairs)
                # print("---------||||||-------")
                # print(pairs)
                # print(generated_tree)
                # print("----------------")
                # print(reference_tree)
                # print("---------||||||-------")
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


def _tted_encoder(text_or_list):
    if isinstance(text_or_list, str):
        return _hash_sentence_embedding(text_or_list)
    return [_hash_sentence_embedding(t) for t in text_or_list]


def _cosine_dist(a, b):
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0 or b_norm == 0:
        return 1.0
    sim = float(np.dot(a, b) / (a_norm * b_norm))
    return 1.0 - sim


def _extract_tagged_nodes_tolerant(content):
    highlight_blocks = re.findall(re.compile("<highlight>.*?</highlight>", re.DOTALL), content, flags=0)
    if highlight_blocks:
        block = highlight_blocks[0]
    else:
        block = content

    # 1) strict format: <T1>...</T1>

    strict_matches = [
        (m.group(1), re.sub(r"\s+", " ", m.group(2)).strip())
        for m in re.finditer(r"<(T[\d\.]+)>(.*?)</\1>", block, flags=re.DOTALL)
    ]

    if strict_matches:
        return strict_matches

    # 2) tolerant line format: <T1>text (without closing tag)
    tolerant_matches = []
    for line in block.splitlines():
        m = re.match(r"^\s*<(T[\d\.]+)>\s*(.*?)\s*$", line)
        if not m:
            continue
        tag = m.group(1)
        text = m.group(2).strip()
        if text:
            tolerant_matches.append((tag, text))
    return tolerant_matches


def _simple_tokens(text):
    return re.findall(r"[A-Za-z0-9']+", text.lower())


def _parse_docs_tolerant(filename):
    # Try original parser first.
    try:
        pairs, word_pairs, length_threshold = parse_docs(filename)
        if len(pairs) > 1:
            return pairs, word_pairs, length_threshold
    except Exception:
        pass

    with open(filename, "r", encoding="utf-8") as f:
        content = f.read()

    nodes = _extract_tagged_nodes_tolerant(content)
    if not nodes:
        return [], [], 0

    tags = [tag.replace("T", "", 1) if tag.startswith("T") else tag for tag, _ in nodes]
    texts = [text for _, text in nodes]
    tag_to_text = {t: c for t, c in zip(tags, texts)}

    pairs = []
    word_pairs = []
    for t in tags:
        cur_text = tag_to_text[t]
        cur_tokens = _simple_tokens(cur_text)
        if "." not in t:
            pairs.append([[], cur_text])
            word_pairs.append([[], cur_tokens])
            continue

        father = t[:-2]
        while father and father not in tag_to_text:
            father = father[:-2]
        if not father:
            continue

        father_text = tag_to_text[father]
        father_tokens = _simple_tokens(father_text)
        if father_text == cur_text:
            continue
        pairs.append([father_text, cur_text])
        word_pairs.append([father_tokens, cur_tokens])

    return pairs, word_pairs, len(texts)


def _story_to_texttree(story_path, TextTreeCls):
    with open(story_path, "r", encoding="utf-8") as f:
        content = f.read()

    matches = _extract_tagged_nodes_tolerant(content)
    if not matches:
        raise ValueError(f"Cannot parse tagged tree from {story_path}")

    tag_to_idx = {}
    nodes = []
    adj = []
    for tag, text in matches:
        tag_to_idx[tag] = len(nodes)
        nodes.append(text)
        adj.append([])

    root_children = []
    for tag, _ in matches:
        cur_idx = tag_to_idx[tag]
        if "." in tag:
            parent = tag.rsplit(".", 1)[0]
            if parent in tag_to_idx:
                adj[tag_to_idx[parent]].append(cur_idx)
        else:
            root_children.append(cur_idx)

    # Force a single-root tree for TTED.
    new_nodes = ["[ROOT]"] + nodes
    new_adj = [[i + 1 for i in root_children]] + [[c + 1 for c in children] for children in adj]
    # print(new_adj)
    return TextTreeCls(new_nodes, new_adj)


def compare_generated_maps(benchmarks, generated_maps_dir, compute_tted=False):
    """
    Compare pre-generated .story maps with target .story maps using
    the same metrics as the main evaluation pipeline.
    """
    targets = [f for f in sorted(listdir(benchmarks)) if f.endswith(".story")]

    totalSim = 0.0
    totalSim_word = 0.0
    evaluator_number = 0
    missing = []

    tted_available = False
    tted_error = None
    avg_tted_fn = None
    TextTreeCls = None
    if compute_tted:
        try:
            tted_code_dir = join(os.path.dirname(__file__), "TTED", "text-tree-distance", "code")
            if tted_code_dir not in sys.path:
                sys.path.append(tted_code_dir)
            from tted.computation import avg_tted as _avg_tted
            from tted.tree_format import TextTree as _TextTree
            avg_tted_fn = _avg_tted
            TextTreeCls = _TextTree
            tted_available = True
        except Exception as e:
            tted_error = e
            print(f"TTED is unavailable ({type(e).__name__}: {e}). Install TTED deps to enable this metric.")

    total_tted = 0.0
    tted_count = 0

    for target in targets:
        target_path = join(benchmarks, target)
        generated_path = join(generated_maps_dir, target)

        if not isfile(generated_path):
            missing.append(target)
            continue

        target_pairs, target_word_pairs, _ = _parse_docs_tolerant(target_path)
        generated_pairs, generated_word_pairs, _ = _parse_docs_tolerant(generated_path)

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

        if tted_available:
            try:
                target_tree = _story_to_texttree(target_path, TextTreeCls)
                generated_tree = _story_to_texttree(generated_path, TextTreeCls)
                tted_val = avg_tted_fn(
                    generated_tree, target_tree, _tted_encoder, _cosine_dist, unordered=True, use_context=False
                )
                total_tted += tted_val
                tted_count += 1
                print(target, sim, sim_word, tted_val)
            except Exception as e:
                print(f"{target} {sim} {sim_word} TTED_error={type(e).__name__}: {e}")
        else:
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
    if compute_tted:
        tted_avg = total_tted / tted_count if tted_count > 0 else None
        if tted_avg is not None:
            print("avg tted:", tted_avg)
        elif tted_error is None:
            print("avg tted: not computed")
        return sentence_avg, keyword_avg, missing, tted_avg
    return sentence_avg, keyword_avg, missing




