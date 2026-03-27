# -*- coding: utf-8 -*-
import re


def _normalize_text(text):
    return re.sub(r"\s+", " ", text.strip().lower())


def _get_rst_parser(preferred="auto", **kwargs):
    """
    Returns (name, parser) where parser supports .parse(text) or is a module with parse(text).
    """
    tried = []
    if preferred in ("auto", "isanlp"):
        try:
            from isanlp_rst.parser import Parser  # type: ignore
            hf_model_name = kwargs.get("hf_model_name", "tchewik/isanlp_rst_v3")
            hf_model_version = kwargs.get("hf_model_version", "gumrrg")
            cuda_device = kwargs.get("cuda_device", -1)
            return "isanlp", Parser(
                hf_model_name=hf_model_name,
                hf_model_version=hf_model_version,
                cuda_device=cuda_device,
            )
        except Exception as exc:
            tried.append(f"isanlp ({exc})")
    if preferred in ("auto", "rstparser"):
        try:
            import rstparser  # type: ignore
            if hasattr(rstparser, "RSTParser"):
                return "rstparser", rstparser.RSTParser()
            if hasattr(rstparser, "Parser"):
                return "rstparser", rstparser.Parser()
            if hasattr(rstparser, "parse"):
                return "rstparser", rstparser
            tried.append("rstparser (no parser class)")
        except Exception as exc:
            tried.append(f"rstparser ({exc})")
    if preferred in ("auto", "rst_parser"):
        try:
            from rst_parser import RSTParser  # type: ignore
            return "rst_parser", RSTParser()
        except Exception as exc:
            tried.append(f"rst_parser ({exc})")

    raise RuntimeError(
        "RST parser not available. Tried: "
        + "; ".join(tried)
        + ". Install one of: isanlp_rst, rstparser or rst_parser, or plug in your parser."
    )


def _extract_edus_and_links(result):
    """
    Attempt to extract (edus, links) from parser output.
    - edus: list of strings
    - links: list of (parent_idx, child_idx) 0-based indices
    """
    edus = None
    links = None

    # IsaNLP RST parser output: dict with 'rst' list of DiscourseUnit roots.
    if isinstance(result, dict) and "rst" in result:
        root = result["rst"][0] if result["rst"] else None
        if root is None:
            return [], []

        edus = []
        links = set()

        def _get_attr(node, name):
            if isinstance(node, dict):
                return node.get(name)
            return getattr(node, name, None)

        def _is_leaf(node):
            return _get_attr(node, "left") is None and _get_attr(node, "right") is None

        def _get_text(node):
            text = _get_attr(node, "text")
            return "" if text is None else str(text)

        def _collect(node):
            if node is None:
                return []
            if _is_leaf(node):
                idx = len(edus)
                edus.append(_get_text(node))
                return [idx]

            left = _get_attr(node, "left")
            right = _get_attr(node, "right")
            left_idxs = _collect(left)
            right_idxs = _collect(right)

            nuclearity = _get_attr(node, "nuclearity")
            if nuclearity == "NS":
                nucleus, satellite = left_idxs, right_idxs
            elif nuclearity == "SN":
                nucleus, satellite = right_idxs, left_idxs
            else:
                nucleus, satellite = left_idxs, right_idxs

            for p in nucleus:
                for c in satellite:
                    links.add((p, c))
            if nuclearity == "NN":
                for p in satellite:
                    for c in nucleus:
                        links.add((p, c))

            return left_idxs + right_idxs

        _collect(root)
        return edus, list(links)

    if isinstance(result, tuple) and len(result) == 2:
        edus, links = result
    elif isinstance(result, dict):
        edus = result.get("edus") or result.get("segments") or result.get("edu")
        links = result.get("relations") or result.get("links") or result.get("edges")
    else:
        if hasattr(result, "edus"):
            edus = getattr(result, "edus")
        if hasattr(result, "relations"):
            links = getattr(result, "relations")
        elif hasattr(result, "links"):
            links = getattr(result, "links")
        elif hasattr(result, "edges"):
            links = getattr(result, "edges")

    if edus is None or links is None:
        raise RuntimeError(
            "RST parser output does not expose edus/relations. "
            "Please adapt _extract_edus_and_links for your parser."
        )

    # Normalize links to list of (parent, child)
    norm_links = []
    for rel in links:
        if isinstance(rel, (list, tuple)) and len(rel) >= 2:
            parent, child = rel[0], rel[1]
        elif isinstance(rel, dict):
            parent = rel.get("parent")
            child = rel.get("child")
        else:
            continue
        if parent is None or child is None:
            continue
        norm_links.append((parent, child))

    # Convert 1-based to 0-based if needed
    if norm_links:
        max_idx = max(max(p, c) for p, c in norm_links)
        if max_idx == len(edus):
            norm_links = [(p - 1, c - 1) for p, c in norm_links]

    return list(edus), norm_links


def _sentence_spans(sentences):
    spans = []
    cursor = 0
    for s in sentences:
        s_clean = s.strip()
        start = cursor
        end = start + len(s_clean)
        spans.append((start, end))
        cursor = end + 1  # space join
    return spans


def _align_edus_to_sentences(sentences, edus):
    """
    Map each EDU to a sentence index by locating EDU text in the joined text.
    """
    full_text = " ".join(s.strip() for s in sentences)
    spans = _sentence_spans(sentences)

    edu_to_sent = []
    cursor = 0
    full_lower = full_text.lower()
    for edu in edus:
        edu_text = edu.strip()
        if not edu_text:
            edu_to_sent.append(0)
            continue
        edu_lower = edu_text.lower()
        pos = full_lower.find(edu_lower, cursor)
        if pos == -1:
            pos = full_lower.find(edu_lower)
        if pos == -1:
            # fallback: pick sentence with max token overlap
            tokens = set(_normalize_text(edu_text).split())
            best_idx = 0
            best_score = -1
            for idx, sent in enumerate(sentences):
                score = len(tokens.intersection(_normalize_text(sent).split()))
                if score > best_score:
                    best_score = score
                    best_idx = idx
            edu_to_sent.append(best_idx)
            continue
        cursor = pos + len(edu_lower)

        sent_idx = 0
        for i, (s_start, s_end) in enumerate(spans):
            if s_start <= pos < s_end:
                sent_idx = i
                break
        edu_to_sent.append(sent_idx)
    return edu_to_sent


def build_labels_with_rst(data, parser_name="auto", parser_kwargs=None):
    """
    Build sentence-level labels using an RST parser.
    Returns list of [id, NxN float32 numpy arrays] like distilbert labels.
    """
    import numpy as np

    parser_kwargs = parser_kwargs or {}
    name, parser = _get_rst_parser(parser_name, **parser_kwargs)

    labels = []
    try:
        from tqdm import tqdm
        iterator = tqdm(data, total=len(data))
    except Exception:
        iterator = data
    for each in iterator:
        doc_id = each[0]
        content_sents = each[7]
        n = len(content_sents)
        if n == 0:
            labels.append([doc_id, np.zeros((0, 0), dtype=np.float32)])
            continue

        text = " ".join(s.strip() for s in content_sents)
        if hasattr(parser, "parse"):
            result = parser.parse(text)
        elif hasattr(parser, "__call__"):
            result = parser(text)
        elif hasattr(parser, "run"):
            result = parser.run(text)
        else:
            raise RuntimeError("RST parser does not expose parse/call/run")

        edus, links = _extract_edus_and_links(result)
        edu_to_sent = _align_edus_to_sentences(content_sents, edus)

        mat = np.zeros((n, n), dtype=np.float32)
        np.fill_diagonal(mat, 1.0)
        for parent, child in links:
            if parent < 0 or child < 0:
                continue
            if parent >= len(edu_to_sent) or child >= len(edu_to_sent):
                continue
            p_sent = edu_to_sent[parent]
            c_sent = edu_to_sent[child]
            if p_sent == c_sent:
                continue
            mat[p_sent, c_sent] = 1.0

        labels.append([doc_id, mat])

    return labels
