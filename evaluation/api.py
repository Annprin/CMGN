import pickle
from evaluations import main

def evaluation_sim(
    my,
    type,
    model_type,
    enable_tted=False,
    tted_model_name="sentence-transformers/paraphrase-distilroberta-base-v2",
):
    benchmarks = "../testing_final_20201012/testing_data_hmt_20201012_final/a_labeling_dev/"

    my_results = dict()
    for e in my:
        id_, content, abs_, content_sent, abs_sent, _, _, probs = e

        if model_type == "rl_root" or model_type == "rl_root_b" or model_type == "rl_all" \
            or model_type == "last_h" or model_type == "self_attention" or model_type == "rank" \
            or model_type == "max_pooling" or model_type == "rl_root_b_N" or model_type == "gcn" \
                or model_type == "gcn_s":
            probs = probs.cpu().detach().numpy()
        my_results[id_] = [content_sent, probs]

    sim_threshold = 0.8
    if enable_tted:
        result = main(
            benchmarks,
            my_results,
            sim_threshold,
            enable_tted=True,
            tted_model_name=tted_model_name,
            return_tted=True,
        )
        if isinstance(result, tuple) and len(result) == 3:
            score, word_score, tted_score = result
            return score, word_score, tted_score
        if isinstance(result, tuple) and len(result) == 2:
            score, word_score = result
            return score, word_score, None
        raise ValueError(f"Unexpected evaluation result format: {type(result)} {result}")

    score, word_score = main(benchmarks, my_results, sim_threshold)
    return score, word_score
