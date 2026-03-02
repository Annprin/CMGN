from evaluations import compare_generated_maps

sentence_avg, keyword_avg, missing = compare_generated_maps(
    "testing_final_20201012/testing_data_hmt_20201012_final/a_labeling_dev",
    "testing_final_20201012/testing_data_hmt_20201012_final/gpt_labeling_dev",
)
print(sentence_avg, keyword_avg, len(missing))
