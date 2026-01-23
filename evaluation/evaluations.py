from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from os import listdir
from os.path import isfile, join
import re
import rake
import pickle
from mind_map_generation import *
import rouge
import time
import datetime

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

def main(benchmarks, my_results, sim_threshold):
    cc = listdir(benchmarks)
    totalSim = 0
    totalSim_word = 0
    evaluator_number = 0
    cc.sort()
    total_second_phase_time = 0
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

            totalSim += sim / len(pairs2)
            totalSim_word += sim_word / len(wordPairs2)
            evaluator_number += 1

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

    with open('log.txt', 'a+', encoding='utf-8') as f:
        f.write('\n')
        f.write(datetime.datetime.now().strftime('%Y-%m-%d  %H:%M:%S'))
        # f.write(f"\nsen: {totalSim / evaluator_number}\n")
        f.write("\nsentence:")
        f.write('\naverage: ' + str(totalSim / evaluator_number))
        # f.write(f"key word: {totalSim_word / evaluator_number}")
        f.write("\nkey word:")
        f.write('\naverage: ' + str(totalSim_word / evaluator_number))

