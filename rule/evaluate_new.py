import json
import argparse
import numpy as np
import os
import rule_application as ra
from grapher import Grapher
from temporal_walk import store_edges
# from baseline import baseline_candidates, calculate_obj_distribution


parser = argparse.ArgumentParser()
# parser.add_argument("--dataset", "-d", default="", type=str)
parser.add_argument("--dataset", "-d", default="icews14", type=str)
parser.add_argument("--test_data", default="test", type=str)
# parser.add_argument("--candidates", "-c", default="", type=str)
parser.add_argument("--candidates", "-c", default="290524162139_n100_exp_sNone_cands_w-1_score_12[0.1,0.5]_test.json", type=str)
parsed = vars(parser.parse_args())


def filter_candidates(test_query, candidates, test_data):
    """
    Filter out those candidates that are also answers to the test query
    but not the correct answer.

    Parameters:
        test_query (np.ndarray): test_query
        candidates (dict): answer candidates with corresponding confidence scores
        test_data (np.ndarray): test dataset

    Returns:
        candidates (dict): filtered candidates
    """

    other_answers = test_data[
        (test_data[:, 0] == test_query[0])
        * (test_data[:, 1] == test_query[1])
        * (test_data[:, 2] != test_query[2])
        * (test_data[:, 3] == test_query[3])
    ]

    if len(other_answers):
        objects = other_answers[:, 2]
        for obj in objects:
            candidates.pop(obj, None)

    return candidates


def calculate_rank(test_query_answer, candidates, num_entities, setting="best"):
    """
    Calculate the rank of the correct answer for a test query.
    Depending on the setting, the average/best/worst rank is taken if there
    are several candidates with the same confidence score.

    Parameters:
        test_query_answer (int): test query answer
        candidates (dict): answer candidates with corresponding confidence scores
        num_entities (int): number of entities in the dataset
        setting (str): "average", "best", or "worst"

    Returns:
        rank (int): rank of the correct answer
    """

    rank = num_entities
    if test_query_answer in candidates:
        conf = candidates[test_query_answer]
        all_confs = list(candidates.values())
        ranks = [idx for idx, x in enumerate(all_confs) if x == conf]
        if setting == "average":
            rank = (ranks[0] + ranks[-1]) // 2 + 1
        elif setting == "best":
            rank = ranks[0] + 1
        elif setting == "worst":
            rank = ranks[-1] + 1

    return rank


dataset = parsed["dataset"]
candidates_file = parsed["candidates"]
dir_path = "../output/" + dataset + "/"
dataset_dir = "../data/" + dataset + "/"
data = Grapher(dataset_dir)
data.read_dataset()
num_entities = len(data.id2entity)
# test_data = data.test_idx if (parsed["test_data"] == "test") else data.valid_idx
learn_edges = store_edges(data.train_idx)
# obj_dist, rel_obj_dist = calculate_obj_distribution(data.train_idx, learn_edges)

train_data = data.train_idx
all_data = data.all_idx
all_candidates = json.load(open(dir_path + candidates_file)) #将json读取到的candidates全部转为int
all_candidates = {int(k): v for k, v in all_candidates.items()}
for k in all_candidates:
    all_candidates[k] = {int(cand): v for cand, v in all_candidates[k].items()}

# 消除all_candidates中预测分数  <= 0.2 的预测结果
# candidates = {k: v for k, v in all_candidates.items() if len(v) !=0 and list(v.values())[0] > 0.2}

# --------------------# --------------------
# for k, v in all_candidates.items():
#     if len(v) !=0 and list(v.values())[0] <= 0.2:
#         all_candidates[k] = {}
# --------------------# --------------------

# def read_test_new():
test_new = data.read_newtest()

hits_1 = 0
hits_3 = 0
hits_10 = 0
mrr = 0

num_same_s_r_o = []
num_same_s_r = []
all_rank = []

num_scores = 0
num_rank = 0

num_samples = len(test_new)
print("Evaluating " + candidates_file + ":")
test_rule_file = dataset_dir + "test_rule.txt"
if os.path.exists(test_rule_file):
    os.remove(test_rule_file)
test_rule, test_rule_len = open(test_rule_file, "a+"), 0
test_dl, test_dl_len = open(dataset_dir + "test_dl.txt", "a+"), 0
for i in range(num_samples):
    test_query = test_new[i]
    if all_candidates[i]:
        test_rule_len += 1
        candidates = all_candidates[i]
        # 这样写入包含正向与逆向
        test_rule.write("{}\t{}\t{}\t{}\n".format(test_query[0], test_query[1], test_query[2], test_query[3]))
        # filter_candidates是过滤掉候选答案中的不正确答案，因为(s, r, ?, t)中历史的？可以是多个，但是针对该样本的？只能是它自己这一个，其他的则需要过滤掉，这样的过滤可以提高H@N
        candidates = filter_candidates(test_query, candidates, test_new)
        rank = calculate_rank(test_query[2], candidates, num_entities)
        if rank:
            if rank <= 10:
                hits_10 += 1
                if rank <= 3:
                    hits_3 += 1
                    if rank == 1:
                        hits_1 += 1
            mrr += 1 / rank
    else:
        pass
        # no candidate in rules
        # test_dl_len += 1
        # test_dl.write("{}\t{}\t{}\t{}\n".format(test_query[0], test_query[1], test_query[2], test_query[3]))

print("have rule test length:",test_rule_len, "no rule test length:", test_dl_len)
test_rule.close()
test_dl.close()
num_samples = test_rule_len

hits_1 /= num_samples
hits_3 /= num_samples
hits_10 /= num_samples
mrr /= num_samples

print("Hits@1: ", round(hits_1, 6))
print("Hits@3: ", round(hits_3, 6))
print("Hits@10: ", round(hits_10, 6))
print("MRR: ", round(mrr, 6))

filename = candidates_file[:-5] + "_eval.txt"
with open(dir_path + filename, "w", encoding="utf-8") as fout:
    fout.write("Hits@1: " + str(round(hits_1, 6)) + "\n")
    fout.write("Hits@3: " + str(round(hits_3, 6)) + "\n")
    fout.write("Hits@10: " + str(round(hits_10, 6)) + "\n")
    fout.write("MRR: " + str(round(mrr, 6)))
