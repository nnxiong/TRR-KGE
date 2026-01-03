import json
import time
import argparse
import itertools
import numpy as np
from joblib import Parallel, delayed

import rule_application as ra
from grapher import Grapher
from rule_mining import rules_statistics

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", "-d", default="icews14", type=str)
parser.add_argument("--train_data", default="test", type=str)
parser.add_argument("--rules", "-r", default="290524162139_n100_exp_sNone_rules.json", type=str)
parser.add_argument("--rule_type", "-rt", default="all", type=str)
parser.add_argument("--window", "-w", default=-1, type=int)
parser.add_argument("--top_k", default=20, type=int)
parser.add_argument("--num_processes", "-p", default=8, type=int)
parsed = vars(parser.parse_args())

dataset = parsed["dataset"]
rules_file = parsed["rules"]
window = parsed["window"]
top_k = parsed["top_k"]
num_processes = parsed["num_processes"]
rule_type = parsed["rule_type"]


dataset_dir = "../data/" + dataset + "/"
dir_path = "../output/" + dataset + "/"
data = Grapher(dataset_dir)
data.read_dataset()


# if parsed["train_data"] == "train":
#     train_data = data.train_idx
# elif parsed["train_data"] == "test":
#     train_data = data.test_idx
# else:
#     train_data = data.valid_idx
#
train_data = data.read_newtest()

rules_dict = json.load(open(dir_path + rules_file))
rules_dict = {int(k): v for k, v in rules_dict.items()}
print("Rules statistics:")
rules_statistics(rules_dict)

rules_dict = ra.filter_rules(rules_dict, min_conf=0.01, min_body_supp=2, rule_type=rule_type)
# rules_dict = ra.filter_rules(rules_dict, min_conf=0.03, min_body_supp=2, rule_type=rule_type)

print("Rules statistics after pruning:")
rules_statistics(rules_dict)


args = [[0.1, 0.5]]


def score_12(rule, cands_walks, test_query_ts, lmbda, a):
    max_cands_ts = max(cands_walks["timestamp_0"])
    score = a * (rule["rule_supp"] / (rule["body_supp"])) + (1 - a) * np.exp(lmbda * (max_cands_ts - test_query_ts))
    return score

score_func = score_12

def apply_rules(i, num_queries):
    print("Start process", i, "...")
    all_candidates = [dict() for _ in range(len(args))]
    no_cands_counter = 0

    num_rest_queries = len(train_data) - (i + 1) * num_queries
    if num_rest_queries >= num_queries:
        train_queries_idx = range(i * num_queries, (i + 1) * num_queries)
    else:
        train_queries_idx = range(i * num_queries, len(train_data))

    cur_ts = train_data[train_queries_idx[0]][3]
    edges = ra.get_window_edges(data.all_idx, cur_ts, window)

    it_start = time.time()
    for j in train_queries_idx:
        train_query = train_data[j]
        cands_dict = [dict() for _ in range(len(args))]

        if train_query[3] != cur_ts:
            cur_ts = train_query[3]
            edges = ra.get_window_edges(data.all_idx, cur_ts, window)

        if train_query[1] in rules_dict:
            dicts_idx = list(range(len(args)))
            for rule in rules_dict[train_query[1]]:
                walk_edges = ra.match_body_relations(rule, edges, train_query[0])

                if 0 not in [len(x) for x in walk_edges]:
                    rule_walks = ra.get_walks(rule, walk_edges)
                    if rule["var_constraints"]:
                        rule_walks = ra.check_var_constraints(
                            rule["var_constraints"], rule_walks
                        )

                    if not rule_walks.empty:
                        cands_dict = ra.get_candidates(
                            rule,
                            rule_walks,
                            cur_ts,
                            cands_dict,
                            score_func,
                            args,
                            dicts_idx,
                        )
                        for s in dicts_idx:
                            cands_dict[s] = {
                                x: sorted(cands_dict[s][x], reverse=True)
                                for x in cands_dict[s].keys()
                            }
                            cands_dict[s] = dict(
                                sorted(
                                    cands_dict[s].items(),
                                    key=lambda item: item[1],
                                    reverse=True,
                                )
                            )
                            top_k_scores = [v for _, v in cands_dict[s].items()][:top_k]
                            unique_scores = list(
                                scores for scores, _ in itertools.groupby(top_k_scores)
                            )
                            if len(unique_scores) >= top_k:
                                dicts_idx.remove(s)
                        if not dicts_idx:
                            break

            if cands_dict[0]:
                for s in range(len(args)):
                    scores = list(
                        map(
                            lambda x: 1 - np.prod(1 - np.array(x)),
                            cands_dict[s].values(),
                        )
                    )
                    cands_scores = dict(zip(cands_dict[s].keys(), scores))
                    noisy_or_cands = dict(
                        sorted(cands_scores.items(), key=lambda x: x[1], reverse=True)
                    )
                    all_candidates[s][j] = noisy_or_cands
            else:
                no_cands_counter += 1
                for s in range(len(args)):
                    all_candidates[s][j] = dict()

        else:
            no_cands_counter += 1
            for s in range(len(args)):
                all_candidates[s][j] = dict()

        if not (j - train_queries_idx[0] + 1) % 100:
            it_end = time.time()
            it_time = round(it_end - it_start, 6)
            print(
                "Process {0}: test samples finished: {1}/{2}, {3} sec".format(
                    i, j - train_queries_idx[0] + 1, len(train_queries_idx), it_time
                )
            )
            it_start = time.time()

    return all_candidates, no_cands_counter


start = time.time()
num_queries = len(train_data) // num_processes
output = Parallel(n_jobs=num_processes)(
    delayed(apply_rules)(i, num_queries) for i in range(num_processes)
)
end = time.time()

final_all_candidates = [dict() for _ in range(len(args))]
for s in range(len(args)):
    for i in range(num_processes):
        final_all_candidates[s].update(output[i][0][s])
        output[i][0][s].clear()

final_no_cands_counter = 0
for i in range(num_processes):
    final_no_cands_counter += output[i][1]

total_time = round(end - start, 6)
print("Application finished in {} seconds.".format(total_time))
print("No candidates: ", final_no_cands_counter, " queries")

for s in range(len(args)):
    score_func_str = score_func.__name__ + str(args[s])
    score_func_str = score_func_str.replace(" ", "")
    ra.save_candidates(
        rules_file,
        dir_path,
        final_all_candidates[s],
        window,
        score_func_str,
        parsed["train_data"]
    )
