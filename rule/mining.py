import time
import argparse
import numpy as np
from datetime import datetime
from joblib import Parallel, delayed

from grapher2 import Grapher
from temporal_walk import Temporal_Walk
from rule_mining import Rule_Mining, rules_statistics

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", "-d", default="ICEWS14", type=str)
# parser.add_argument("--dataset", "-d", default="ICEWS05-15", type=str)
# parser.add_argument("--rule_lengths", "-l", default="3", type=int, nargs="+")
# parser.add_argument("--rule_lengths", "-l", default="4", type=int, nargs="+")
parser.add_argument("--rule_lengths", "-l", default="5", type=int, nargs="+")
parser.add_argument("--num_walks", "-n", default="100", type=int)
parser.add_argument("--transition_distr", "-t", default="exp", type=str)
parser.add_argument("--num_processes", "-p", default=10, type=int)
parser.add_argument("--seed", "-s", default=None, type=int)
# parser.add_argument("--top", "-top", default=1, type=int)
parser.add_argument("--top", "-top", default=2, type=int)
parsed = vars(parser.parse_args())
dataset = parsed["dataset"]
rule_lengths = parsed["rule_lengths"]
# rule_lengths = [rule_lengths] if (type(rule_lengths) == int) else rule_lengths
num_walks = parsed["num_walks"]
transition_distr = parsed["transition_distr"]
num_processes = parsed["num_processes"]
seed = parsed["seed"]

dataset_dir = "../data/" + dataset + "/"
data = Grapher(dataset_dir)

# data.get_data2idx() #预处理data
data.read_dataset()
# data.get_his_s_r_num() # get the history of test
# data.split_test() #spilt test with test_new and dl

temporal_walk = Temporal_Walk(data.train_idx, data.inv_relation_id, transition_distr)
rm = Rule_Mining(temporal_walk.edges, data.inv_relation_id, dataset)
all_relations = sorted(temporal_walk.edges)

def learn_rules(i, num_relations):
    if seed:
        np.random.seed(seed)
    num_rest_relations = len(all_relations) - (i + 1) * num_relations
    if num_rest_relations >= num_relations:
        relations_idx = range(i * num_relations, (i + 1) * num_relations)
    else:
        relations_idx = range(i * num_relations, len(all_relations))

    num_rules = [0]
    # 1. Symmetry 对称(hop1)
    # 2. Inverse 反转(hop1)
    # 3. Equivalent 等价 (hop1)
    # 4. Transitive2  传递（hop2）
    # 5. Transitive3  传递（hop3）
    types = ["Symmetry", "Inverse", "Equivalent", "Transitive"]
    # types = ["Transitive"]
    for k in relations_idx:
        rel = all_relations[k]
        for type in types:
            if type == "Transitive":
                # Chain规则可以有多条路径规则
                for length in range(2, rule_lengths+1):
                    it_start = time.time()
                    for _ in range(num_walks):
                        # # tail start
                        # walk_successful, walk_tail = temporal_walk.sample_walk(length + 1, rel, type,'tail')  # TODO 由 tail 出发，用于创建多条路径
                        # if walk_successful:
                        #     rl.create_rule(walk_tail,'tail')  # TODO create rule for each walk, it contains rule support, and body support and so on.
                        # # head start
                        # walk_successful, walk_head = temporal_walk.sample_walk(length + 1, rel, type, 'head')  # TODO 由 head 出发，用于创建多条路径
                        # if walk_successful:
                        #     rl.create_rule(walk_head, 'head')
                        walk_successful, walk = temporal_walk.sample_walk(length + 1, rel, type)
                        if walk_successful:
                            rm.create_rule(walk)

                    it_end = time.time()
                    it_time = round(it_end - it_start, 6)
                    num_rules.append(sum([len(v) for k, v in rm.rules_dict.items()]) // 2)
                    num_new_rules = num_rules[-1] - num_rules[-2]
                    print(
                        "Process {0}: relation {1}/{2}, type {3}: {4} sec, {5} rules".format(
                            i,
                            k - relations_idx[0] + 1,
                            len(relations_idx),
                            type + str(length),
                            it_time,
                            num_new_rules,
                        )
                    )
            else:
                it_start = time.time()
                for _ in range(num_walks):
                    walk_successful, walk = temporal_walk.sample_walk(2, rel, type)
                    if walk_successful:
                        #print(type)
                        rm.create_rule(walk)
                it_end = time.time()
                it_time = round(it_end - it_start, 6)
                num_rules.append(sum([len(v) for k, v in rm.rules_dict.items()]) // 2)
                num_new_rules = num_rules[-1] - num_rules[-2]
                print(
                    "Process {0}: relation {1}/{2}, type {3}: {4} sec, {5} rules".format(
                        i,
                        k - relations_idx[0] + 1,
                        len(relations_idx),
                        type,
                        it_time,
                        num_new_rules,
                    )
                )

    return rm.rules_dict


start = time.time()
# 没个进程处理的关系数量
num_relations = len(all_relations) // num_processes
# num_processes进程数
# Parallel(n_jobs=num_processes)：创建一个并行计算对象，指定了要使用的进程数量。
# delayed(learn_rules)(i, num_relations) for i in range(num_processes)：
# 这是一个生成器表达式，它为每个进程生成一个调用learn_rules(i, num_relations)的延迟调用对象。
# 结果将被收集到一个列表中
output = Parallel(n_jobs=num_processes)(
    delayed(learn_rules)(i, num_relations) for i in range(num_processes)
)
end = time.time()

all_rules = output[0]
for i in range(1, num_processes):
    all_rules.update(output[i])

total_time = round(end - start, 6)
print("Learning finished in {} seconds.".format(total_time))

rm.rules_dict = all_rules
rm.sort_rules_dict(parsed['top'])
dt = datetime.now()
dt = dt.strftime("%d%m%y%H%M%S")
# rm.save_rules(dt, num_walks, transition_distr, seed)
rm.save_rules(dt, rule_lengths, parsed['top'])
#rm.save_rules_verbalized(dt, num_walks, transition_distr, seed)
rules_statistics(rm.rules_dict)
