# from collections import defaultdict
#
# # 假设我们有以下列表
# list_of_dicts = [
#     {'rule_type': 'type1', 'conf': 5, 'other_key': 'value1'},
#     {'rule_type': 'type2', 'conf': 3, 'other_key': 'value2'},
#     {'rule_type': 'type1', 'conf': 8, 'other_key': 'value3'},
#     {'rule_type': 'type2', 'conf': 9, 'other_key': 'value4'},
#     {'rule_type': 'type1', 'conf': 5, 'other_key': 'value1'},
#     {'rule_type': 'type2', 'conf': 3, 'other_key': 'value2'},
#     {'rule_type': 'type1', 'conf': 100, 'other_key': 'value3'},
#     {'rule_type': 'type2', 'conf': 9, 'other_key': 'value4'},
# ]
#
# # 使用defaultdict进行分组
# groups = defaultdict(list)
# for d in list_of_dicts:
#     groups[d['rule_type']].append(d)
#
# # 对每个组找到conf值最大的字典
# max_conf_dicts = []
# for group_name, group in groups.items():
#     max_conf_dict = max(group, key=lambda x: x['conf'])
#     max_conf_dicts.append(max_conf_dict)
#
# # max_conf_dicts 现在包含了每个rule_type对应的conf值最大的字典
# print(max_conf_dicts)


# from collections import defaultdict
#
# list_of_dicts = [
#     {'rule_type': 'type1', 'conf': 5, 'other_key': 'value1'},
# ]
#
# # 初始化分组
# groups = defaultdict(list)
# for d in list_of_dicts:
#     groups[d['rule_type']].append(d)
#
# # 保留每组中conf值最大的前两个字典
# max_conf_dicts = []
# for rule_type, group in groups.items():
#     # 对分组内的字典按conf值降序排序
#     sorted_group = sorted(group, key=lambda x: x['conf'], reverse=True)
#     # 取前两个字典（如果存在）
#     max_conf_dicts.extend(sorted_group[:2])
#
# print(max_conf_dicts)
# # 打印结果
# for rule_type, dicts in max_conf_dicts.items():
#     print(f"Rule type: {rule_type}")
#     for d in dicts:
#         print(d)
#     print()


# a=[]
# a[1] = 2
# print(a)


# import torch
# # #
# # # # 假设有两个向量v1和v2
# # # v1 = torch.tensor([1.0, 2.0, 3.0])
# # # v2 = torch.tensor([-1.0, -2.0, 3.0])
# # #
# # # # 计算欧氏距离
# # # # distance = torch.norm(v2 - v1)
# # # # print(distance)
# # #
# # # # 计算余弦相似度（1减去余弦距离）
# # # # cosine_similarity = torch.nn.functional.cosine_similarity(v1, v2, dim=1)
# # # # print(cosine_similarity.item())
# # #
# # # print(v1[1])

from tqdm import tqdm
import time

# 简单的for循环进度条
for i in tqdm(range(100)):
    time.sleep(0.01)  # 假设这里有一些需要时间的操作
"""
Provide military aid	185
Express intent to provide military aid	141
test:
Barack Obama	Provide military aid	Armed Rebel (Syria)	2013-05-11
Barack Obama	Provide military aid	Iraq	2014-08-12
Barack Obama	Provide military aid	Military (Ukraine)	2015-02-01
Barack Obama	Provide military aid	Japan	2009-11-17
train:
Barack Obama	Express intent to provide military aid	Iraq	2014-06-25
Barack Obama	Express intent to provide military aid	Armed Rebel (Syria)	2014-09-15
"""






