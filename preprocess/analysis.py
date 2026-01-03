import numpy as np
from mycode.grapher import Grapher
import json
dir_path = "../output/icews14/240424202540_r[1,2,3]_n200_exp_s12_cands_r[1,2,3]_w0_score_12[0.1,0.5]_his_s_r_rank.txt"

def analysize1():
    total0, count0 = 0, 0
    total1, count1 = 0, 0
    total2, count2 = 0, 0
    total3, count3 = 0, 0
    with open(dir_path, 'r') as file:
        # 逐行读取文件内容
        for line in file:
            # 分割每行数据
            data = line.strip().split("\t")
            # 检查第一列为0或1且第二列小于10的条件
            if int(data[0]) == 0:
                total0 += 1
                if int(data[1]) > 10:
                    count0 += 1
            # if int(data[0]) == 1:
            #     total1 += 1
            #     if int(data[1]) > 10:
            #         count1 += 1


            # # if int(data[0]) == 2:
            # #     total2 += 1
            # #     if int(data[1]) > 10:
            # #         count2 += 1
            # # if int(data[0]) == 3:
            # #     total3 += 1
            # #     if int(data[1])> 10:
            # #         count3 += 1

    print(total0, count0, total1, count1)
    dir_path2 = 'analysis.txt'
    with open(dir_path2, 'w') as file:
        file.write("{}\t{}\t{}\n".format('his_s_r', 'num_total', 'num_rank>10'))
        file.write("{}\t{}\t{}\n".format(0, total0, count0))
        file.write("{}\t{}\t{}\n".format(1, total1, count1))
        file.write("{}\t{}\t{}\n".format(2, total2, count2))
        file.write("{}\t{}\t{}\n".format(3, total3, count3))


def analysize2():
    num_his, rank1, rank3, rank10 = 0, 0, 0, 0
    with open(dir_path, 'r') as file:
        # 逐行读取文件内容
        for line in file:
            data = line.strip().split()
            if int(data[0]) >= 10:
                num_his += 1
                if int(data[1]) == 1:
                    rank1 += 1
                if int(data[1]) <= 3:
                    rank3 += 1
                if int(data[1]) <= 10:
                    rank10 += 1
    print(num_his, rank1, rank3, rank10) # 13724 4983 7271 9229

# compute history similarity
def rel_similarity(data_name):
    dataset_dir = "../data/" + data_name + "/"
    file = "../data/" + data_name + "/rel_similarity.json"
    data = Grapher(dataset_dir)
    data.read_dataset()
    train_data = data.train_idx
    rel_entities = {}
    rel_similarity = {}
    for rel in range(data.num_relations * 2):
        rel_entities[rel] = set([sub for sub in train_data[train_data[:, 1] == rel][:, 2]]) #set，不考虑重复
    # 计算相似度，包含本身
    for rel1, rel_en1 in rel_entities.items():
        similarity = []
        for rel2, rel_en2 in rel_entities.items():
            common = len(rel_en1.intersection(rel_en2))
            maximum_common = min(len(rel_en1), len(rel_en2))
            similarity.append({rel2: round(common / maximum_common, 4)} if maximum_common != 0 else {rel2: 0})  # 为零时需要另外处理，还是就是为0？
        rel_similarity[rel1] = similarity

    with open(file, "w", encoding="utf-8") as fout:
        json.dump(rel_similarity, fout)
    print("complete to get the similarity for relations")

if __name__ == '__main__':
    # analysize1()
    rel_similarity("icews0515")
