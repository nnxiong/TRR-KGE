import json
import numpy as np
import os
import pickle
class Grapher(object):
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        # inv_relation_id的 key:value表示逆向关系与正相关系对应的id. such as, 230:0,表示逆向关系230对应正向关系0；0:230表示表示正向关系0对应逆向关系230
        self.inv_relation_id = dict()
        examples = pickle.load(open(dataset_dir + "stat", 'rb'))
        self.num_relations = num_relations = int(examples[1])
        for i in range(num_relations):
            self.inv_relation_id[i] = i + num_relations
        for i in range(num_relations, num_relations * 2):
            self.inv_relation_id[i] = i % num_relations

    def read_dataset(self):
        self.train_idx = self.read_data(self.dataset_dir + 'train.pickle')  # 既包含正向的边，又包含所有的逆向的边，r>230都是逆向
        self.valid_idx = self.read_data(self.dataset_dir + 'valid.pickle')
        self.test_idx = self.read_data(self.dataset_dir  + 'test.pickle')
        self.all_idx = np.vstack((self.train_idx, self.valid_idx, self.test_idx))
        print("Grapher initialized.")

    def read_data(self, file):
        examples = pickle.load(open(file, 'rb'))
        data = np.array(examples).astype(int)
        # TODO 则返回的数据集quads_idx对应的关系id大于230的都是逆向的关系
        data = self.add_inverses(data)
        return data

    def add_inverses(self, quads_idx):
        subs = quads_idx[:, 2]
        # TODO 则返回的quads_idx对应的大于230的都是逆向的关系
        rels = [self.inv_relation_id[x] for x in quads_idx[:, 1]]
        objs = quads_idx[:, 0]
        tss = quads_idx[:, 3]
        inv_quads_idx = np.column_stack((subs, rels, objs, tss))
        quads_idx = np.vstack((quads_idx, inv_quads_idx))
        return quads_idx

    def read_newtest(self):
        split_q = []
        with open(self.dataset_dir + 'test_new.txt', "r") as f:
            quads = f.readlines()
            for quad in quads:
                split_q.append([int(i) for i in quad.split("\t")])
        return np.array(split_q)

    def split_test(self):
        test_new_file = self.dataset_dir + "test_new.txt"
        if os.path.exists(test_new_file):
            os.remove(test_new_file)
        test_new = open(test_new_file, "a+")
        test_dl_file = self.dataset_dir + "test_dl.txt"
        if os.path.exists(test_dl_file):
            os.remove(test_dl_file)
        test_dl = open(test_dl_file, "a+")

        test_data = self.test_idx
        total0 = 0
        with open(self.dataset_dir + 'test_his_s_r_num.txt', 'r') as file:
            # 逐行读取文件内容
            for line, test_data in zip(file, test_data):
                # 分割每行数据
                data = line.strip()
                # 检查第一列为0或1且第二列小于10的条件
                if int(data) == 0:
                    test_dl.write("{}\t{}\t{}\t{}\n".format(test_data[0], test_data[1], test_data[2], test_data[3]))
                    total0 += 1
                # if int(data[0]) == 1:
                #     total1 += 1
                #     if int(data[1]) > 10:
                #         count1 += 1

                else:
                    test_new.write("{}\t{}\t{}\t{}\n".format(test_data[0], test_data[1], test_data[2], test_data[3]))
        test_new.close()
        test_dl.close()
        print(total0)

    def get_his_s_r_num(self):
        num_same_s_r = []
        for test_query in self.test_idx:
            # num_same_s_r.append(np.sum(
            #     (self.all_idx[:, 0] == test_query[0])
            #     * (self.all_idx[:, 1] == test_query[1])
            #     * (self.all_idx[:, 3] < test_query[3])
            # ))
            num_same_s_r.append(np.sum(
                (self.train_idx[:, 0] == test_query[0])
                * (self.train_idx[:, 1] == test_query[1])
                # * (self.train_idx[:, 3] < test_query[3])
            ))
        test_his_file = self.dataset_dir + 'test_his_s_r_num.txt'
        if os.path.exists(test_his_file):
            os.remove(test_his_file)
        with open(test_his_file, "w") as out:
            num_samples = len(num_same_s_r)
            for i in range(num_samples):
                # out.write("{}\t{}\t{}\n".format(num_same_s_r[i], num_same_s_r_o[i], all_rank[i]))
                out.write("{}\n".format(num_same_s_r[i]))
