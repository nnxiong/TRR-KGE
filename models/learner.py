import argparse
from typing import Dict
import torch
from torch import optim
import codecs
import tqdm
from datasets import Dataset, Train
from models import *
from rule_utils import *
import os
import time
import numpy as np
import json
parser = argparse.ArgumentParser(description="Combine Rule")
parser.add_argument('--dataset', type=str, default= 'ICEWS14',help="Dataset name")
# parser.add_argument('--dataset', type=str, default= 'ICEWS05-15',help="Dataset name")
# parser.add_argument('--dataset', type=str, default= 'GDELT',help="Dataset name")
parser.add_argument('--model', type=str, default= 'CTRule')
parser.add_argument('--max_epochs', default=60, type=int,help="Number of epochs.")
parser.add_argument('--valid_freq', default=2, type=int,help="Number of epochs between each valid.")
parser.add_argument('--rank', default=1000, type=int,help="Factorization rank.") ###

parser.add_argument('--batch_size', default=1000, type=int,help="Batch size.")
parser.add_argument('--learning_rate', default=0.1, type=float,help="Learning rate")
parser.add_argument('--lambda1', default=0.06, type=float,help="Influence weight 1")

parser.add_argument('--emb_reg', default=3e-3, type=float)
parser.add_argument('--time_reg', default=3e-2, type=float)

parser.add_argument('--gpu', default=1, type=int,help="Use CUDA for training")
parser.add_argument('--cuda', type=str, default='cuda:0')
parser.add_argument("--rules_dir", default='../rule_result/', type=str)
parser.add_argument("--rule_type", "-rt", default="all", type=str)
parser.add_argument("--min_conf", "-mc", default=0.03, type=int)
parser.add_argument("--min_body_supp", "-mbs", default=2, type=int)

parser.add_argument("--rules", "-r", default="240624104521_maxlen2_top3_rules.json", type=str)

args = parser.parse_args()


if __name__ == '__main__':
    save_path = "results/{}/{}_{}_{}_{}_{}".format(args.dataset, args.dataset, args.model, args.rank, args.learning_rate, int(time.time()))
    print("rank:",args.rank, "learning_rate",args.learning_rate, 'lambda', args.lambda1, 'rules', args.rules)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    dataset = Dataset(args.dataset, is_cuda=True if args.gpu == 1 else False)
    fw = codecs.open("{}/log.txt".format(save_path), 'w')
    sizes = dataset.get_shape()
    # --- get the rules
    dir_path = args.rules_dir + args.dataset + "/"
    rules_dict = json.load(open(dir_path + args.rules))
    rules_dict = {int(k): v for k, v in rules_dict.items()}

    print("Rules statistics:")
    print("Number of relations with rules: ", len(rules_dict))
    print("Total number of rules: ", sum([len(v) for k, v in rules_dict.items()]))

    # ---
    model = CTRule(args, sizes, args.rank, rules_dict, is_cuda=True if args.gpu == 1 else False)
    # in case a user want to train on a non-cuda machine
    if args.gpu == 0:
        model = model.to('cpu')
    else:
        model = model.cuda()

    best_hits1 = 0
    best_res_test = {}
    opt = optim.Adagrad(model.parameters(), lr=args.learning_rate)
    # TODO obtain preprocessing data
    # neis_path_all, path_distance, neis_path_all_neis, path_distance_neis = load_merw(args)
    # neis_path_all, neis_timestamps, path_weight = load_merw(args)

    for epoch in range(args.max_epochs):
        examples = torch.from_numpy(
            dataset.get_train().astype('int64')
        ) #得到正向与反向训练集

        model.train()
        optimizer = Train(model, args.dataset, sizes[1] // 2, opt, batch_size=args.batch_size)
        mode = "Training"
        # optimizer.epoch(examples, args, mode, neis_path_all, neis_timestamps, path_weight, epoch)
        optimizer.epoch(examples, args, mode)

        def avg_both(mr, mrrs: Dict[str, float], hits: Dict[str, torch.FloatTensor]):
            m = (mrrs['lhs'] + mrrs['rhs']) / 2.
            h = (hits['lhs'] + hits['rhs']) / 2.
            mr = (mr['lhs'] + mr['rhs']) / 2.
            return {'MR':mr, 'MRR': m, 'hits@[1,3,10]': h}
        if epoch < 0 or (epoch + 1) % args.valid_freq == 0:
            model.eval()

            test, valid = [
                # avg_both(*dataset.eval(model, split, args, split, neis_path_all, neis_timestamps, path_weight, epoch))
                avg_both(*dataset.eval(model, split, args, split))
                for split in ['test', 'valid']
            ]
            print("valid: ", epoch, valid['MR'], valid['MRR'], valid['hits@[1,3,10]'])
            print("test: ", epoch, test['MR'], test['MRR'], test['hits@[1,3,10]'])
            fw.write("valid: epoch:{}, MR:{}, MRR:{}, Hist:{}\n".format(epoch, valid['MR'], valid['MRR'], valid['hits@[1,3,10]']))
            fw.write("test: epoch:{}, MR:{}, MRR:{}, Hist:{}\n".format(epoch, test['MR'], test['MRR'], test['hits@[1,3,10]']))
            if valid['hits@[1,3,10]'][0] > best_hits1:
                torch.save({'MRR':test['MRR'], 'Hist':test['hits@[1,3,10]'], 'MR':test['MR'], 'param':model.state_dict()}, '{}/best.pth'.format(save_path, args.model, args.dataset))
                print('best')
                best_hits1 = valid['hits@[1,3,10]'][0]
                best_res_test = [test['MR'], test['MRR'], test['hits@[1,3,10]']]

    fw.write("{}\t{}\t{}\t{}\t{}\n".format(best_res_test[0], best_res_test[1], best_res_test[2][0], best_res_test[2][1], best_res_test[2][2]))