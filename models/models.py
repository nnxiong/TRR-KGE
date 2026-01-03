from typing import Tuple, List, Dict
from contrastive_learning import ConLoss_his
# from his_embdding import EventEmbeddingModel
import torch
from torch import nn
import torch.nn.functional as F


class CTRule(nn.Module):
    def __init__(
            self, args, sizes: Tuple[int, int, int, int], rank: int, rules_dict,
            is_cuda: bool = False, cycle=120
    ):
        super(CTRule, self).__init__()
        self.model_name = "CTRule"
        self.cycle = cycle
        self.sizes = sizes
        self.rank = rank
        self.embeddings = nn.ModuleList([
            nn.Embedding(sizes[0], rank, sparse=True),  # 0 entityies (7128,1200*2)
            nn.Embedding(sizes[1], rank, sparse=True),  # 1 relaitions
            nn.Embedding(sizes[3], rank, sparse=True),  # 2 表示关系的timstamps
            nn.Embedding(sizes[3], rank, sparse=True),  # 3 表示实体的timstamps
            nn.Embedding(sizes[3], rank, sparse=True),
            nn.Embedding(sizes[3] // self.cycle + 1, rank, sparse=True),
            nn.Embedding(sizes[3] // self.cycle + 1, rank, sparse=True),  # 6 实体共享时间
        ])
        self.is_cuda = is_cuda
        self.emb_reg= args.emb_reg
        self.time_reg= args.time_reg
        self.contrastive_leanring_his = ConLoss_his(args).to(args.cuda)
        if rank % 2 != 0:
            raise "rank need to be devided by 2.."
        for i in range(len(self.embeddings)):
            torch.nn.init.xavier_uniform_(self.embeddings[i].weight.data)

        self.rules = rules_dict

    def mul4(self, emb1, emb2):
        a, b, c, d = torch.chunk(emb1, 4, dim=1)
        e, f, g, h = torch.chunk(emb2, 4, dim=1)
        return torch.cat(
            [(a * e - b * f - c * g - d * h), (b * e + a * f + c * h - d * g), (c * e + a * g + d * f - b * h),
             (d * e + a * h + b * g - c * f)], dim=1)

    def complex_mul(self, emb1, emb2):
        a, b = torch.chunk(emb1, 2, dim=1)
        c, d = torch.chunk(emb2, 2, dim=1)
        return torch.cat(((a * c + b * d), (a * d - b * c)), dim=1)

    def mul(self, emb1, emb2):
        a, b = torch.chunk(emb1, 2, dim=0)
        c, d = torch.chunk(emb2, 2, dim=0)
        return torch.cat([(a * c - b * d), (a * d + b * c)], dim=0)
    def mul1(self, emb1, emb2):
        a, b = torch.chunk(emb1, 2, dim=1)
        c, d = torch.chunk(emb2, 2, dim=1)
        return torch.cat([(a * c - b * d), (a * d + b * c)], dim=1)

    def forward(self, x, batch_history, mode, filters: Dict[Tuple[int, int, int], List[int]] = None):
        lhs = self.embeddings[0](x[:, 0])
        rel = self.embeddings[1](x[:, 1])
        rhs = self.embeddings[0](x[:, 2])

        base_ent = self.embeddings[6](torch.div(x[:, 3], self.cycle, rounding_mode='floor'))
        base = self.embeddings[5](torch.div(x[:, 3], self.cycle, rounding_mode='floor'))
        comp_time = self.embeddings[4](x[:, 3])
        time = self.embeddings[2](x[:, 3]) + base #2
        time_ent = self.embeddings[3](x[:, 3]) + base_ent #1

        rule_score = self.score(lhs, x[:, 1], rel, comp_time)
        rel_rule = self.complex_mul(rel, rule_score)
        rel_time = self.complex_mul(rel, comp_time)
        rel_ = rel + rel_time + rel_rule

        lhs_rel = self.mul4(torch.cat([lhs, time_ent], dim=1), torch.cat([rel_, time], dim=1))
        right = self.embeddings[0].weight.transpose(0, 1)
        a, b = lhs_rel[:, :self.rank], lhs_rel[:, self.rank:]
        if mode == 'Training':
            # get the contrastive_leanring_loss
            contrastive_leanring_loss = self.contrastive_leanring_his(x[:, 1], self.embeddings[1].weight, comp_time,
                                                                      batch_history)  # rel
            scores = (a + b) @ right
            enti_reg = self.learn_embedding((lhs, rhs))
            time_reg = self.learn_time((self.embeddings[2].weight[:-1],
                                        self.embeddings[3].weight[:-1],
                                        self.embeddings[4].weight[:-1],
                                        self.embeddings[5].weight[:-1],
                                        self.embeddings[6].weight[:-1]))

            return scores, enti_reg, time_reg, contrastive_leanring_loss
            # return scores, enti_reg, time_reg

        else:
            with torch.no_grad():
                scores = (a+b) @ right
                targets = []
                for (score, target_index) in zip(scores, x[:, 2]):
                    targets.append(score[target_index])
                targets = torch.tensor(targets).view(-1,1).to("cuda:0")
                # set filtered and true scores to -1e6 to be ignored
                # take care that scores are chunked
                for i, query in enumerate(x):
                    filter_out = filters[(query[0].item(), query[1].item(), query[3].item())] #并发事件
                    filter_out += [x[i, 2].item()] #b_begin是 n*batch_size # queries[b_begin + i, 2].item()准确结果
                    scores[i, torch.LongTensor(filter_out)] = -1e6
            return scores, targets


    def learn_time(self, time):
        l = 0
        weight = self.time_reg
        for f in time:
            rank = int(f.shape[1] / 2)
            ddiff = f[1:] - f[:-1]
            diff = torch.sqrt((ddiff[:, :rank] ** 2 + ddiff[:, rank:] ** 2)) ** 4
            l = l + weight * torch.sum(diff)
        return l / time[0].shape[0]

    def learn_embedding(self, embed):
        norm = 0
        weight = self.emb_reg
        if embed is not None:
            for em in embed[:3]:
                norm += weight * torch.sum(torch.abs(em) ** 3)
            return norm / embed[0].shape[0]

    def score(self, sub, rel, rel_em, time):
        rel_keys = list(self.rules.keys())
        rel_rules = []  # obtain the batch rules
        for i in rel:
            if i.item() in rel_keys:  # Check whether the query relation has rules
                rel_rules.append(self.rules[i.item()])
            else:
                rel_rules.append([])  # no rule
        score = torch.zeros([len(rel), self.rank]).cuda()
        score1, score2, score3, score4, score5 = torch.zeros(self.rank).cuda(), torch.zeros(self.rank).cuda(), torch.zeros(
            self.rank).cuda(), torch.zeros(self.rank).cuda(), torch.zeros(self.rank).cuda()  # .cuda()
        for (rel_rule, i) in zip(rel_rules, range(len(rel))):  # compute each relation score through its rule and sub
            if len(rel_rule) == 0:
                rel_sub = self.mul(rel_em[i], sub[i])
                sub_ = sub[i] + rel_sub
                score[i] = sub_
            else:
                score1.zero_()
                score2.zero_()
                score3.zero_()

                score4.zero_()
                score5.zero_()
                for rule in rel_rule:  # dic{}
                    if rule['rule_type'] == 'Transitive2':
                        body_rels = self.embeddings[1].weight[rule['body_rels'][0]] + self.embeddings[1].weight[
                            rule['body_rels'][1]]
                        reason = self.mul(time[i], body_rels) - rel_em[i]
                        score2 = reason * rule['conf'] + score2
                    elif rule['rule_type'] == 'Transitive3':
                        body_rels = self.embeddings[1].weight[rule['body_rels'][0]] + self.embeddings[1].weight[
                            rule['body_rels'][1]] + self.embeddings[1].weight[rule['body_rels'][2]]
                        reason = self.mul(time[i], body_rels) - rel_em[i]
                        score3 = reason * rule['conf'] + score3
                    elif rule['rule_type'] == 'Transitive4':
                        body_rels = self.embeddings[1].weight[rule['body_rels'][0]] + self.embeddings[1].weight[
                            rule['body_rels'][1]] + self.embeddings[1].weight[rule['body_rels'][2]] + self.embeddings[1].weight[rule['body_rels'][3]]
                        reason = self.mul(time[i], body_rels) - rel_em[i]
                        score4 = reason * rule['conf'] + score4
                    elif rule['rule_type'] == 'Transitive5':
                        body_rels = self.embeddings[1].weight[rule['body_rels'][0]] + self.embeddings[1].weight[
                            rule['body_rels'][1]] + self.embeddings[1].weight[rule['body_rels'][2]]+ self.embeddings[1].weight[rule['body_rels'][3]]+ self.embeddings[1].weight[rule['body_rels'][4]]
                        reason = self.mul(time[i], body_rels) - rel_em[i]
                        score5 = reason * rule['conf'] + score5
                    else:  # Symmetry, Inverse, Equivalent
                        reason = self.mul(time[i], self.embeddings[1].weight[rule['body_rels'][0]]) - rel_em[i]
                        score1 = reason * rule['conf'] + score1
                score[i] = score1 + score2 + score3

        return score