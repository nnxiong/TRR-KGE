import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class ConLoss_his(nn.Module):

    def __init__(self, args, temperature = 1, scale_by_temperature=True):
        super(ConLoss_his, self).__init__()
        self.device = args.cuda
        self.temperature = temperature
        self.scale_by_temperature = scale_by_temperature

    def forward(self, entities, entity_embedding, time_ent, history):
        # 与历史实体的对比学习
        # 分别获得当前批次的每个实体与其历史实体,得到不重复实体
        # 在entity_embedding上，以entitys和neis_all构建mask
        features = entity_embedding
        # set_entities = list(set(entities.tolist()))
        # num_dif = len(set_entities)
        list_entities = entities.tolist()
        set_entities = []
        # set_en_time = []
        his_entities, his_time = [], []  # 第一维度为len(set_entities)
        for index, en in enumerate(list_entities):
            if en not in set_entities:
                set_entities.append(en)
                # set_en_time.append(time_ent[index])
                # his_en, his_time = self.batch_his_e_t(history[index])
                his_entities.append(history[index])
        # set_en_time = torch.stack(set_en_time).to(self.device)
        num_dif = len(set_entities)
        similar_entities = []  # get the samilar entities ids. shape(num_dif, )   让交集越多的两者相乘num_intersection
        # 历史仍是(s, r)相同的为历史，其他的不为历史，但以(s, r, t)为key来存历史，如果t2>t1，那么(s, r, t1)的历史应该是(s, r, t2)的子集。
        for en1 in his_entities:
            similar_ = []
            if (len(en1) != 0):
                for index, en2 in enumerate(his_entities):
                    num_intersection = len(set(en1).intersection(set(en2)))  # 求交集数量  best 2
                    if (num_intersection >= 5):
                        similar_.append((index, num_intersection))  # 记录索引与交集数量
            similar_entities.append(similar_)

        # 构建batch+neis特征
        features_dim = entity_embedding.shape[1]
        batch_embed = torch.zeros(num_dif, features_dim).to(self.device)
        for index, entity in enumerate(set_entities):
            batch_embed[index] = features[entity]

        # batch_embed = torch.cat([batch_embed, set_en_time], dim=1)

        # batch_embed = F.normalize(batch_embed, p=2, dim=1)  #不用
        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(batch_embed, batch_embed.T),
            self.temperature)  # 计算两两样本间点乘相似度
        # for numerical stability

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        exp_logits = torch.exp(logits)

        # make mask
        # positive mask 对角线要为0，可对路径长度做限制，减少positive sample number
        # 计算Loss而言，(i,i)位置表示样本本身的相似度，对Loss是没用的，所以要mask掉
        positives_mask = torch.zeros(num_dif, num_dif, dtype=torch.float32).to(self.device)

        for index in range(num_dif):
            similar_en = similar_entities[index]
            if (len(similar_en) == 0):
                continue
            for i, _ in similar_en:
                if index != i:
                    positives_mask[index][i] = 1
                    positives_mask[i][index] = 1
        # negatives_mask = 1 - positives_mask
        # for index in range(num_dif):
        #     negatives_mask[index][index] = 0

        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # mask_pos_pairs = positives_mask.sum(1)
        # mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)

        count = (positives_mask > 0).sum(1)
        count = torch.where(count < 1e-6, 1, count)
        log_probs = (positives_mask * log_prob).sum(1) / count

        # loss
        loss = -log_probs
        if self.scale_by_temperature:
            loss *= self.temperature
        loss = loss.mean()
        return loss

    def batch_his_e_t(self, his):
        entity, time = [], []
        for en, tim in his:
            # entity.append(int(en))
            # time.append(int(tim))
            entity.append(en)
            time.append(tim)
        # return torch.tensor(entity).to(self.device), torch.tensor(time).to(self.device)
        return entity, time

