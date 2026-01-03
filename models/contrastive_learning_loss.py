import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class ConLoss_Path(nn.Module):

    def __init__(self, args, temperature=0.5, scale_by_temperature=True):
        super(ConLoss_Path, self).__init__()
        self.device = args.cuda
        self.temperature = temperature
        self.scale_by_temperature = scale_by_temperature

    def forward(self, entities, entity_embedding, neis_all):
        # 在entity_embedding上，以entitys和neis_all构建mask
        features = entity_embedding
        set_entities = list(set(entities.tolist()))
        set_batch_size = len(set_entities)

        neis_entityies = neis_all[entities]
        # data = [neis_all[i] for i in entities]
        flat_entities = list(set(neis_entityies.view(-1).tolist()))
        # for nei in neis_all[entities]:
        # flat_data = [item for sublist in data for item in sublist]
        set_entities_and_path = copy.deepcopy(set_entities) #kepp order with set_entities
        for en in flat_entities: #去重，且不改变set_entities对应的顺序
            if en not in set_entities_and_path:
                set_entities_and_path.append(en)
        num_dif = len(set_entities_and_path) # 不重复实体个数

        # 构建batch+neis特征矩阵
        features_dim = entity_embedding.shape[1]
        batch_embed = torch.zeros(num_dif, features_dim).to(self.device)
        for index, i in zip(set_entities_and_path, range(num_dif)):
            batch_embed[i] = features[index]


        batch_embed = F.normalize(batch_embed, p=2, dim=1)
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

        for index in range(set_batch_size):
            rel_ind = set_entities[index] #真实实体编号
            neis = neis_all[rel_ind].tolist() #对应路径，真实实体编号，应该只计算entities对应的path,不含path对应node，所以前面用set_batch_size，set_entities[index
            for nei in neis:
                ne = set_entities_and_path.index(nei) #转换为set_entities_and_path对应索引
                if index != ne:
                    positives_mask[index][ne] = 1
                    positives_mask[ne][index] = 1

        negatives_mask = 1 - positives_mask
        for index in range(num_dif):
            negatives_mask[index][index] = 0

        num_positives_per_row = torch.sum(positives_mask, axis=1)  # 除了自己之外，正样本的个数：（s,p,o,t）=>邻居信息、P、O
        denominator = torch.sum(
            exp_logits * negatives_mask, axis=1, keepdims=True).to(self.device) + torch.sum(
            exp_logits * positives_mask, axis=1, keepdims=True).to(self.device)

        log_probs = logits - torch.log(denominator)
        if torch.any(torch.isnan(log_probs)):
            raise ValueError("Log_prob has nan!")

        log_probs = torch.sum(log_probs * positives_mask, axis=1)[num_positives_per_row > 0] / num_positives_per_row[
            num_positives_per_row > 0]

        # loss
        loss = -log_probs
        if self.scale_by_temperature:
            loss *= self.temperature
        loss = loss.mean()
        return loss

