import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class EventEmbeddingModel(nn.Module):
    def __init__(self, device, input_dim_m, output_dim_sm):
        super(EventEmbeddingModel, self).__init__()
        self.LinearQ = nn.Linear(input_dim_m, output_dim_sm)
        self.device = device
        
    def forward(self, lhs, entities, current_time, history, entities_emb, times_embd):
        # entities为张量
        # entities_his_Nor = F.normalize(entities_his, p=2, dim=1)
        # event_times_nor = F.normalize(event_times, p=2, dim=1)
        his_tim_embedding = [] #batch_size * dimension ,get the batch entities&time history info
        for cur_en, event_his, ent, event_time in zip(lhs, history, entities, current_time):
            if len(event_his) == 0: #没有历史记录的事件
                # his_1 = (entities_emb(ent) + times_embd(event_time)).view(1,-1)
                his_1 = entities_emb(ent).view(1,-1)
            else:
                entity, time = self.batch_his_e_t(event_his) #entity，time为列表
                entity_embed = entities_emb(entity) #张量
                # time_embed = times_embd[time]
                
                # entity = torch.tensor(entity) #转为tensor
                # time = torch.tensor(time)
                # event_time = torch.tensor(event_time).to(self.device)
                time_weights = self.time_decay_weight(event_time - time)  #历史个
                # 当历史事件很多，只考虑前 10 个，
                # if len(time_weights.shape)  >10 :
                #     his_1 = torch.mean(entity_embed[:10]*time_weights[:10], dim=0).view(1,-1) #way 1
                #     his_1 = torch.sum(entity_embed+time_embed*time_weights, im=0, keepdim=True) # w.ay 2
                # his_1 = cur_en + torch.mean(entity_embed*time_weights, dim=0, keepdim=True) # w.ay 2
                his_1 =  torch.matmul(time_weights, entity_embed).view(1,-1)
                # else:
                # his_1 = torch.mean(entity_embed[:10]*time_weights[:10], dim=0).view(1,-1) #way 1
                # his_1 = torch.sum((entity_embed+time_embed)*time_weights, im=0, keepdim=True) # w.ay 2  shape (his_num, embed_dimension)-->(1, embed_dimension) in order to inpput the linear
                
            #(1, embed_dimension)
            # his_tim_embedding.append(his_1)
            his_tim_embedding.append(self.LinearQ(his_1))
            # his_tim_embedding.append(torch.sigmoid(comb_his_tim)) # Can be omited
        his_time_embedding_1 = torch.cat(his_tim_embedding, dim=0)
        return his_time_embedding_1

    # get current evernt history
    def batch_his_e_t(self, his):
        entity, time = [], []
        for en, tim in his:
            # entity.append(int(en))
            # time.append(int(tim))
            entity.append(en)
            time.append(tim)
        return torch.tensor(entity).to(self.device), torch.tensor(time).to(self.device)

    # Define an exponential time decay function
    def time_decay_weight(self, delta_t, decay_rate = torch.tensor(1)):
        decay_rate = decay_rate.to(self.device)
        delta = torch.exp(-decay_rate * delta_t)
        return delta
    
