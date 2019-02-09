import torch
import torch.nn as nn
import torch.nn.functional as F


class SkipGramModel(nn.Module):
    def __init__(self, node_num, dim, l, k, m):
        super().__init__()
        self.node_embedding = nn.Embedding(node_num, dim)
        nn.init.normal_(self.node_embedding.weight.data, std=0.1)
        self.dim = dim
        self.l = l
        self.k = k
        self.m = m

    def forward(self, walk, negative):
        # [B, L]
        positive = torch.cat([walk[:, i+1:(i+self.k+1)] for i in range(self.l-self.k)], dim=1)
        positive = positive.view(-1, self.l-self.k, self.k)
        # [B, L-K, K]

        walk = self.node_embedding(walk[:, :(self.l-self.k)])
        # [B, L-K, D]
        positive = self.node_embedding(positive).transpose(2, 3)
        # [B, L-K, D, K]
        negative = self.node_embedding(negative).transpose(2, 3)
        # [B, L-K, D, M]

        walk = walk.view(-1, 1, self.dim)
        # [B*(L-K), 1, D]
        positive = positive.view(-1, self.dim, self.k)
        # [B*(L-K), D, K]
        negative = negative.view(-1, self.dim, self.m)
        # [B*(L-K), D, M]

        pos = torch.bmm(walk, positive)
        # [B*(L-K), 1, K]
        neg = torch.bmm(walk, negative)
        # [B*(L-K), 1, M]

        pos = pos.view(-1, self.l-self.k, self.k)
        # [B, L-K, K]
        neg = neg.view(-1, self.l-self.k, self.m)
        # [B, L-K, M]
        
        return pos, neg


def get_type(tensor, type_interval_dict, type_str_dict):
    with torch.no_grad():
        result = torch.zeros_like(tensor)
        for k, v in type_interval_dict.items():
            result += torch.where(v[0]<=tensor, 
                                  torch.where(tensor<=v[1], 
                                              torch.full_like(tensor, fill_value=type_str_dict[k]), 
                                              torch.zeros_like(tensor)), 
                                  torch.zeros_like(tensor))
    return result


def apply_tensor(tensor, subtensor):
    with torch.no_grad():
        result = torch.zeros_like(tensor)
        for i in range(subtensor.shape[0]):
            result += torch.where(tensor==i, subtensor[i], result.new_zeros(1))
    return result


class BalancedSkipGramModel(nn.Module):
    def __init__(self,
                 node_num,
                 dim,
                 l,
                 k,
                 m,
                 type_interval_dict,
                 type_str_dict,
                 type_str_inverse_dict,
                 criterion):
        super().__init__()
        # Parameter
        self.node_embedding = nn.Embedding(node_num, dim)
        nn.init.normal_(self.node_embedding.weight.data, std=0.001)

        # Hyperparameter
        self.dim = dim
        self.l = l
        self.k = k
        self.m = m
        self.type_str_dict = type_str_dict
        self.type_str_inverse_dict = type_str_inverse_dict
        self.type_interval_dict = type_interval_dict
        self.criterion = criterion

    def forward(self, walk, negative):
        # [B, L]
        positive = torch.cat([walk[:, i+1:(i+self.k+1)] for i in range(self.l-self.k)], dim=1)
        positive = positive.view(-1, self.l-self.k, self.k)
        # [B, L-K, K]

        walk = walk[:, :(self.l-self.k)]
        walk_type = get_type(walk, self.type_interval_dict, self.type_str_dict)
        # [B, L-K]
        positive_type = get_type(positive, self.type_interval_dict, self.type_str_dict)
        # [B, L-K, K]
        negative_type = get_type(negative, self.type_interval_dict, self.type_str_dict)
        # [B, L-K, M]

        with torch.no_grad():
            positive_interaction_type = torch.add(4*walk_type.unsqueeze(2), positive_type)
            negative_interaction_type = torch.add(4*walk_type.unsqueeze(2), negative_type)

        walk = self.node_embedding(walk)
        # [B, L-K, D]
        positive = self.node_embedding(positive).transpose(2, 3)
        # [B, L-K, D, K]
        negative = self.node_embedding(negative).transpose(2, 3)
        # [B, L-K, D, M]

        walk = walk.view(-1, 1, self.dim)
        # [B*(L-K), 1, D]
        positive = positive.view(-1, self.dim, self.k)
        # [B*(L-K), D, K]
        negative = negative.view(-1, self.dim, self.m)
        # [B*(L-K), D, M]

        pos = torch.bmm(walk, positive)
        # [B*(L-K), 1, K]
        neg = torch.bmm(walk, negative)
        # [B*(L-K), 1, M]

        pos = pos.view(-1, self.l-self.k, self.k)
        # [B, L-K, K]
        neg = neg.view(-1, self.l-self.k, self.m)
        # [B, L-K, M]

        positive_per_type = [torch.masked_select(pos, positive_interaction_type==i)
                             for i in range(len(self.type_str_dict)*len(self.type_str_dict))]
        negative_per_type = [torch.masked_select(neg, negative_interaction_type==i)
                             for i in range(len(self.type_str_dict)*len(self.type_str_dict))]
        # [T*T]

        label_positive_per_type = [torch.ones_like(x) for x in positive_per_type]
        label_negative_per_type = [torch.zeros_like(x) for x in negative_per_type]

        y_per_type = [torch.cat((pos, neg), dim=0) for pos, neg in zip(positive_per_type, negative_per_type)]
        label_per_type = [torch.cat((pos, neg), dim=0) for pos, neg in zip(label_positive_per_type, label_negative_per_type)]
        loss_per_type = [self.criterion(pred, true) for pred, true in zip(y_per_type, label_per_type)]
        loss = torch.cat(loss_per_type, dim=0).mean()
        loss_per_type = torch.cat([x.mean().unsqueeze(0) for x in loss_per_type], dim=0)

        return loss, loss_per_type

