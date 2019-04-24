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
        # [B, L-K, D, K*M]

        walk = walk.view(-1, 1, self.dim)
        # [B*(L-K), 1, D]
        positive = positive.view(-1, self.dim, self.k)
        # [B*(L-K), D, K]
        negative = negative.view(-1, self.dim, self.k*self.m)
        # [B*(L-K), D, K*M]

        pos = torch.bmm(walk, positive)
        # [B*(L-K), 1, K]
        neg = torch.bmm(walk, negative)
        # [B*(L-K), 1, K*M]

        pos = pos.view(-1, self.l-self.k, self.k)
        # [B, L-K, K]
        neg = neg.view(-1, self.l-self.k, self.k*self.m)
        # [B, L-K, K*M]

        return pos, neg


class BalancedSkipGramModel(nn.Module):
    def __init__(self, node_num, type_num, dim, l, k, m):
        super().__init__()
        # Parameter
        self.node_embedding = nn.Parameter(torch.empty(node_num, dim))
        self.relationship_embedding = nn.Parameter(torch.empty(type_num*type_num, dim))
        nn.init.normal_(self.node_embedding.data, std=0.1)
        nn.init.normal_(self.relationship_embedding.data, std=1)

        # Data information
        self.node_num = node_num
        self.type_num = type_num

        # Hyperparameter
        self.dim = dim
        self.l = l
        self.k = k
        self.m = m

    def forward(self, walk, pos, neg, walk_type, pos_type, neg_type):
        """Forward process.

        Args:
            walk (torch.LongTensor): random walk index. shape: [B, L-K]
            pos (torch.LongTensor): positive sample. shape: [B, L-K, K]
            neg (torch.LongTensor): negative sample. shape: [B, L-K, K, M]
            walk_type (torch.LongTensor): type of random walk index. shape: [B, L-K]
            pos_type (torch.LongTensor): type of positive sample. shape: [B, L-K, K]
            neg_type (torch.LongTensor): type of negative sample. shape: [B, L-K, K, M]
        """
        pos_pair_type = self.type_num*walk_type.unsqueeze(2)+pos_type
        neg_pair_type = self.type_num*walk_type.unsqueeze(2).unsqueeze(3)+neg_type
        # [B, L-K, K], [B, L-K, K, M]

        pos_pair = self.relationship_embedding[pos_pair_type]
        neg_pair = self.relationship_embedding[neg_pair_type]
        # [B, L-K, K, D], [B, L-K, K, M, D]

        pos_pair_type = self.type_num*self.type_num*torch.arange(self.k).cuda().unsqueeze(0).unsqueeze(1) + pos_pair_type
        neg_pair_type = self.type_num*self.type_num*torch.arange(self.k).cuda().unsqueeze(0).unsqueeze(1).unsqueeze(3) + neg_pair_type
        # [B, L-K, K], [B, L-K, K, M]

        neg = neg.view(-1, self.l-self.k, self.k*self.m)
        neg_type = neg_type.view(-1, self.l-self.k, self.k*self.m)
        neg_pair = neg_pair.view(-1, self.l-self.k, self.k*self.m, self.dim)
        neg_pair_type = neg_pair_type.view(-1, self.l-self.k, self.k*self.m)

        walk = self.node_embedding[walk, :]
        pos = self.node_embedding[pos, :]
        neg = self.node_embedding[neg, :]
        # [B, L-K, D], [B, L-K, K, D], [B, L-K, K*M, D]

        pos = torch.mul(pos, torch.sigmoid(pos_pair)).transpose(2, 3)
        neg = torch.mul(neg, torch.sigmoid(neg_pair)).transpose(2, 3)
        # [B, L-K, D, K], [B, L-K, D, K*M]

        walk = walk.view(-1, 1, self.dim)
        # [B*(L-K), 1, D]
        pos = pos.view(-1, self.dim, self.k)
        # [B*(L-K), D, K]
        neg = neg.view(-1, self.dim, self.k*self.m)
        # [B*(L-K), D, K*M]

        pos = torch.bmm(walk, pos)
        # [B*(L-K), 1, K]
        neg = torch.bmm(walk, neg)
        # [B*(L-K), 1, K*M]

        pos = pos.view(-1, self.l-self.k, self.k)
        # [B, L-K, K]
        neg = neg.view(-1, self.l-self.k, self.k*self.m)
        # [B, L-K, K*M]

        pos = pos.view(-1)
        neg = neg.view(-1)
        pos_pair_type = pos_pair_type.view(-1)
        neg_pair_type = neg_pair_type.view(-1)

        return pos, neg, pos_pair_type, neg_pair_type
