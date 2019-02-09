import numpy as np

import torch
from torch.utils.data import Dataset


class RandomWalkDataset(Dataset):
    def __init__(self, node_num, walk_fun, neg_dist, l, k, m):
        self.node_num = node_num
        self.walk_fun = walk_fun
        self.neg_dist = neg_dist
        self.l = l
        self.k = k
        self.m = m

    def __getitem__(self, idx):
        walk = self.walk_fun(idx, l=self.l) 
        negative = np.random.choice(self.node_num, 
                                    size=self.m*(self.l-self.k), 
                                    p=self.neg_dist)
        walk = torch.tensor(walk, dtype=torch.long)
        negative = torch.tensor(negative, dtype=torch.long)
        negative = negative.view(self.l-self.k, self.m)
        return walk, negative

    def __len__(self):
        return self.node_num
