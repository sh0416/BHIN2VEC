import os
import numpy as np
import scipy.sparse as sp
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


author_df = pd.read_csv(os.path.join('data', 'dblp', 'author3.txt'), sep='\t', header=None)
paper_df = pd.read_csv(os.path.join('data', 'dblp', 'paper3.txt'), sep='\t', header=None)
topic_df = pd.read_csv(os.path.join('data', 'dblp', 'topic3.txt'), sep='\t', header=None)
venue_df = pd.read_csv(os.path.join('data', 'dblp', 'venue3.txt'), sep='\t', header=None)

write_df = pd.read_csv(os.path.join('data', 'dblp', 'write3.txt'), sep='\t', header=None)
publish_df = pd.read_csv(os.path.join('data', 'dblp', 'publish3.txt'), sep='\t', header=None)
mention_df = pd.read_csv(os.path.join('data', 'dblp', 'mention3.txt'), sep='\t', header=None)
cite_df = pd.read_csv(os.path.join('data', 'dblp', 'cite3.txt'), sep='\t', header=None)

type_num = 4
node_num = len(author_df) + len(paper_df) + len(topic_df) + len(venue_df)
edge_num = len(write_df) + len(publish_df) + len(mention_df) + len(cite_df)

print("Total type number", type_num)
print("Total node number", node_num)
print("Total edge number", edge_num)

dim = 32

class Model(nn.Module):
    def __init__(self, node_num, type_num, dim):
        super().__init__()
        self.node_embedding = nn.Embedding(node_num, dim)
        self.type_embedding = nn.Embedding(type_num, dim)

    def forward(self):
        return torch.mm(self.node_embedding.weight, self.type_embedding.weight.t())

model = Model(node_num, type_num, dim).cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)

type_true = torch.zeros(node_num, dtype=torch.long).cuda()
start = 0
for label, interval in enumerate([len(author_df), len(paper_df), len(topic_df), len(venue_df)]):
    type_true[start:start+interval] = label
    start = start + interval

for idx in range(100000):
    optimizer.zero_grad()
    pred = model()
    assert not torch.any(torch.isnan(pred))
    loss = criterion(pred, type_true)
    loss.backward()
    optimizer.step()
    if idx % 100 == 99:
        print('loss: %.4f' % loss)
    if idx % 1000 == 999:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(pred[:10, :].detach().cpu().numpy(), cmap=plt.get_cmap('Greys'))
        fig.colorbar(cax)
        plt.show()

