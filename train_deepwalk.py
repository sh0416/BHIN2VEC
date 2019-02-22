import os
import logging
import argparse

import numpy as np
import scipy.sparse as sp
import pandas as pd
import matplotlib.pyplot as plt
import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter

from data import RandomWalkDataset
from models import SkipGramModel, BalancedSkipGramModel
from utils import get_preprocessed_data


def add_argument(parser):
    parser.add_argument('--root', type=str, default='data')
    parser.add_argument('--dataset', type=str, default='dblp', choices=['blog', 'dblp', 'yelp'])
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--d', type=int, default=128)
    parser.add_argument('--l', type=int, default=100)
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--m', type=int, default=25)
    parser.add_argument('--restore', action='store_true')


def get_output_name(args):
    name = 'deepwalk_%d_%d_%d_%d' % (args.d, args.l, args.k, args.m)
    return name


def deepwalk(v, l, adj_data, adj_size, adj_start):
    """Random walk

    Args:
        v (torch.LongTensor): [Batch_size] start index.
        l (int): length of random walk.
        adj_data (torch.LongTensor): [E] data bank for adjacency list.
        adj_size (torch.FloatTensor): [V] degree for each node.
        adj_start (torch.LongTensor): [V] start index for each node in `adj_data`.
    Returns:
        torch.LongTensor [B, L]
    """
    walk = [None] * l
    node = v
    walk[0] = node

    for i in range(1, l):
        offset = torch.floor(torch.rand_like(node, dtype=torch.float) * adj_size[node]).long()
        idx = adj_start[node] + offset
        node = adj_data[idx]
        walk[i] = node
    return torch.stack(walk).t()


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    #handler = logging.StreamHandler()
    #logger.addHandler(handler)

    node_df, edge_df, _, data = get_preprocessed_data(args)

    adj_data = data['adj_data']
    adj_size = data['adj_size']
    adj_start = data['adj_start']
    type_order = data['type_order']
    degree = data['degree']

    adj_data = torch.tensor(adj_data, dtype=torch.long)
    adj_size = torch.tensor(adj_size, dtype=torch.float)
    adj_start = torch.tensor(adj_start, dtype=torch.long)
    degree = torch.from_numpy(degree)

    type_ = [None] *len(node_df)
    for idx, row in node_df.iterrows():
        type_[row['index']] = type_order.index(row['type'])
    type_ = torch.tensor(type_, dtype=torch.long)

    adj_size = adj_size.sum(dim=1)
    adj_start = adj_start[:, 0]

    model = SkipGramModel(len(node_df), args.d, args.l, args.k, args.m).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    os.makedirs('output', exist_ok=True)

    # TensorboardX
    writer = SummaryWriter(os.path.join('runs', get_output_name(args)))

    start_node = torch.arange(len(node_df))[degree > 0]
    num_iter = start_node.shape[0] // args.batch_size

    n_iter = 0
    for epoch in range(args.epoch):
        with torch.no_grad():
            random_idx = torch.randperm(start_node.shape[0])
            start_node = start_node[random_idx][:num_iter*args.batch_size].view(num_iter, args.batch_size)

        with tqdm.tqdm(range(num_iter), total=num_iter, ascii=True) as t:
            for idx in t:
                with torch.no_grad():
                    walk = deepwalk(start_node[idx], args.l, adj_data, adj_size, adj_start)
                    #negative = torch.multinomial(degree,
                    #                             args.batch_size*args.m*(args.l-args.k),
                    #                             replacement=True).view(args.batch_size,
                    #                                                    args.l-args.k,
                    #                                                    args.m)
                    negative = torch.randint(len(node_df), (args.batch_size, args.l-args.k, args.m))
                    walk = walk.to(device)
                    negative = negative.to(device)

                optimizer.zero_grad()
                pos, neg = model(walk, negative)

                label_pos = torch.ones_like(pos)
                label_neg = torch.zeros_like(neg)

                y = torch.cat((pos, neg), dim=2)
                label = torch.cat((label_pos, label_neg), dim=2)
                # [B, L-K, K+M]

                loss = criterion(y, label)
                loss.backward()
                optimizer.step()

                writer.add_scalar('data/loss', loss, n_iter)
                n_iter += 1
                t.set_postfix(loss=loss.item())

        # Save embedding
        total_embedding = model.node_embedding.weight.data
        for i, t in enumerate(type_order):
            writer.add_embedding(total_embedding[type_==i],
                                 global_step=epoch, tag=t+'_none')
            if 'L' in node_df.columns:
                writer.add_embedding(total_embedding[type_==i],
                                     metadata=node_df[node_df['type']==t]['L'].values,
                                     global_step=epoch, tag=t+'_L')
            if 'L2' in node_df.columns:
                writer.add_embedding(total_embedding[v[0]:v[1]+1, :],
                                     metadata=node_df[node_df['type']==t]['L2'].values,
                                     global_step=epoch, tag=t+'_L2')

        np.save(os.path.join('output', get_output_name(args)+'.npy'),
                model.node_embedding.weight.detach().cpu().numpy())
    writer.close()

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    add_argument(parser)
    args = parser.parse_args()
    main(args)

