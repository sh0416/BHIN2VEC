import os
import pickle
import logging
import argparse
import functools
import numpy as np
import scipy.sparse as sp
import pandas as pd
import matplotlib.pyplot as plt
import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter

from data import RandomWalkDataset
from models import SkipGramModel, BalancedSkipGramModel
from utils import get_preprocessed_data, deepwalk, metapath_walk, TypeChecker, pca, balanced_walk, BalancedWalkFactory, just_walk
from random_walk_statistic import normalize_row, random_walk_2, create_adjacency_matrix


def add_argument(parser):
    parser.add_argument('--root',
                        type=str,
                        default='data')

    parser.add_argument('--dataset',
                        type=str,
                        default='dblp',
                        choices=['dblp', 'yelp'])

    parser.add_argument('--epoch',
                        type=int,
                        default=10)

    parser.add_argument('--batch_size',
                        type=int,
                        default=512)

    parser.add_argument('--d',
                        type=int,
                        default=128)

    parser.add_argument('--l',
                        type=int,
                        default=80)

    parser.add_argument('--k',
                        type=int,
                        default=5)

    parser.add_argument('--m',
                        type=int,
                        default=5)

    parser.add_argument('--restore',
                        action='store_true')


def get_output_name(args):
    name = 'deepwalk_%d_%d_%d_%d' % (args.d, args.l, args.k, args.m)
    return name


def serialize_adj_indice(data):
    adj_data = [list(data['adj_indice'][x]) for x in range(data['node_num'])]
    adj_size = [len(x) for x in adj_data]
    adj_data = [item for sublist in adj_data for item in sublist]
    adj_start = [None] * data['node_num']
    cnt = 0
    for x in range(data['node_num']):
        adj_start[x] = cnt
        cnt += adj_size[x]
    return adj_data, adj_size, adj_start


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

    data = get_preprocessed_data(args, type_split=False)
    adj_data, adj_size, adj_start = serialize_adj_indice(data)
    adj_data = torch.tensor(adj_data, dtype=torch.long, device=device)
    adj_size = torch.tensor(adj_size, dtype=torch.float, device=device)
    adj_start = torch.tensor(adj_start, dtype=torch.long, device=device)

    node_num = data['node_num']
    degree_dist = data['degree']
    class_dict = data['class']
    type_interval = data['type_interval']

    logger.info("Total type number: " + str(node_num))
    logger.info("Total node number: ")
    for k, v in data['type_interval'].items():
        logger.info('\t\t' + k + ': ' + str(v[1]-v[0]+1))
    logger.info("Degree distribution:" + str(degree_dist))

    model = SkipGramModel(node_num, args.d, args.l, args.k, args.m).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-2)

    if args.restore:
        embedding = np.load(os.path.join('output', get_output_name(args)+'.npy'))
        model.node_embedding.weight.data.copy_(torch.from_numpy(embedding))

    os.makedirs('output', exist_ok=True)

    # TensorboardX
    writer = SummaryWriter(os.path.join('runs', get_output_name(args)))

    n_iter = 0
    num_iter = data['node_num'] // args.batch_size
    degree_dist = torch.from_numpy(degree_dist).to(device)
    for epoch in range(args.epoch):
        start_node = torch.randperm(data['node_num'], device=device)[:num_iter*args.batch_size].view(num_iter, args.batch_size)

        for idx in tqdm.tqdm(range(num_iter), ascii=True):
            with torch.no_grad():
                walk = deepwalk(start_node[idx], args.l, adj_data, adj_size, adj_start)
                negative = torch.multinomial(degree_dist,
                                             args.batch_size*args.m*(args.l-args.k),
                                             replacement=True).view(args.batch_size,
                                                                    args.l-args.k,
                                                                    args.m)

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

        # Save embedding
        total_embedding = model.node_embedding.weight.data
        for k, v in type_interval.items():
            writer.add_embedding(total_embedding[v[0]:v[1]+1, :],
                                 metadata=class_dict.get(k),
                                 global_step=epoch,
                                 tag=k)

    np.save(os.path.join('output', get_output_name(args)+'.npy'),
            model.node_embedding.weight.detach().cpu().numpy())
    writer.close()

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    add_argument(parser)
    args = parser.parse_args()
    main(args)

