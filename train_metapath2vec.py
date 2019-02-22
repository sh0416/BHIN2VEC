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

from models import SkipGramModel
from utils import get_preprocessed_data


def add_argument(parser):
    parser.add_argument('--root',
                        type=str,
                        default='data')

    parser.add_argument('--dataset',
                        type=str,
                        default='dblp',
                        choices=['blog', 'dblp', 'yelp'])

    parser.add_argument('--metapath',
                        type=str,
                        default='APTPVPTP')

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
    name = 'metapath2vec_%d_%d_%d_%d_%s' % (args.d, args.l, args.k, args.m, args.metapath)
    return name

def get_type(v, type_interval_dict, type_order):
    for k, interval in type_interval_dict.items():
        if interval[0]<= v and v<=interval[1]:
            return type_order.index(k)
    raise Exception

def serialize_adj_indice(data):
    type_order = list(data['type_interval'].keys())
    adj_data = [data['adj_indice'][x] for x in range(data['node_num'])]
    adj_data = [[list(x[y]) for y in type_order] for x in adj_data]
    adj_size = [[len(y) for y in x] for x in adj_data]
    adj_start = [[None]*len(type_order) for _ in range(data['node_num'])]
    count = 0
    for i in range(data['node_num']):
        for j in range(len(type_order)):
            adj_start[i][j] = count
            count += adj_size[i][j]
    adj_data = [item for sublist in adj_data for subsublist in sublist for item in subsublist]
    return adj_data, adj_size, adj_start, type_order

def metapathwalk(v, v_t, l, metapath, adj_data, adj_size, adj_start):
    """Random walk

    Args:
        v (torch.LongTensor): [Batch_size] start index.
        v_t (torch.LongTensor): [Batch_size] type of start node.
        l (int): length of random walk.
        metapath (torch.LongTensor)): metapath
        adj_data (torch.LongTensor): [E] data bank for adjacency list.
        adj_size (torch.FloatTensor): [V] degree for each node.
        adj_start (torch.LongTensor): [V] start index for each node in `adj_data`.
    Returns:
        torch.LongTensor [B, L]
    """
    walk = [None] * l
    node = v
    metapath_idx = torch.tensor([(metapath==x).nonzero()[0] for x in v_t], dtype=torch.long)
    walk[0] = node

    for i in range(1, l):
        metapath_idx = (metapath_idx + 1) % metapath.shape[0]
        adj_size_type = adj_size[node].gather(dim=1, index=metapath[metapath_idx].unsqueeze(1)).squeeze(1)
        assert not torch.any(adj_size_type == 0)

        offset = torch.floor(torch.rand_like(node, dtype=torch.float) * adj_size_type).long()

        adj_start_type = adj_start[node].gather(dim=1, index=metapath[metapath_idx].unsqueeze(1)).squeeze(1)
        idx = adj_start_type + offset

        node = adj_data[idx]
        walk[i] = node
    return torch.stack(walk).t()


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    #handler = logging.StreamHandler()
    #logger.addHandler(handler)

    data = get_preprocessed_data(args, type_split=True)
    adj_data, adj_size, adj_start, type_order = serialize_adj_indice(data)
    adj_data = torch.tensor(adj_data, dtype=torch.long)#.to(device)
    adj_size = torch.tensor(adj_size, dtype=torch.float)#.to(device)
    adj_start = torch.tensor(adj_start, dtype=torch.long)#.to(device)

    metapath = torch.tensor([type_order.index(x) for x in args.metapath], dtype=torch.long)#.to(device)

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
    degree_dist = torch.from_numpy(degree_dist)#.to(device)
    for epoch in range(args.epoch):
        with torch.no_grad():
            start_node = torch.randperm(data['node_num'])[:num_iter*args.batch_size].view(num_iter, args.batch_size)#.to(device)

        with tqdm.tqdm(range(num_iter), total=num_iter, ascii=True) as t:
            for idx in t:
                with torch.no_grad():
                    node = start_node[idx]
                    node_type = [get_type(x, data['type_interval'], type_order) for x in node]

                    walk = metapathwalk(node, node_type, args.l, metapath, adj_data, adj_size, adj_start)
                    negative = torch.multinomial(degree_dist,
                                                 args.batch_size*args.m*(args.l-args.k),
                                                 replacement=True).view(args.batch_size,
                                                                        args.l-args.k,
                                                                        args.m)
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
        for k, v in type_interval.items():
            writer.add_embedding(total_embedding[v[0]:v[1]+1, :],
                                 metadata=class_dict.get(k),
                                 global_step=epoch, tag=k)
            if (k+'2') in class_dict.keys():
                writer.add_embedding(total_embedding[v[0]:v[1]+1, :],
                                     metadata=class_dict.get(k+'2'),
                                     global_step=epoch, tag=k+'2')

        np.save(os.path.join('output', get_output_name(args)+'.npy'),
                model.node_embedding.weight.detach().cpu().numpy())
    writer.close()

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    add_argument(parser)
    args = parser.parse_args()
    main(args)

