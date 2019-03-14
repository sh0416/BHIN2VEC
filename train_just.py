import os
import pickle
import logging
import argparse
import functools
from collections import deque
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
from utils import load_data, create_graph


def add_argument(parser):
    parser.add_argument('--root', type=str, default='data')
    parser.add_argument('--dataset', type=str, default='dblp', choices=['douban_movie', 'aminer', 'dblp', 'yelp'])
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--que_size', type=int, default=2)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--d', type=int, default=128)
    parser.add_argument('--l', type=int, default=100)
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--m', type=int, default=5)
    parser.add_argument('--restore', action='store_true')


def get_output_name(args):
    name = 'just_%s_%d_%d_%d_%d' % (args.dataset, args.d, args.l, args.k, args.m)
    name += '_%.2f_%d' % (args.alpha, args.que_size)
    return name

def get_type(v, type_interval_dict, type_order):
    for k, interval in type_interval_dict.items():
        if interval[0]<= v and v<=interval[1]:
            return type_order.index(k)
    raise Exception

def justwalk(v, v_t, l, alpha, que_size, adj_data, adj_size, adj_start):
    same_domain_count = torch.zeros(v.shape[0])
    que = torch.ones(v.shape[0], que_size, dtype=torch.long)
    que = torch.mul(que, v_t.unsqueeze(1))
    que_idx = torch.zeros(v.shape[0], dtype=torch.long)

    walk = torch.empty(v.shape[0], l, dtype=torch.long)
    walk[:, 0] = v

    for idx in range(1, l):
        same_domain_count += 1
        type_all = adj_size[walk[:, idx-1]]
        type_same = type_all.gather(dim=1, index=v_t.unsqueeze(1)).squeeze(1)
        type_sum = type_all.sum(dim=1)

        stay = torch.full((v.shape[0],), fill_value=-1)
        stay = torch.where(type_same==0, torch.tensor(0.), stay)
        stay = torch.where(type_sum==type_same, torch.tensor(1.), stay)
        stay = torch.where(stay==-1, torch.pow(alpha, same_domain_count), stay)
        stay = torch.bernoulli(stay)

        type_stay = torch.zeros_like(type_all, dtype=torch.float)
        type_stay.scatter_(dim=1, index=v_t.unsqueeze(1), src=torch.tensor(1.))

        type_jump = type_all.clone().detach().float()
        type_jump[type_jump!=0] = 1
        type_jump.scatter_(dim=1, index=que, src=torch.tensor(0.))

        type_jump2 = type_all.clone().detach().float()
        type_jump2[type_jump2!=0] = 1

        type_jump_cond = (type_jump.sum(1)==0).float().unsqueeze(1)
        type_jump = torch.mul(type_jump_cond, type_jump2) + torch.mul(1-type_jump_cond, type_jump)

        type_prob = torch.mul(stay.unsqueeze(1), type_stay) + torch.mul((1-stay).unsqueeze(1), type_jump)

        v_t = torch.multinomial(type_prob, num_samples=1).squeeze(1)

        offset = torch.rand_like(v_t, dtype=torch.float)
        offset = torch.floor(offset * type_all.gather(dim=1, index=v_t.unsqueeze(1)).squeeze(1).float()).long()
        start = adj_start[walk[:, idx-1]].gather(dim=1, index=v_t.unsqueeze(1)).squeeze(1)
        walk[:, idx] = adj_data[start + offset]

    return walk


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

def main(args):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    #handler = logging.StreamHandler()
    #logger.addHandler(handler)

    node_type, edge_df, test_node_df, _ = load_data(args)

    node_num = max([x[1] for x in node_type.values()]) + 1
    type_num = len(node_type)
    type_order = list(node_type.keys())

    adj_data, adj_size, adj_start = create_graph(edge_df, node_num, type_order)

    adj_data = torch.tensor(adj_data, dtype=torch.long)
    adj_size = torch.tensor(adj_size, dtype=torch.long)
    adj_start = torch.tensor(adj_start, dtype=torch.long)

    print("Type:", type_order)
    print("Node:", node_num, ", Edge: ", len(edge_df))

    model = SkipGramModel(node_num, args.d, args.l, args.k, args.m).cuda()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0025)

    if args.restore:
        embedding = np.load(os.path.join('output', get_output_name(args)+'.npy'))
        model.node_embedding.weight.data.copy_(torch.from_numpy(embedding))

    os.makedirs('output', exist_ok=True)

    # TensorboardX
    writer = SummaryWriter(os.path.join('runs', get_output_name(args)))

    n_iter = 0
    num_iter = node_num // args.batch_size
    for epoch in range(args.epoch):
        with torch.no_grad():
            start_node = torch.randperm(node_num)[:num_iter*args.batch_size].view(num_iter, args.batch_size)

        with tqdm.tqdm(range(num_iter), total=num_iter, ascii=True) as t:
            for idx in t:
                with torch.no_grad():
                    node = start_node[idx]
                    node_t = torch.empty_like(node)
                    for k, v in node_type.items():
                        node_t = torch.where((v[0]<=node) & (node<=v[1]), torch.tensor(type_order.index(k)), node_t)

                    walk = justwalk(node, node_t, args.l, args.alpha, args.que_size, adj_data, adj_size, adj_start)
                    negative = torch.randint(0, node_num, size=(args.batch_size, args.l-args.k, args.k*args.m))

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
            if t in test_node_df:
                tmp = test_node_df[t].drop(t, axis=1)
                if len(tmp.columns) == 1:
                    writer.add_embedding(total_embedding[test_node_df[t][t].values],
                                         metadata=tmp.values[:, 0],
                                         global_step=epoch, tag=t)
                else:
                    writer.add_embedding(total_embedding[test_node_df[t][t].values],
                                         metadata_header=list(tmp.columns),
                                         metadata=tmp.values,
                                         global_step=epoch, tag=t)

        np.save(os.path.join('output', get_output_name(args)+'.npy'),
                model.node_embedding.weight.detach().cpu().numpy())
    writer.close()

if __name__=='__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser()
    add_argument(parser)
    args = parser.parse_args()
    main(args)

