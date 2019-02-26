import os
import pickle
import logging
import argparse
import functools
from decimal import Decimal
from collections import defaultdict

import numpy as np
import scipy.sparse as sp
import pandas as pd
import matplotlib.pyplot as plt
import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter
from graphviz import Digraph

from models import BalancedSkipGramModel
from utils import load_data, create_graph


def make_dot(var, params):
    """ Produces Graphviz representation of PyTorch autograd graph

    Blue nodes are the Variables that require grad, orange are Tensors
    saved for backward in torch.autograd.Function

    Args:
        var: output Variable
        params: dict of (name, Variable) to add names to node that
            require grad (TODO: make optional)
    """
    param_map = {id(v): k for k, v in params.items()}
    print(param_map)
    node_attr = dict(style='filled',
                     shape='box',
                     align='left',
                     fontsize='12',
                     ranksep='0.1',
                     height='0.2')
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()
    def size_to_str(size):
        return '('+(', ').join(['%d'% v for v in size])+')'

    def add_nodes(var):
        if var not in seen:
            if torch.is_tensor(var):
                dot.node(str(id(var)), size_to_str(var.size()), fillcolor='orange')
            elif hasattr(var, 'variable'):
                u = var.variable
                node_name = '%s\n %s' % (param_map.get(id(u)), size_to_str(u.size()))
                dot.node(str(id(var)), node_name, fillcolor='lightblue')
            else:
                dot.node(str(id(var)), str(type(var).__name__))
            seen.add(var)
            if hasattr(var, 'next_functions'):
                for u in var.next_functions:
                    if u[0] is not None:
                        dot.edge(str(id(u[0])), str(id(var)))
                        add_nodes(u[0])
            if hasattr(var, 'saved_tensors'):
                for t in var.saved_tensors:
                    dot.edge(str(id(t)), str(id(var)))
                    add_nodes(t)
    add_nodes(var.grad_fn)
    return dot


def add_argument(parser):
    parser.add_argument('--root', type=str, default='data')
    parser.add_argument('--dataset', type=str, default='dblp', choices=['blog', 'douban_movie', 'dblp', 'yelp'])
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--d', type=int, default=128)
    parser.add_argument('--l', type=int, default=100)
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--m', type=int, default=5)
    parser.add_argument('--l_q', type=float, default=0.05)
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--restore', action='store_true')


def get_output_name(args):
    return 'experimental_%s_%d_%d_%d_%d_%d_%E' % (args.dataset, args.d, args.l, args.k, args.m, args.alpha, Decimal(args.l_q))


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


def balancewalk(v, v_t, l, A, t, adj_data, adj_size, adj_start):
    """Random walk

    Args:
        v (torch.LongTensor): [Batch_size] start index.
        v_t (torch.LongTensor): [Batch_size] type of start node.
        l (int): length of random walk.
        A (torch.FloatTensor)): [T, T] inverse training rate matrix.
        t (int): number of type
        adj_data (torch.LongTensor): [E] data bank for adjacency list.
        adj_size (torch.FloatTensor): [V] degree for each node.
        adj_start (torch.LongTensor): [V] start index for each node in `adj_data`.
    Returns:
        torch.LongTensor [L, B]
        torch.LongTensor [L, B]
    """
    walk = [None] * l
    walk_type = [None] * l
    node = v
    node_type = v_t
    walk[0] = node
    walk_type[0] = node_type

    # Initialize data structure
    context_matrix = A.new_zeros(v.shape[0], t, t)
    go_vector = A.new_zeros(v.shape[0], t)
    back_vector = A.new_zeros(v.shape[0], t)

    for i in range(1, l):
        # Process 1
        back_vector.scatter_(dim=1, index=node_type.unsqueeze(1), src=torch.tensor(0))
        node_type_stack = torch.stack([node_type]*len(type_order)).t()

        # Process 2
        back_vector += context_matrix.gather(dim=2, index=node_type_stack.unsqueeze(2)).squeeze(2)
        context_matrix.scatter_(dim=2, index=node_type_stack.unsqueeze(2), src=torch.tensor(0))
        go_vector.scatter_(dim=1, index=node_type.unsqueeze(1), src=torch.tensor(0))

        # Process 3
        src = torch.stack([A]*v.shape[0]).gather(dim=1, index=node_type_stack.unsqueeze(1))
        context_matrix.scatter_add_(dim=1, index=node_type_stack.unsqueeze(1), src=src)
        go_vector += src.squeeze(1)

        # Process 4
        weight = go_vector + back_vector
        weight = weight * (adj_size[node] > 0).float()
        node_type = torch.multinomial(weight, 1).squeeze(1)

        adj_size_type = adj_size[node].gather(dim=1, index=node_type.unsqueeze(1)).squeeze(1)
        assert not torch.any(adj_size_type == 0)

        offset = torch.floor(torch.rand_like(node, dtype=torch.float) * adj_size_type).long()

        adj_start_type = adj_start[node].gather(dim=1, index=node_type.unsqueeze(1)).squeeze(1)
        idx = adj_start_type + offset

        node = adj_data[idx]
        walk[i] = node
        walk_type[i] = node_type
    return torch.stack(walk).t(), torch.stack(walk_type).t()


if __name__=='__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    add_argument(parser)
    args = parser.parse_args()

    node_type, edge_df, test_node_df, _ = load_data(args)

    node_num = max([x[1] for x in node_type.values()]) + 1
    type_num = len(node_type)
    type_order = list(node_type.keys())

    adj_data, adj_size, adj_start = create_graph(edge_df, node_num, type_order)

    adj_data = torch.tensor(adj_data, dtype=torch.long)
    adj_size = torch.tensor(adj_size, dtype=torch.float)
    adj_start = torch.tensor(adj_start, dtype=torch.long)

    type_min = torch.tensor([node_type[k][0] for k in type_order], dtype=torch.float)
    type_size = torch.tensor([node_type[k][1]-node_type[k][0]+1 for k in type_order], dtype=torch.float)

    type_indicator = torch.zeros(node_num, dtype=torch.long)
    for idx, k in enumerate(type_order):
        type_indicator[node_type[k][0]:node_type[k][1]+1] = idx
    type_indicator_gpu = type_indicator.to(device)

    print("Type:", type_order)
    print("Node:", node_num, ", Edge: ", len(edge_df))

    model = BalancedSkipGramModel(node_num, type_num, args.d, args.l, args.k, args.m).to(device)
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    optimizer = optim.SGD(model.parameters(), lr=0.0025)

    # Report result (embedding result, TensorboardX)
    os.makedirs('output', exist_ok=True)
    writer = SummaryWriter(os.path.join('runs', get_output_name(args)))

    start_node = torch.arange(node_num)[adj_size.sum(dim=1)>0]
    num_iter = start_node.shape[0] // args.batch_size

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epoch*num_iter, eta_min=1e-5)

    n_iter = 0
    L0 = 0.6931  # Theoretical initial loss

    relationship_log = []

    A = torch.ones(len(type_order), len(type_order))
    for epoch in range(args.epoch):
        # Shuffle start node
        with torch.no_grad():
            random_idx = torch.randperm(start_node.shape[0])
            start_node = start_node[random_idx][:num_iter*args.batch_size].view(num_iter, args.batch_size)

        training_bar = tqdm.tqdm(range(num_iter), total=num_iter, ascii=True)
        for idx in training_bar:
            scheduler.step()
            with torch.no_grad():
                node = start_node[idx]
                node_type = type_indicator[node]

                walk, walk_type = balancewalk(node, node_type, args.l, A, len(type_order), adj_data, adj_size, adj_start)

                # Move to GPU
                walk, walk_type = walk.to(device), walk_type.to(device)

                # Make positive node
                pos = torch.stack([walk[:, i+1:(i+args.k+1)] for i in range(args.l-args.k)], dim=1)
                pos_type = torch.stack([walk_type[:, i+1:(i+args.k+1)] for i in range(args.l-args.k)], dim=1)
                # [B, L-K, K]

                # Make negative node
                neg = torch.rand((args.batch_size, args.l-args.k, args.k, args.m))
                neg = torch.floor(neg * type_size[pos_type].unsqueeze(3)) + type_min[pos_type].unsqueeze(3)
                neg = neg.long()
                neg_type = type_indicator_gpu[neg]
                assert torch.all(torch.eq(pos_type.unsqueeze(3), neg_type))
                neg = neg.view(neg.shape[0], neg.shape[1], -1)
                neg_type = neg_type.view(neg_type.shape[0], neg_type.shape[1], -1)
                # [B, L-K, K, M]

                # Trim last walk
                walk, walk_type = walk[:, :(args.l-args.k)], walk_type[:, :(args.l-args.k)]
                # [B, L-K]

            optimizer.zero_grad()
            pos_type_pred, neg_type_pred = model(walk, pos, neg, walk_type, pos_type, neg_type)

            neg_type_pred = [x.view(-1, args.m) for x in neg_type_pred]

            pos_type_true = [torch.ones_like(x) for x in pos_type_pred]
            neg_type_true = [torch.zeros_like(x) for x in neg_type_pred]

            type_pred = [torch.cat((pos.unsqueeze(1), neg), dim=1) for pos, neg in zip(pos_type_pred, neg_type_pred)]
            type_true = [torch.cat((pos.unsqueeze(1), neg), dim=1) for pos, neg in zip(pos_type_true, neg_type_true)]

            loss_type_raw = [criterion(pred, true) for pred, true in zip(type_pred, type_true)]
            loss = torch.cat(loss_type_raw).sum()

            #g = make_dot(loss, model.state_dict())
            #g.render()

            #loss_type_raw = [x.sum(dim=1) for x in loss_type_raw]
            #tmp = [torch.mul(torch.exp(x), x) for x in loss_type_raw]
            #for x in tmp:
            #    assert not torch.any(torch.isnan(x))
            #inverse_ratio = torch.cat([x.mean().unsqueeze(0) for x in tmp], dim=0)

            loss_type = torch.cat([x.mean().unsqueeze(0) for x in loss_type_raw], dim=0)
            loss_ratio = loss_type / L0
            inverse_ratio = loss_ratio / loss_ratio.mean()
            # Small -> Well train

            A = inverse_ratio.view(len(type_order), len(type_order)).cpu()
            A = torch.pow(A, args.alpha)

            writer.add_scalar('data/loss', loss, n_iter)
            for i, t1 in enumerate(type_order):
                for j, t2 in enumerate(type_order):
                    writer.add_scalar('data/%s%s'%(t1, t2), loss_type[i*len(type_order)+j], n_iter)
                    writer.add_scalar('ratio/%s%s'%(t1, t2), inverse_ratio[i*len(type_order)+j], n_iter)
            layout = {'Loss': {'loss': ['data/%s%s'%(t1, t2) for t1 in type_order for t2 in type_order]},
                      'ratio': {'ratio': ['ratio/%s%s'%(t1, t2) for t1 in type_order for t2 in type_order]}}
            writer.add_custom_scalars(layout)

            writer.add_image(tag='relationship_embedding',
                             img_tensor=torch.where(model.relationship_embedding.detach().cpu()>=0, torch.tensor(1.), torch.tensor(0.)),
                             global_step=n_iter,
                             dataformats='HW')
            n_iter += 1
            training_bar.set_postfix(epoch=epoch, loss=loss.item(), learning_rate=optimizer.param_groups[0]['lr'])

            loss.backward()
            optimizer.step()

        # Save embedding
        total_embedding = model.node_embedding.data
        writer.add_embedding(total_embedding, global_step=epoch, tag='total')
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
                model.node_embedding.detach().cpu().numpy())
    writer.close()
