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
from utils import load_data, create_graph, make_dot


def add_argument(parser):
    parser.add_argument('--root', type=str, default='data')
    parser.add_argument('--dataset', type=str, default='dblp', choices=['blog-catalog', 'douban_movie', 'dblp', 'yago'])
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--d', type=int, default=128)
    parser.add_argument('--l', type=int, default=100)
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--m', type=int, default=5)
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--lr', type=float, default=0.0025)
    parser.add_argument('--lr2', type=float, default=0.0025)
    parser.add_argument('--restore', action='store_true')
    parser.add_argument('--approximate_naive', action='store_true')


def get_output_name(args):
    return 'experimental_%s_%d_%d_%d_%d_%.2f_%.6f_%6f' % (args.dataset, args.d, args.l, args.k, args.m, args.alpha, args.lr, args.lr2)


def biased_walk(node, node_t, l, stochastic_matrix, adj_data, adj_size, adj_start):
    walk, walk_t = [None] * l, [None] * l
    walk[0], walk_t[0] = node, node_t

    for i in range(1, l):
        # Previous node and node type
        node = walk[i-1]
        node_t = walk_t[i-1]

        # Sample current node type
        weight = stochastic_matrix[node_t, :]
        weight = weight * (adj_size[node]>0).float()
        node_nxt_t = torch.multinomial(weight, 1)

        # Sample current node
        start_t = adj_start[node].gather(dim=1, index=node_nxt_t)
        size_t = adj_size[node].gather(dim=1, index=node_nxt_t)
        assert not torch.any(size_t == 0)

        offset_t = torch.floor(torch.rand_like(size_t) * size_t).long()
        idx = start_t + offset_t

        node_nxt = adj_data[idx].squeeze(1)
        node_nxt_t = node_nxt_t.squeeze(1)

        walk[i] = node_nxt
        walk_t[i] = node_nxt_t
    return torch.stack(walk).t(), torch.stack(walk_t).t()


def main(args):
    torch.set_printoptions(precision=5)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    node_type, edge_df, test_node_df, _ = load_data(args)

    # Node metadata
    node_num = max([x[1] for x in node_type.values()]) + 1

    # Type metadata
    type_num = len(node_type)
    type_order = list(node_type.keys())
    type_min = torch.tensor([node_type[k][0] for k in type_order], dtype=torch.float)
    type_size = torch.tensor([node_type[k][1]-node_type[k][0]+1 for k in type_order], dtype=torch.float)
    type_indicator = torch.zeros(node_num, dtype=torch.long)
    for idx, k in enumerate(type_order):
        type_indicator[node_type[k][0]:node_type[k][1]+1] = idx
    type_indicator_gpu = type_indicator.to(device)

    # Create graph
    adj_data, adj_size, adj_start = create_graph(edge_df, node_num, type_order)
    adj_data = torch.tensor(adj_data, dtype=torch.long)
    adj_size = torch.tensor(adj_size, dtype=torch.float)
    adj_start = torch.tensor(adj_start, dtype=torch.long)
    assert torch.all(adj_size.sum(dim=1) > 0)

    print("Type:", type_order)
    print("Node:", node_num, ", Edge: ", len(edge_df))

    node_idx = torch.arange(node_num)
    num_iter = node_idx.shape[0] // args.batch_size

    model = BalancedSkipGramModel(node_num, type_num, args.d, args.l, args.k, args.m).to(device)
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Report result (embedding result, TensorboardX)
    os.makedirs('output', exist_ok=True)
    writer = SummaryWriter(os.path.join('runs', get_output_name(args)))

    # Theoretical initial loss
    L0 = 0.6931

    # Meta adjacency matrix
    meta_adjacency_matrix = torch.zeros(len(type_order), len(type_order), dtype=torch.float).cuda()
    edge_type = edge_df[['t1', 't2']].apply(lambda x: (x['t1'], x['t2']), axis=1).unique()
    for t1, t2 in edge_type:
        meta_adjacency_matrix[type_order.index(t1), type_order.index(t2)] = 1
        meta_adjacency_matrix[type_order.index(t2), type_order.index(t1)] = 1

    # Possible tensor
    type_normal_tensor = torch.zeros(args.k, len(type_order), len(type_order)).cuda()
    tmp = meta_adjacency_matrix / meta_adjacency_matrix.sum(dim=1, keepdim=True)
    for i in range(args.k):
        type_normal_tensor[i] = tmp
        tmp = torch.mm(tmp, meta_adjacency_matrix / meta_adjacency_matrix.sum(dim=1, keepdim=True))
    possible_tensor = type_normal_tensor > 0

    # Stochastic matrix
    stochastic_matrix = meta_adjacency_matrix
    stochastic_matrix = stochastic_matrix / stochastic_matrix.sum(dim=1, keepdim=True)
    stochastic_matrix = stochastic_matrix.clone().detach().requires_grad_(True)

    stochastic_matrix_normal = stochastic_matrix.clone().detach().requires_grad_(False)

    # Loss
    previous_case_loss = torch.full((args.k, len(type_order), len(type_order)), fill_value=0.6931)

    loss_stochastic_f = open('loss_stochastic.log', 'w')
    inverse_ratio_f = open('inverse_ratio.log', 'w')
    stochastic_matrix_f = open('stochastic_matrix.log', 'w')
    # Counter
    n_iter = 0
    for epoch in range(args.epoch):
        # Shuffle start node
        with torch.no_grad():
            random_idx = torch.randperm(node_idx.shape[0])
            node_idx_epoch = node_idx[random_idx][:num_iter*args.batch_size].view(num_iter, args.batch_size)

        training_bar = tqdm.tqdm(range(num_iter), total=num_iter, ascii=True)
        for idx in training_bar:
            with torch.no_grad():
                node = node_idx_epoch[idx]
                node_t = type_indicator[node]

                walk, walk_type = biased_walk(node, node_t, args.l, stochastic_matrix.cpu(), adj_data, adj_size, adj_start)

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
                # [B, L-K, K, M]

                # Trim last walk
                walk = walk[:, :(args.l-args.k)]
                walk_type = walk_type[:, :(args.l-args.k)]
                # [B, L-K]

            optimizer.zero_grad()
            pos_pred, neg_pred, pos_type, neg_type = model(walk, pos, neg, walk_type, pos_type, neg_type)

            pos_true = torch.ones_like(pos_pred)
            neg_true = torch.zeros_like(neg_pred)

            pred = torch.cat((pos_pred, neg_pred))
            true = torch.cat((pos_true, neg_true))
            type_ = torch.cat((pos_type, neg_type))

            loss = criterion(pred, true)
            #g = make_dot(loss, model.state_dict())
            #g.render()

            loss.mean().backward()
            optimizer.step()

            n_iter += 1
            training_bar.set_postfix(epoch=epoch,
                    loss=loss.mean().item(),
                    learning_rate=optimizer.param_groups[0]['lr'])

            training_case_loss = torch.zeros(args.k, len(type_order), len(type_order)).cuda()
            count = 0
            for i in range(args.k):
                for j in range(len(type_order)):
                    for k in range(len(type_order)):
                        tmp = loss[type_==count]
                        if tmp.shape[0] > 0:
                            training_case_loss[i, j, k] = tmp.mean()
                        else:
                            training_case_loss[i, j, k] = previous_case_loss[i, j, k]
                        count += 1
            previous_case_loss = training_case_loss.clone().detach()

            loss_ratio = training_case_loss / L0
            inverse_ratio = loss_ratio / loss_ratio[possible_tensor].mean()
            # Small -> Well train

            # Create target
            if args.approximate_naive:
                stochastic_matrix_true = torch.pow(inverse_ratio, args.alpha)
            else:
                stochastic_matrix_true = type_normal_tensor + args.alpha * (inverse_ratio-1)
                stochastic_matrix_true[~possible_tensor] = 0
            # Row normalize
            stochastic_matrix_true = stochastic_matrix_true / stochastic_matrix_true.sum(dim=2, keepdim=True)
            stochastic_matrix_true = stochastic_matrix_true.clone().detach()

            # Predict tensor using stochastic matrix
            stochastic_matrix_pred = []
            tmp = stochastic_matrix
            for i in range(args.k):
                stochastic_matrix_pred.append(tmp)
                tmp = torch.mm(tmp, stochastic_matrix)
            stochastic_matrix_pred = torch.stack(stochastic_matrix_pred)

            # Calculate loss_stochastic
            loss_stochastic = torch.pow(stochastic_matrix_true[possible_tensor] - stochastic_matrix_pred[possible_tensor], 2).mean()

            # 그냥 트레이닝 안 시켯을 때, 움직이는 양
            with torch.no_grad():
                # Predict tensor using stochastic matrix
                stochastic_matrix_pred = []
                tmp = stochastic_matrix_normal
                for i in range(args.k):
                    stochastic_matrix_pred.append(tmp)
                    tmp = torch.mm(tmp, stochastic_matrix_normal)
                stochastic_matrix_pred = torch.stack(stochastic_matrix_pred)
                loss_stochastic_normal = torch.pow(stochastic_matrix_true[possible_tensor] - stochastic_matrix_pred[possible_tensor], 2).mean()


            # Rander graph
            #g = make_dot(loss_stochastic, {})
            #g.render()

            # Backpropagation
            loss_stochastic.backward()
            print('움직였을 때: %.4f, 가만히 있을 때: %.4f' % (loss_stochastic, loss_stochastic_normal))

            # Update stochastic matrix
            with torch.no_grad():
                stochastic_matrix -= args.lr2 * stochastic_matrix.grad
                stochastic_matrix.grad.zero_()
                stochastic_matrix[stochastic_matrix<1e-3] = 1e-3
                # Condition: 1. update only explicit edge, 2. row-normalize
                stochastic_matrix = stochastic_matrix * meta_adjacency_matrix
                stochastic_matrix = stochastic_matrix / stochastic_matrix.sum(dim=1, keepdim=True)
                print(stochastic_matrix)
            stochastic_matrix.requires_grad_(True)

            # Summary
            if n_iter % 10 == 9:
                loss_stochastic_f.write('%f\n' % loss_stochastic.item())
                for hop_count in range(inverse_ratio.shape[0]):
                    for src_type in range(inverse_ratio.shape[1]):
                        for tgt_type in range(inverse_ratio.shape[2]):
                            inverse_ratio_f.write('%.4f ' % inverse_ratio[hop_count, src_type, tgt_type].item())
                inverse_ratio_f.write('\n')
                for src_type in range(stochastic_matrix.shape[0]):
                    for tgt_type in range(stochastic_matrix.shape[1]):
                        stochastic_matrix_f.write('%.4f ' % stochastic_matrix[src_type, tgt_type].item())
                stochastic_matrix_f.write('\n')

                writer.add_scalar('total/loss', loss.mean(), n_iter)
                for t in range(args.k):
                    for i, t1 in enumerate(type_order):
                        for j, t2 in enumerate(type_order):
                            writer.add_scalar('loss%d/%s%s'%(t, t1, t2), training_case_loss[t, i, j], n_iter)
                            writer.add_scalar('ratio%d/%s%s'%(t, t1, t2), inverse_ratio[t, i, j], n_iter)
                layout = {'loss': {'loss': ['loss%d/%s%s' % (t, t1, t2)
                                            for t1 in type_order
                                            for t2 in type_order
                                            for t in range(args.k)]},
                          'ratio': {'ratio': ['ratio%d/%s%s'%(t, t1, t2)
                                              for t1 in type_order
                                              for t2 in type_order
                                              for t in range(args.k)]}}
                writer.add_custom_scalars(layout)
                embedding = model.relationship_embedding.detach().cpu()
                writer.add_image(tag='relationship_embedding',
                                 img_tensor=torch.sigmoid(embedding),
                                 global_step=n_iter,
                                 dataformats='HW')

        np.save(os.path.join('output', get_output_name(args)+'.npy'),
                model.node_embedding.detach().cpu().numpy())
    writer.close()


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    add_argument(parser)
    args = parser.parse_args()
    main(args)
