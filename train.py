import os
import shutil
import pickle
import logging
import argparse
import functools
from decimal import Decimal
from collections import defaultdict

import tqdm
import numpy as np
import scipy.sparse as sp
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, recall_score, f1_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter
#from graphviz import Digraph

from models import BalancedSkipGramModel
from utils import load_data, create_graph, make_dot, get_name, add_argument


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


def train(args, node_type, edge_df, logger):
    # pyTorch 세팅
    np.random.seed(0)
    torch.manual_seed(0)
    torch.set_printoptions(precision=7)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

    logger.info("Type: %s" % str(type_order))
    logger.info("Node: %d , Edge: %d" % (node_num, len(edge_df)))

    node_idx = torch.arange(node_num)
    possible_node_idx = node_idx[adj_size.sum(dim=1)>0]
    num_iter = possible_node_idx.shape[0] // args.batch_size

    model = BalancedSkipGramModel(node_num, type_num, args.d, args.l, args.k, args.m).to(device)
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Report result (embedding result, TensorboardX)
    os.makedirs('output', exist_ok=True)
    logdir_fpath = os.path.join('log', get_name(args))
    if os.path.exists(logdir_fpath):
        shutil.rmtree(logdir_fpath)
    writer = SummaryWriter(logdir_fpath)

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
    tmp = meta_adjacency_matrix / (meta_adjacency_matrix.sum(dim=1, keepdim=True) + 1e-15)
    for i in range(args.k):
        type_normal_tensor[i] = tmp
        tmp = torch.mm(tmp, meta_adjacency_matrix / (meta_adjacency_matrix.sum(dim=1, keepdim=True)) + 1e-15)
    possible_tensor = type_normal_tensor > 0

    # Stochastic matrix
    stochastic_matrix = meta_adjacency_matrix
    stochastic_matrix = stochastic_matrix / (stochastic_matrix.sum(dim=1, keepdim=True) + 1e-15)
    stochastic_matrix = stochastic_matrix.clone().detach().requires_grad_(True)

    stochastic_matrix_normal = stochastic_matrix.clone().detach().requires_grad_(False)

    # Loss
    previous_case_loss = torch.full((args.k, len(type_order), len(type_order)), fill_value=0.6931)

    # Log file for each data structure
    loss_stochastic_f = open(os.path.join('log', get_name(args)+'_loss_stochastic.log'), 'w')
    inverse_ratio_f = open(os.path.join('log', get_name(args)+'_inverse_ratio.log'), 'w')
    stochastic_matrix_f = open(os.path.join('log', get_name(args)+'_stochastic_matrix.log'), 'w')

    # Counter
    n_iter = 0
    for epoch in range(args.epoch):
        # Shuffle start node
        with torch.no_grad():
            random_idx = torch.randperm(possible_node_idx.shape[0])
            node_idx_epoch = possible_node_idx[random_idx][:num_iter*args.batch_size].view(num_iter, args.batch_size)

        training_bar = tqdm.tqdm(range(num_iter), total=num_iter, ascii=True)
        for idx in training_bar:
            with torch.no_grad():
                node = node_idx_epoch[idx]
                node_t = type_indicator[node]

                walk, walk_type = biased_walk(node, node_t, args.l, stochastic_matrix.cpu(), adj_data, adj_size, adj_start)
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
                #assert torch.all(torch.eq(pos_type.unsqueeze(3), neg_type))
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
            loss_ratio[~possible_tensor] = 0
            inverse_ratio = loss_ratio / (loss_ratio.sum(dim=2, keepdim=True)/(possible_tensor.sum(dim=2, keepdim=True).float()+1e-15))
            # Small -> Well train
            #print('POSSIBLE', possible_tensor[0])
            # Create target
            if args.approximate_naive:
                stochastic_matrix_true = torch.pow(inverse_ratio, args.alpha)
                stochastic_matrix_true[~possible_tensor] = 0
            else:
                stochastic_matrix_true = type_normal_tensor + args.alpha * (inverse_ratio-1)
                stochastic_matrix_true.clamp_(0, 1)
                stochastic_matrix_true[~possible_tensor] = 0
            # Row normalize
            stochastic_matrix_true = stochastic_matrix_true / (stochastic_matrix_true.sum(dim=2, keepdim=True) + 1e-15)
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
            #loss_stochastic = torch.pow(stochastic_matrix_true[0, possible_tensor[0]] - stochastic_matrix_pred[0, possible_tensor[0]], 2).mean()

            # Backpropagation
            loss_stochastic.backward()
            #print('TYPE NORMAL', type_normal_tensor[0])
            #print('INVERSE RATIO', inverse_ratio[0])
            #print('REF', stochastic_matrix_true[0])
            #print('MAT', stochastic_matrix)
            #print('GRAD', stochastic_matrix.grad)

            # Predict tensor using stochastic matrix
            stochastic_matrix_pred = []
            tmp = stochastic_matrix_normal
            for i in range(args.k):
                stochastic_matrix_pred.append(tmp)
                tmp = torch.mm(tmp, stochastic_matrix_normal)
            stochastic_matrix_pred = torch.stack(stochastic_matrix_pred)

            # Calculate loss_stochastic
            loss_stochastic_normal = torch.pow(stochastic_matrix_true[possible_tensor] - stochastic_matrix_pred[possible_tensor], 2).mean()

            # Update stochastic matrix
            with torch.no_grad():
                #print('Before', stochastic_matrix_true)
                #print(stochastic_matrix.grad)
                stochastic_matrix -= args.lr2 * stochastic_matrix.grad
                #print('AFTER', stochastic_matrix)
                stochastic_matrix.grad.zero_()
                stochastic_matrix[stochastic_matrix<1e-3] = 1e-3
                # Condition: 1. update only explicit edge, 2. row-normalize
                stochastic_matrix = stochastic_matrix * meta_adjacency_matrix
                stochastic_matrix = stochastic_matrix / (stochastic_matrix.sum(dim=1, keepdim=True) + 1e-15)
                #print(stochastic_matrix)
            stochastic_matrix.requires_grad_(True)

            # Summary
            if n_iter % 10 == 9:
                """
                loss_stochastic_f.write('%f %f\n' % (loss_stochastic.item(), loss_stochastic_normal.item()))
                for hop_count in range(inverse_ratio.shape[0]):
                    for src_type in range(inverse_ratio.shape[1]):
                        for tgt_type in range(inverse_ratio.shape[2]):
                            inverse_ratio_f.write('%.4f ' % inverse_ratio[hop_count, src_type, tgt_type].item())
                inverse_ratio_f.write('\n')
                """
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
                relationship_embedding = model.relationship_embedding.detach().cpu()
                writer.add_image(tag='relationship_embedding',
                                 img_tensor=torch.sigmoid(relationship_embedding),
                                 global_step=n_iter,
                                 dataformats='HW')
    writer.close()

    # 마지막 임베딩 저장 후 리턴
    node_embedding = model.node_embedding.detach().cpu().numpy()
    np.save(os.path.join('output', get_name(args)+'.npy'), node_embedding)
    return node_embedding


def main(args):
    # 로거 생성
    os.makedirs('log', exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())

    # 데이터 로드
    node_type, edge_df, _, _ = load_data(args)

    filename = os.path.join('log', get_name(args)+'.log')
    handler = logging.FileHandler(filename)
    # 훈련
    logger.addHandler(handler)
    node_embedding = train(args, node_type, edge_df, logger)
    logger.removeHandler(handler)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    add_argument(parser)
    args = parser.parse_args()
    main(args)
