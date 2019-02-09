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
from utils import get_preprocessed_data, deepwalk, metapath_walk, TypeChecker, pca, balanced_walk, BalancedWalkFactory
from random_walk_statistic import normalize_row, random_walk_2, create_adjacency_matrix


def add_argument(parser):
    parser.add_argument('--root',
                        type=str,
                        default='data')

    parser.add_argument('--dataset',
                        type=str,
                        default='dblp',
                        choices=['dblp', 'yelp'])

    parser.add_argument('--model',
                        type=str,
                        default='experimental',
                        choices=['deepwalk', 'metapath2vec', 'experimental'])

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
    return '%s_%d_%d_%d_%d' % (args.model, args.d, args.l, args.k, args.m)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    add_argument(parser)
    args = parser.parse_args()

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler()
    logger.addHandler(handler)

    data = get_preprocessed_data(args, type_split=True)

    node_num = data['node_num']
    degree_dist = data['degree']
    class_dict = data['class']
    type_interval = data['type_interval']
    logger.info("Total type number: " + str(node_num))
    logger.info("Total node number: ")
    for k, v in data['type_interval'].items():
        logger.info('\t\t' + k + ': ' + str(v[1]-v[0]+1))
    logger.info("Degree distribution:" + str(degree_dist))

    if args.model == 'deepwalk':
        walk_fun = functools.partial(deepwalk, adj=data['adj_indice'])
        dataset = RandomWalkDataset(node_num,
                                    walk_fun=walk_fun,
                                    neg_dist=degree_dist,
                                    l=args.l,
                                    k=args.k,
                                    m=args.m)
        loader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=24)
    elif args.model == 'metapath2vec':
        typechecker = TypeChecker(type_interval)
        walk_fun = functools.partial(metapath_walk,
                                     adj_dict=data['adj_indice'],
                                     metapath=args.metapath,
                                     typechecker=typechecker)
        dataset = RandomWalkDataset(node_num,
                                    walk_fun=walk_fun,
                                    neg_dist=degree_dist,
                                    l=args.l,
                                    k=args.k,
                                    m=args.m)
        loader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=24)
    elif args.model == 'experimental':
        type_str_dict = {char: num for num, char in enumerate(type_interval.keys())}
        type_str_inverse_dict = {num: char for char, num in type_str_dict.items()}
        logger.info(str(type_str_dict))
        logger.info(str(type_str_inverse_dict))

        possible_type_mask = torch.zeros(len(data['adj_indice'].keys()), 4)
        for k, v in data['adj_indice'].items():
            for k2 in v.keys():
                possible_type_mask[k, type_str_dict[k2]] = 1

        adjacent_node_num_per_type = torch.zeros(data['node_num'], 4, dtype=torch.long)
        for k, v in data['adj_indice'].items():
            for k2, v2 in v.items():
                adjacent_node_num_per_type[k, type_str_dict[k2]] = len(v2)

        start_idx = torch.zeros(data['node_num'], 4, dtype=torch.long)
        tmp = torch.zeros(4, dtype=torch.long)
        for i in range(data['node_num']):
            start_idx[i, :] = tmp
            tmp += adjacent_node_num_per_type[i]

        packed_adj = [None] * 4
        for type_k, type_idx in type_str_dict.items():
            packed_adj[type_idx] = [None] * data['node_num']
            for k, v in data['adj_indice'].items():
                packed_adj[type_idx][k] = torch.tensor(list(v[type_k]) if v.get(type_k) is not None else [], dtype=torch.long)
            packed_adj[type_idx] = torch.cat(packed_adj[type_idx], dim=0)
        pad_packed_adj = nn.utils.rnn.pad_sequence(packed_adj)

        balanced_walk = BalancedWalkFactory(possible_type_mask,
                                    adjacent_node_num_per_type,
                                    start_idx,
                                    pad_packed_adj,
                                    torch.from_numpy(np.asarray([[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]])).float(),
                                    data['type_interval'],
                                    type_str_dict)
        balanced_walk = balanced_walk.to(torch.device("cuda:0")) 
        num_iter = data['node_num'] // args.batch_size
        start_node_idx = torch.randperm(data['node_num'])[:num_iter*args.batch_size].view(num_iter, args.batch_size).cuda()
        """
        dataset = RandomWalkDataset(node_num,
                                    walk_fun=walk_fun,
                                    neg_dist=degree_dist,
                                    l=args.l,
                                    k=args.k,
                                    m=args.m)
        """

    if args.model == 'deepwalk' or args.model == 'metapath2vec':
        model = SkipGramModel(node_num, args.d, args.l, args.k, args.m).cuda()
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-2)
    elif args.model == 'experimental':
        criterion = nn.BCEWithLogitsLoss(reduction='none')
        model = BalancedSkipGramModel(node_num,
                                      args.d,
                                      args.l,
                                      args.k,
                                      args.m,
                                      type_interval,
                                      type_str_dict,
                                      type_str_inverse_dict,
                                      criterion).cuda()
        optimizer = optim.Adam(model.parameters(), lr=1e-2)

    if args.restore:
        embedding = np.load(os.path.join('output', get_output_name(args)+'.npy'))
        model.node_embedding.weight.data.copy_(torch.from_numpy(embedding))

    os.makedirs('output', exist_ok=True)

    # TensorboardX
    writer = SummaryWriter(os.path.join('runs', get_output_name(args)))

    n_iter = 0
    L0 = 0.6931  # Theoretical initial loss
    for epoch in range(args.epoch):
        for idx in tqdm.tqdm(range(num_iter), total=num_iter, ascii=True):
            walk = balanced_walk(start_node_idx[idx], args.l).t()
            negative = torch.multinomial(torch.from_numpy(degree_dist).cuda(),
                                         args.batch_size*args.m*(args.l-args.k),
                                         replacement=True).view(args.batch_size,
                                                                args.l-args.k,
                                                                args.m)

            optimizer.zero_grad()
            loss, loss_per_type= model(walk, negative)
            loss_ratio = loss_per_type / L0
            inverse_ratio = loss_ratio / loss_ratio.mean()
            # Small -> Well train

            balanced_walk.type_attention = inverse_ratio.view(4, 4)
            loss.backward()
            optimizer.step()

            writer.add_scalar('data/loss', loss, n_iter)
            for i in range(16):
                writer.add_scalar('data/loss%d'%i, loss_per_type[i], n_iter)
                writer.add_scalar('ratio/ratio%d'%i, inverse_ratio[i], n_iter)
            n_iter += 1

        # Save embedding
        total_embedding = model.node_embedding.weight.data
        for k, v in type_interval.items():
            writer.add_embedding(total_embedding[v[0]:v[1]+1, :],
                                 metadata=class_dict.get(k),
                                 global_step=epoch, tag=k)

    np.save(os.path.join('output', get_output_name(args)+'.npy'),
            model.node_embedding.weight.detach().cpu().numpy())
    writer.close()
