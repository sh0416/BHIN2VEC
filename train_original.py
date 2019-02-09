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

    parser.add_argument('--model',
                        type=str,
                        default='deepwalk',
                        choices=['deepwalk', 'metapath2vec', 'just'])

    parser.add_argument('--metapath',
                        type=str,
                        default='APTPVPTP')

    parser.add_argument('--alpha',
                        type=float,
                        default=0.5)

    parser.add_argument('--que_size',
                        type=int,
                        default=2)

    parser.add_argument('--epoch',
                        type=int,
                        default=10)

    parser.add_argument('--batch_size',
                        type=int,
                        default=128)

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
    name = '%s_%d_%d_%d_%d' % (args.model, args.d, args.l, args.k, args.m)
    if args.model == 'metapath2vec':
        name += '_%s' % (args.metapath)
    elif args.model == 'just':
        name += '_%.2f_%d' % (args.alpha, args.que_size)
    return name

def main(args):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    #handler = logging.StreamHandler()
    #logger.addHandler(handler)

    if args.model == 'deepwalk':
        data = get_preprocessed_data(args, type_split=False)
    elif args.model == 'metapath2vec':
        data = get_preprocessed_data(args, type_split=True)
    elif args.model == 'just':
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
    elif args.model == 'metapath2vec':
        typechecker = TypeChecker(type_interval)
        walk_fun = functools.partial(metapath_walk,
                                     adj_dict=data['adj_indice'],
                                     metapath=args.metapath,
                                     typechecker=typechecker)
    elif args.model == 'just':
        typechecker = TypeChecker(type_interval)
        walk_fun = functools.partial(just_walk,
                                     adj_dict=data['adj_indice'],
                                     typechecker=typechecker,
                                     alpha=args.alpha,
                                     que_size=args.que_size)
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

    model = SkipGramModel(node_num, args.d, args.l, args.k, args.m).cuda()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-2)

    if args.restore:
        embedding = np.load(os.path.join('output', get_output_name(args)+'.npy'))
        model.node_embedding.weight.data.copy_(torch.from_numpy(embedding))

    os.makedirs('output', exist_ok=True)

    # TensorboardX
    writer = SummaryWriter(os.path.join('runs', get_output_name(args)))

    n_iter = 0
    for epoch in range(args.epoch):
        epoch_loss = 0
        for idx, (walk, negative) in tqdm.tqdm(enumerate(loader), total=len(loader), ascii=True):
            walk = walk.cuda()
            negative = negative.cuda()

            optimizer.zero_grad()
            pos, neg = model(walk, negative)

            label_pos = torch.ones_like(pos)
            label_neg = torch.zeros_like(neg)

            y = torch.cat((pos, neg), dim=2)
            label = torch.cat((label_pos, label_neg), dim=2)
            # [B, L-K, K+M]

            loss = criterion(y, label)
            epoch_loss += loss
            loss.backward()
            optimizer.step()

            if idx % 10 == 9:
                writer.add_scalar('data/loss', loss, n_iter)
            n_iter += 1
        epoch_loss /= len(loader)

        # Save embedding
        total_embedding = model.node_embedding.weight.data
        for k, v in type_interval.items():
            writer.add_embedding(total_embedding[v[0]:v[1]+1, :],
                                 metadata=class_dict.get(k),
                                 global_step=epoch, tag=k)

    np.save(os.path.join('output', get_output_name(args)+'.npy'),
            model.node_embedding.weight.detach().cpu().numpy())
    writer.close()

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    add_argument(parser)
    args = parser.parse_args()
    main(args)

