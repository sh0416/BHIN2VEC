import os
import random
import argparse
from collections import defaultdict
from utils import load_data

import numpy as np
import pandas as pd


def get_type(idx, type_interval):
    for k, v in type_interval.items():
        if v[0]<=idx and idx<=v[1]:
            return k
    raise Exception


def convert_deepwalk(args):
    if args.reverse:
        with open(os.path.join('other-method', 'deepwalk', '%s.embeddings' % (args.dataset))) as f:
            for idx, line in enumerate(f):
                if idx == 0:
                    data = list(map(int, line.split(' ')))
                    embedding = np.random.normal(scale=0.01, size=(data[0], data[1]))
                else:
                    data = list(map(float, line.split(' ')))
                    embedding[int(data[0]), :] = data[1:]
        np.save(os.path.join('output', 'deepwalk_%s.npy' % (args.dataset)), embedding)
    else:
        _, edge_df, _, _ = load_data(args)
        os.makedirs(os.path.join('other-method', 'deepwalk'), exist_ok=True)
        edge_df[['v1', 'v2']].to_csv(os.path.join('other-method', 'deepwalk', '%s.edgelist' % (args.dataset)),
                                     sep='\t',
                                     index=False,
                                     header=False)


def convert_LINE(args):
    if args.reverse:
        with open(os.path.join('other-method', 'LINE', '%s.embeddings' % (args.dataset))) as f:
            for idx, line in enumerate(f):
                if idx == 0:
                    data = list(map(int, line.split(' ')))
                    embedding = np.random.normal(scale=0.1, size=(data[0], data[1]))
                else:
                    line = line.rstrip()
                    data = list(map(float, line.split(' ')))
                    assert len(data) == embedding.shape[1]+1
                    embedding[int(data[0]), :] = data[1:]
        np.save(os.path.join('output', 'LINE_%s.npy' % (args.dataset)), embedding)
    else:
        _, edge_df, _, _ = load_data(args)
        os.makedirs(os.path.join('other-method', 'LINE'), exist_ok=True)
        tmp = edge_df[['v1', 'v2']].copy()
        tmp2 = edge_df[['v2', 'v1']].copy()
        tmp2.columns = ['v1', 'v2']
        edge_df = pd.concat((tmp, tmp2), axis=0, sort=False)
        edge_df['weight'] = 1
        edge_df[['v1', 'v2', 'weight']].to_csv(os.path.join('other-method', 'LINE', '%s.edgelist' % (args.dataset)),
                                     sep='\t',
                                     index=False,
                                     header=False)

def convert_metapath2vec(args):
    if args.reverse:
        node_type, _, _, _ = load_data(args)
        node_num = sum([max_v-min_v+1 for min_v, max_v in node_type.values()])
        with open(os.path.join('other-method', 'metapath2vec', '%s.txt' % (args.dataset))) as f:
            for idx, line in enumerate(f):
                if idx == 0:
                    data = list(map(int, line.split(' ')))
                    embedding = np.zeros((node_num, data[1]))
                else:
                    line = line.rstrip()
                    index = line.split(' ')[0]
                    data = list(map(float, line.split(' ')[1:]))
                    if index == '</s>':
                        continue
                    embedding[int(index[1:]), :] = data
        np.save(os.path.join('output', 'metapaht2vec_%s.npy' % (args.dataset)), embedding)
    else:
        metapath = args.metapath
        node_type, edge_df, _, _ = load_data(args)
        os.makedirs(os.path.join('other-method', 'metapath2vec'), exist_ok=True)

        # Indice information
        graph = defaultdict(lambda : defaultdict(set))
        for idx, v in edge_df.groupby(by=['v1', 't2'])['v2'].apply(set).iteritems():
            graph[idx[0]][idx[1]] = graph[idx[0]][idx[1]].union(v)
        for idx, v in edge_df.groupby(by=['v2', 't1'])['v1'].apply(set).iteritems():
            graph[idx[0]][idx[1]] = graph[idx[0]][idx[1]].union(v)

        if args.dataset == 'douban_movie':
            type_mapping = {'U': 'v', 'M': 'a', 'A': 'i', 'D': 'f'}
        elif args.dataset == 'yelp':
            type_mapping = {'U': 'v', 'B': 'a', 'C': 'i', 'W': 'f'}
        elif args.dataset == 'dblp':
            type_mapping = {'A': 'v', 'P': 'a', 'T': 'i', 'V': 'f'}
        elif args.dataset == 'aminer':
            type_mapping = {'A': 'v', 'P': 'a', 'R': 'i', 'C': 'f'}
        elif args.dataset == 'blog-catalog':
            type_mapping = {'U': 'v', 'G': 'a'}
        else:
            raise Exception()

        walk = [None] * 100
        with open(os.path.join('other-method', 'metapath2vec', '%s.randomwalk' % (args.dataset)), 'w') as f:
            for i in range(10):
                print('Walking... %d' % i)
                for idx, j in enumerate(graph.keys()):
                    for k, v in node_type.items():
                        if v[0]<=j and j<=v[1]:
                            metapath_type = k
                    if metapath_type not in metapath:
                        continue
                    metapath_idx = metapath.index(metapath_type)
                    walk[0] = type_mapping[metapath_type] + str(j)

                    for k in range(1, 100):
                        if len(graph[int(walk[k-1][1:])][metapath[(metapath_idx+1)%len(metapath)]]) == 0:
                            break
                        walk[k] = type_mapping[metapath[(metapath_idx+1)%len(metapath)]] + \
                                str(random.choice(list(graph[int(walk[k-1][1:])][metapath[(metapath_idx+1)%len(metapath)]])))
                        metapath_idx += 1

                    if idx % 10000 == 9999:
                        print(idx)
                    f.write(' '.join(walk[:k])+'\n')


def convert_hin2vec(args):
    if args.reverse:
        with open(os.path.join('other-method', 'hin2vec', '%s.embeddings' % (args.dataset))) as f:
            for idx, line in enumerate(f):
                if idx == 0:
                    data = list(map(int, line.split(' ')))
                    embedding = np.zeros((data[0], data[1]))
                else:
                    data = list(map(float, line.split(' ')))
                    embedding[int(data[0]), :] = data[1:]
        np.save(os.path.join('output', 'hin2vec_%s.npy'%(args.dataset)), embedding)
    else:
        _, edge_df, _, _ = load_data(args)
        os.makedirs(os.path.join('other-method', 'hin2vec'), exist_ok=True)
        tmp = edge_df[['v1', 't1', 'v2', 't2']].copy()
        tmp2 = edge_df[['v2', 't2', 'v1', 't1']].copy()
        tmp2.columns = ['v1', 't1', 'v2', 't2']
        edge_df = pd.concat((tmp, tmp2), axis=0, sort=False)
        edge_df['edge_type'] = edge_df['t1'] + '-' + edge_df['t2']
        edge_df.to_csv(os.path.join('other-method', 'hin2vec', '%s.edgelist' % (args.dataset)),
                       sep='\t',
                       index=False,
                       header=False)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='data')
    parser.add_argument('--model', type=str, default='deepwalk', choices=['deepwalk', 'LINE', 'metapath2vec', 'hin2vec'])
    parser.add_argument('--metapath', type=str)
    parser.add_argument('--dataset', type=str, default='dblp', choices=['douban_movie', 'blog-catalog', 'dblp', 'yago'])
    parser.add_argument('--reverse', action='store_true')
    args = parser.parse_args()

    if args.model == 'deepwalk':
        convert_deepwalk(args)
    elif args.model == 'LINE':
        convert_LINE(args)
    elif args.model == 'metapath2vec':
        convert_metapath2vec(args)
    elif args.model == 'hin2vec':
        convert_hin2vec(args)
