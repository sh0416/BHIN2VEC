import os
import random
import pickle
import functools
import warnings
import multiprocessing
from collections import deque, defaultdict

import tqdm
import numpy as np
import pandas as pd
import scipy.sparse as sp

import torch


def add_argument(parser):
    parser.add_argument('--root', type=str, default='data')
    parser.add_argument('--model', type=str, default='experimental',
                        choices=['experimental', 'hin2vec', 'metapath2vec'])
    parser.add_argument('--dataset', type=str, default='dblp',
                        choices=['blog-catalog', 'douban_movie', 'dblp', 'dblp-expert-knowledge', 'yago'])
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


def get_name(args):
    if args.model == 'experimental':
        return 'experimental_%s_%d_%d_%d_%d_%.2f_%.6f_%6f' % (args.dataset, args.d, args.l, args.k, args.m, args.alpha, args.lr, args.lr2)
    else:
        return '%s_%s' % (args.model, args.dataset)


def deprecated(replacement=None):
    def outer(fun):
        msg = '%s is deprecated' % fun.__name__
        if replacement is not None:
            msg += '; use %s instead' % replacement
        if fun.__doc__ is None:
            fun.__doc__ = msg

        @functools.wraps(fun)
        def inner(*args, **kwargs):
            warnings.warn(msg, category=DeprecationWarning, stacklevel=1)
            return fun(*args, **kwargs)
        return inner
    return outer


def convert_defaultdict_to_dict(x):
    """nested defaultdict을 dict으로 변환."""
    # Not working if list exist
    if isinstance(x, defaultdict):
        x = {k: convert_defaultdict_to_dict(v) for k, v in x.items()}
    return x


def create_graph(edge_df, node_num, type_order):
    # Indice information
    graph = defaultdict(lambda : defaultdict(set))
    for idx, v in edge_df.groupby(by=['v1', 't2'])['v2'].apply(set).iteritems():
        graph[idx[0]][idx[1]] = graph[idx[0]][idx[1]].union(v)
    for idx, v in edge_df.groupby(by=['v2', 't1'])['v1'].apply(set).iteritems():
        graph[idx[0]][idx[1]] = graph[idx[0]][idx[1]].union(v)

    # Compact representation of graph
    adj_data = [graph[x] for x in range(node_num)]
    adj_data = [[x[y] for y in type_order] for x in adj_data]
    adj_size = [[len(y) for y in x] for x in adj_data]
    adj_start = [[None]*len(type_order) for _ in range(node_num)]
    count = 0
    for i in range(node_num):
        for j in range(len(type_order)):
            adj_start[i][j] = count
            count += adj_size[i][j]
    adj_data = [item for sublist in adj_data for subsublist in sublist for item in subsublist]

    return adj_data, adj_size, adj_start


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


def load_metadata(data_dir):
    with open(os.path.join(data_dir, 'node_type.pickle'), 'rb') as f:
        node_type = pickle.load(f)
    return node_type


def load_data(args):
    data_dir = os.path.join(args.root, args.dataset)
    node_df, test_edge_df = None, None
    if args.dataset == 'blog-catalog':
        node_type = load_metadata(data_dir)
        edge_df = pd.read_csv(os.path.join(data_dir, 'edge.csv'), sep='\t')
        test_edge_df = pd.read_csv(os.path.join(data_dir, 'test_edge.csv'), sep='\t')
    elif args.dataset == 'douban_movie':
        node_type = load_metadata(data_dir)
        edge_df = pd.read_csv(os.path.join(data_dir, 'edge.csv'), sep='\t')
        test_edge_df = pd.read_csv(os.path.join(data_dir, 'test_edge.csv'), sep='\t')
    elif args.dataset == 'dblp':
        node_type = load_metadata(data_dir)
        node_df = pd.read_csv(os.path.join(data_dir, 'node.csv'), sep='\t')
        edge_df = pd.read_csv(os.path.join(data_dir, 'edge.csv'), sep='\t')
    elif args.dataset == 'dblp-expert-knowledge':
        data_dir = os.path.join(args.root, 'dblp')
        node_type = load_metadata(data_dir)
        node_df = pd.read_csv(os.path.join(data_dir, 'node.csv'), sep='\t')
        edge_df = pd.read_csv(os.path.join(data_dir, 'edge.csv'), sep='\t')
        # A-P인 것과 V-P인 것만 남기고 나머지 데이터 모두 제거
        edge_df_AP = edge_df[(edge_df['t1']=='A') & (edge_df['t2']=='P')]
        edge_df_PA = edge_df[(edge_df['t1']=='P') & (edge_df['t2']=='A')]
        edge_df_VP = edge_df[(edge_df['t1']=='V') & (edge_df['t2']=='P')]
        edge_df_PV = edge_df[(edge_df['t1']=='P') & (edge_df['t2']=='V')]
        edge_df = pd.concat([edge_df_AP, edge_df_PA, edge_df_VP, edge_df_PV], sort=False)
    elif args.dataset == 'yago':
        with open(os.path.join(data_dir, 'node_type.pickle'), 'rb') as f:
            node_type = pickle.load(f)
        edge_df = pd.read_csv(os.path.join(data_dir, 'train_edge.csv'), sep='\t')
        test_edge_df = pd.read_csv(os.path.join(data_dir, 'test_edge.csv'), sep='\t')
    else:
        raise Exception("Undefined dataset")
    return node_type, edge_df, node_df, test_edge_df


def create_adjacency_matrix(adj_list: dict, shape: (int, int)):
    row = [k for k, v in adj_list.items() for _ in v]
    col = [x for _, v in adj_list.items() for x in v]

    adj_matrix = sp.coo_matrix((np.ones(len(row)), (row, col)), shape=shape).tocsr()
    return adj_matrix
