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


def load_data(args):
    root = args.root
    if args.dataset == 'dblp':
        with open(os.path.join(root, 'dblp', 'node_type.pickle'), 'rb') as f:
            node_type = pickle.load(f)
        edge_df = pd.read_csv(os.path.join(root, 'dblp', 'edge.csv'), sep='\t', usecols=['v1', 'v2', 't1', 't2'])
        test_node_df = {'A': pd.read_csv(os.path.join(root, 'dblp', 'node_classification', 'author_label.csv'),
                                         sep='\t', usecols=['index', 'L']),
                        'P': pd.read_csv(os.path.join(root, 'dblp', 'node_classification', 'paper_label.csv'),
                                         sep='\t', usecols=['index', 'L'])}
        test_node_df['A'].columns = ['A', 'L']
        test_node_df['P'].columns = ['P', 'L']
        test_edge_df = {}
    elif args.dataset == 'blog-catalog':
        with open(os.path.join(root, 'blog-catalog', 'node_type.pickle'), 'rb') as f:
            node_type = pickle.load(f)
        edge_df = pd.read_csv(os.path.join(root, 'blog-catalog', 'edge.csv'), sep='\t')
        test_node_df = {}
        test_edge_df = pd.read_csv(os.path.join(root, 'blog-catalog', 'test_edge.csv'), sep='\t')
    elif args.dataset == 'douban_movie':
        with open(os.path.join(root, 'douban_movie', 'node_type.pickle'), 'rb') as f:
            node_type = pickle.load(f)
        edge_df = pd.read_csv(os.path.join(root, 'douban_movie', 'edge.csv'), sep='\t')
        test_node_df = {}
        test_edge_df = pd.read_csv(os.path.join(root, 'douban_movie', 'test_edge.csv'), sep='\t')
    elif args.dataset == 'yago':
        with open(os.path.join(root, 'yago', 'node_type.pickle'), 'rb') as f:
            node_type = pickle.load(f)
        edge_df = pd.read_csv(os.path.join(root, 'yago', 'train_edge.csv'), sep='\t')
        test_node_df = {}
        test_edge_df = pd.read_csv(os.path.join(root, 'yago', 'test_edge.csv'), sep='\t')
    elif args.dataset == 'yelp':
        with open(os.path.join(root, 'yelp', 'node_type.pickle'), 'rb') as f:
            node_type = pickle.load(f)
        edge_df = pd.read_csv(os.path.join(root, 'yelp', 'edge.csv'), sep='\t', usecols=['v1', 'v2', 't1', 't2'])
        test_node_df = {'B': pd.read_csv(os.path.join(root, 'yelp', 'node_classification', 'business_category.csv'),
                                         sep='\t',
                                         usecols=['B', 'L']),
                        'C': pd.read_csv(os.path.join(root, 'yelp', 'node_classification', 'city_state.csv'),
                                         sep='\t',
                                         usecols=['C', 'L'])}
        test_edge_df = {}
    elif args.dataset == 'aminer':
        with open(os.path.join(root, 'aminer', 'node_type.pickle'), 'rb') as f:
            node_type = pickle.load(f)
        edge_df = pd.read_csv(os.path.join(root, 'aminer', 'edge.csv'), sep='\t', usecols=['v1', 'v2', 't1', 't2'])
        test_node_df = {'P': pd.read_csv(os.path.join(root, 'aminer', 'node_classification', 'paper_label.csv'),
                                         sep='\t',
                                         usecols=['P', 'L'])}
        test_edge_df = {'PA': pd.read_csv(os.path.join(root, 'aminer', 'link_prediction', 'test_paper_author.csv'),
                                          sep='\t',
                                          usecols=['v1', 'v2', 't1', 't2']),
                        'PC': pd.read_csv(os.path.join(root, 'aminer', 'link_prediction', 'test_paper_conf.csv'),
                                          sep='\t',
                                          usecols=['v1', 'v2', 't1', 't2']),
                        'PR': pd.read_csv(os.path.join(root, 'aminer', 'link_prediction', 'test_paper_reference.csv'),
                                          sep='\t',
                                          usecols=['v1', 'v2', 't1', 't2'])}
    return node_type, edge_df, test_node_df, test_edge_df


def create_adjacency_matrix(adj_list: dict, shape: (int, int)):
    row = [k for k, v in adj_list.items() for _ in v]
    col = [x for _, v in adj_list.items() for x in v]

    adj_matrix = sp.coo_matrix((np.ones(len(row)), (row, col)), shape=shape).tocsr()
    return adj_matrix


@deprecated()
def preprocess(node_df_dict, edge_df_dict):
    # Deprecated
    print('Preprocessing: Re-indexing node index')
    start_idx = 0
    node_idx_dict = {}
    for k, v in node_df_dict.items():
        node_idx_dict[k] = {x: i for i, x in enumerate(v['index'], start=start_idx)}
        start_idx += len(v)
    for k in edge_df_dict.keys():
        edge_df_dict[k][0] = edge_df_dict[k][0].apply(lambda x: node_idx_dict[k[0]][x])
        edge_df_dict[k][1] = edge_df_dict[k][1].apply(lambda x: node_idx_dict[k[1]][x])
        edge_df_dict[k][2] = k[0]
        edge_df_dict[k][3] = k[1]
    for k in node_df_dict.keys():
        node_df_dict[k]['index'] = node_df_dict[k]['index'].apply(lambda x: node_idx_dict[k][x])
        node_df_dict[k]['type'] = k

    print('Preprocessing: merge dict')
    node_df = pd.concat(node_df_dict.values(), axis=0, sort=False)
    edge_df = pd.concat(edge_df_dict.values(), axis=0, sort=False)
    edge_df.columns = ['v1', 'v2', 't1', 't2']

    print('Preprocessing: make symmetric and filter unique edge')
    tmp = edge_df.copy()
    tmp.columns = ['v2', 'v1', 't2', 't1']
    edge_df = pd.concat((edge_df, tmp), axis=0, sort=False)
    edge_df = edge_df.drop_duplicates()

    # Take unique edge
    edge_df = edge_df[edge_df['v1']<edge_df['v2']]
    print('\tFinal edge number: %d' % len(edge_df))
    edge_df['edge_type'] = edge_df['t1'] + '-' + edge_df['t2']
    edge_df = edge_df.reset_index(drop=True)

    print('Preprocessing: Remove edge for link prediction')
    test_edge_df = edge_df.groupby('edge_type').apply(lambda x: x.sample(len(x)//10)).reset_index(drop=True)
    test_edge_df = edge_df.reset_index().merge(test_edge_df, on=list(edge_df.columns)).set_index('index')
    edge_df = edge_df.drop(test_edge_df.index, axis=0)
    print('\tAfter removing testing edge: %d' % len(edge_df))

    return node_df, edge_df, test_edge_df


if __name__=='__main__':
    preprocess(None, None)
