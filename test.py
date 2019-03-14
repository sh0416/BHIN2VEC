import pickle
import argparse
import functools
import concurrent.futures
from collections import defaultdict

import tqdm
import numpy as np
import scipy.sparse as sp

from utils import load_data, convert_defaultdict_to_dict

def add_argument(parser):
    parser.add_argument('--root', type=str, default='data')
    parser.add_argument('--dataset', type=str, default='dblp', choices=['blog', 'douban_movie', 'dblp', 'yelp'])
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--d', type=int, default=128)
    parser.add_argument('--l', type=int, default=100)
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--m', type=int, default=5)


def random_walk(i, adj, type_vector):
    w, w_t = [None] * 80, [None] * 80
    w[0] = i
    w_t[0] = type_vector[w[0]]
    for i in range(1, 80):
        w[i] = np.random.choice(adj[w[i-1],:].indices, size=1)[0]
        w_t[i] = type_vector[w[i]]
    return w, w_t


def create_dataset(node_num, adj, type_vector):
    dataset = defaultdict(set)
    f = functools.partial(random_walk, adj=adj, type_vector=type_vector)
    for i in tqdm.tqdm(range(node_num), ascii=True):
        w, w_t = f(i)
        for i in range(80-5):
            m = str(w_t[i])
            for j in range(1, 6):
                m += str(w_t[i+j])
                dataset[m] = (w[i], w[j])
    return dataset

def main(args):
    node_type, edge_df, test_node_df, _ = load_data(args)

    node_num = max([x[1] for x in node_type.values()]) + 1
    type_num = len(node_type)
    type_order = list(node_type.keys())

    adj = sp.coo_matrix((np.ones(len(edge_df)), (edge_df['v1'], edge_df['v2'])), shape=(node_num, node_num))
    adj = adj + adj.T
    adj[adj!=0] = 1
    adj.eliminate_zeros()

    for i in range(node_num):
        assert all([x==y for x, y in zip(adj[i,:].indices, sp.find(adj[i,:])[1])])

    type_vector = np.zeros(node_num, dtype=np.int)
    for idx, k in enumerate(type_order):
        type_vector[node_type[k][0]:node_type[k][1]+1] = idx

    dataset = defaultdict(set)
    with concurrent.futures.ProcessPoolExecutor() as exec:
        f = functools.partial(create_dataset, adj=adj, type_vector=type_vector)
        for d in exec.map(f, [node_num]*10):
            for k, v in d.items():
                dataset[k] = dataset[k].union(v)

    dataset = convert_defaultdict_to_dict(dataset)
    with open('dataset.pickle', 'wb') as handle:
        pickle.dump(dataset, handle)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    add_argument(parser)
    args = parser.parse_args()
    main(args)
