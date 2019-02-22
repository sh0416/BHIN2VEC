import os
import argparse
from utils import get_preprocessed_data

import numpy as np
import pandas as pd

def get_type(idx, type_interval):
    for k, v in type_interval.items():
        if v[0]<=idx and idx<=v[1]:
            return k
    raise Exception


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='data')
    parser.add_argument('--dataset', type=str, default='dblp', choices=['blog', 'dblp', 'yelp'])
    parser.add_argument('--reverse', action='store_true')
    args = parser.parse_args()

    if args.reverse:
        with open('blog_nodes.txt') as f:
            for idx, line in enumerate(f):
                if idx == 0:
                    data = list(map(int, line.split(' ')))
                    embedding = np.zeros((data[0]+10, data[1]))
                else:
                    data = list(map(float, line.split(' ')))
                    embedding[int(data[0]), :] = data[1:]
        np.save(os.path.join('output', 'hin2vec_%s.npy'%(args.dataset)), embedding)
    else:
        node_df, edge_df, test_edge_df, data = get_preprocessed_data(args)
        print(edge_df.head())
        tmp = edge_df.copy()
        tmp.columns = ['v2', 'v1', 't2', 't1', 'edge_type']
        edge_df = pd.concat((edge_df, tmp), axis=0, sort=False)
        edge_df[['v1', 't1', 'v2', 't2', 'edge_type']].to_csv(args.dataset+'_hin2vec_input.txt', sep='\t', index=False, header=False)
