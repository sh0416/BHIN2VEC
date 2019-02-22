import os
import random
import argparse
from collections import defaultdict
from utils import load_data, create_graph

import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, recall_score, f1_score
import tqdm


def main(args):
    # Load data
    node_type, edge_df, test_node_df, test_edge_df = load_data(args)
    embedding = np.load(args.embedding)

    node_num = max([x[1] for x in node_type.values()]) + 1
    type_order = list(node_type.keys())

    adj_data, adj_size, adj_start = create_graph(edge_df, node_num, type_order)

    for type_idx1, type1 in enumerate(type_order):
        for type_idx2, type2 in enumerate(type_order):
            pos_edge = edge_df[(edge_df['t1']==type1) & (edge_df['t2']==type2)]
            if len(pos_edge) > 0:

                test = test_edge_df[(test_edge_df['t1']==type1) & (test_edge_df['t2']==type2)]
                if len(test) == 0:
                    print('No test edge. skip')
                    continue

                type1_min_idx, type1_max_idx = node_type[type1]
                type2_min_idx, type2_max_idx = node_type[type2]

                pos_train_idx = pos_edge[['v1', 'v2']]

                type1_sampled_idx = np.random.randint(type1_min_idx, type1_max_idx, size=(2*len(pos_train_idx),))
                type2_sampled_idx = np.random.randint(type2_min_idx, type2_max_idx, size=(2*len(pos_train_idx),))
                neg_train_idx = np.vstack((type1_sampled_idx, type2_sampled_idx)).T
                neg_train_idx = pd.DataFrame(neg_train_idx, columns=['v1', 'v2'])

                overlapped_idx = neg_train_idx.reset_index().merge(pos_train_idx, on=list(pos_train_idx.columns)).set_index('index')
                neg_train_idx = neg_train_idx.drop(overlapped_idx.index, axis=0)
                neg_train_idx = neg_train_idx[:len(pos_train_idx)]

                pos_train_idx['l'] = 1
                neg_train_idx['l'] = 0
                train_idx = pd.concat((pos_train_idx, neg_train_idx), axis=0)

                if args.vector_f == 'hadamard':
                    X = embedding[train_idx.loc[:, 'v1']] * embedding[train_idx.loc[:, 'v2']]
                elif args.vector_f == 'average':
                    X = (embedding[train_idx.loc[:, 'v1']] + embedding[train_idx.loc[:, 'v2']]) / 2
                elif args.vector_f == 'minus':
                    X = (embedding[train_idx.loc[:, 'v1']] - embedding[train_idx.loc[:, 'v2']])
                else:
                    X = np.abs(embedding[train_idx.loc[:, 'v1']] - embedding[train_idx.loc[:, 'v2']])
                y = train_idx.loc[:, 'l'].values
                print(X.shape, y.shape)

                classifier = LogisticRegression(solver='liblinear', class_weight='balanced').fit(X, y)

                print('Evaluation start')
                if type1 == type2:
                    test2 = test.copy()
                    test2.columns = ['v2', 'v1', 't2', 't1']
                    test = pd.concat((test, test2), axis=0, sort=False)
                    test = test.drop_duplicates()

                    pos_edge2 = pos_edge.copy()
                    pos_edge2.columns = ['v2', 'v1', 't2', 't1']
                    pos_edge = pd.concat((pos_edge, pos_edge2), axis=0, sort=False)
                    pos_edge = pos_edge.drop_duplicates()

                test = test.groupby('v1')['v2'].apply(set)
                pos_edge = pos_edge.groupby('v1')['v2'].apply(set)
                total_edge = test.reset_index().merge(pos_edge.reset_index(), on='v1').set_index('v1')
                test = test[total_edge.index]
                total_edge['v2'] = total_edge.apply(lambda x: set().union(x['v2_x'], x['v2_y']), axis=1)

                hit, count = 0, 0
                t = tqdm.tqdm(test.iteritems(), total=len(test), ascii=True)
                for i, label in t:
                    for true in label:
                        unobserved_node = list(set(range(type2_min_idx, type2_max_idx+1))-set(total_edge.loc[i, 'v2']))
                        candidate = np.random.choice(unobserved_node, size=min([99, len(unobserved_node)]))
                        candidate = np.concatenate([[true], candidate])

                        if args.vector_f == 'hadamard':
                            X = embedding[i, :] * embedding[candidate, :]
                        elif args.vector_f == 'average':
                            X = (embedding[i, :] + embedding[candidate, :]) / 2
                        elif args.vector_f == 'minus':
                            X = (embedding[i, :] - embedding[candidate, :])
                        else:
                            X = np.abs(embedding[i, :] - embedding[candidate, :])

                        y_pred = classifier.predict_proba(X)
                        y_pred = np.argsort(y_pred[:, 0])[:10]
                        if 0 in y_pred:
                            hit += 1
                        count += 1
                        t.set_postfix(recall=hit/count)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='data')
    parser.add_argument('--dataset', type=str, default='dblp', choices=['blog', 'douban_movie', 'dblp', 'yelp'])
    parser.add_argument('--embedding', type=str, required=True)
    parser.add_argument('--vector_f', type=str, default='hadamard', choices=['hadamard', 'average', 'minus', 'abs_minus'])
    parser.add_argument('--result', type=str, default='')
    args = parser.parse_args()

    main(args)
