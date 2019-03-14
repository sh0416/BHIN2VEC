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


def create_negative_train_data(train_edge, src_type, tgt_type, node_type, src_vertex, tgt_vertex):
    src_type_min_idx, src_type_max_idx = node_type[src_type]
    tgt_type_min_idx, tgt_type_max_idx = node_type[tgt_type]

    # Create sample negative
    src_type_sampled_idx = np.random.randint(src_type_min_idx, src_type_max_idx, size=(2*len(train_edge),))
    tgt_type_sampled_idx = np.random.randint(tgt_type_min_idx, tgt_type_max_idx, size=(2*len(train_edge),))

    # Create dataframe
    train_edge_negative = np.vstack((src_type_sampled_idx, tgt_type_sampled_idx)).T
    train_edge_negative = pd.DataFrame(train_edge_negative, columns=[src_vertex, tgt_vertex])

    # Remove overlapped edge
    overlapped_idx = train_edge_negative.reset_index().merge(train_edge, on=list(train_edge.columns)).set_index('index')
    train_edge_negative = train_edge_negative.drop(overlapped_idx.index, axis=0)
    train_edge_negative = train_edge_negative[:len(train_edge)]
    return train_edge_negative


def create_edge_embedding(embedding, edge, src_vertex, tgt_vertex, vector_f):
    if vector_f == 'hadamard':
        return embedding[edge.loc[:, src_vertex]] * embedding[edge.loc[:, tgt_vertex]]
    elif vector_f == 'average':
        return (embedding[edge.loc[:, src_vertex]] + embedding[edge.loc[:, tgt_vertex]]) / 2
    elif vector_f == 'minus':
        return (embedding[edge.loc[:, src_vertex]] - embedding[edge.loc[:, tgt_vertex]])
    else:
        raise Exception('Invalid vector function')


def evaluate(node_embedding, train_edge, test_edge, src_type, tgt_type, node_type, vector_f):
    assert len(train_edge['t1'].unique()) == 1
    assert len(train_edge['t2'].unique()) == 1
    if train_edge['t1'].unique()[0] == src_type and train_edge['t2'].unique()[0] == tgt_type:
        src_vertex = 'v1'
        tgt_vertex = 'v2'
    elif train_edge['t1'].unique()[0] == tgt_type and train_edge['t2'].unique()[0] == src_type:
        src_vertex = 'v2'
        tgt_vertex = 'v1'
    else:
        raise Exception("source type and target type doesn't match")

    train_edge = train_edge[[src_vertex, tgt_vertex]]
    train_edge_negative = create_negative_train_data(train_edge, src_type, tgt_type, node_type, src_vertex, tgt_vertex)

    # Add label
    train_edge['l'] = 1
    train_edge_negative['l'] = 0
    train_edge = pd.concat((train_edge, train_edge_negative), axis=0)

    # Create edge embedding
    X = create_edge_embedding(node_embedding, train_edge, src_vertex, tgt_vertex, vector_f)
    y = train_edge.loc[:, 'l'].values

    # Train classifier
    classifier = LogisticRegression(solver='liblinear', class_weight='balanced').fit(X, y)

    # Evaluation protocol
    src_type_min_idx, src_type_max_idx = node_type[src_type]
    tgt_type_min_idx, tgt_type_max_idx = node_type[tgt_type]

    test_set = test_edge.groupby(src_vertex)[tgt_vertex].apply(set)
    train_set = train_edge.groupby(src_vertex)[tgt_vertex].apply(set)
    total_set = test_set.reset_index().merge(train_set.reset_index(), on=src_vertex).set_index(src_vertex)
    total_set = total_set.apply(lambda x: set().union(x[tgt_vertex+'_x'], x[tgt_vertex+'_y']), axis=1)
    unobserved_list = total_set.apply(lambda x: np.random.choice(list(set(range(tgt_type_min_idx, tgt_type_max_idx+1)) - x), size=99))

    # Remove test node which doesn't observed in the training time
    test_set = test_set[total_set.index]


    hit, count = 0, 0
    # For each test edge,
    t = tqdm.tqdm(test_set.iteritems(), total=len(test_set), ascii=True)
    for i, label in t:
        for true in label:
            candidate = unobserved_list[i]
            candidate_edge = np.concatenate([[true], candidate])
            candidate_edge = pd.DataFrame(candidate_edge, columns=[tgt_vertex])
            candidate_edge[src_vertex] = i

            X = create_edge_embedding(node_embedding, candidate_edge, src_vertex, tgt_vertex, vector_f)

            y_pred = classifier.predict_proba(X)
            y_pred = np.argsort(y_pred[:, 0])[:10]
            if 0 in y_pred:
                hit += 1
            count += 1
            t.set_postfix(hit_rate=hit/count)
    return hit/count


def main(args):
    # Load data
    node_type, edge_df, test_node_df, test_edge_df = load_data(args)
    embedding = np.load(args.embedding)

    node_num = max([x[1] for x in node_type.values()]) + 1
    type_order = list(node_type.keys())

    adj_data, adj_size, adj_start = create_graph(edge_df, node_num, type_order)

    for type_idx1, type1 in enumerate(type_order):
        for type_idx2, type2 in enumerate(type_order):
            train_edge = edge_df[(edge_df['t1']==type1) & (edge_df['t2']==type2)]
            test_edge = test_edge_df.get(type1+type2, None)
            if len(train_edge) == 0 or test_edge is None:
                continue

            # 두 타입이 같으면 반대 방향 에지도 포함
            if type1 == type2:
                train_edge2 = train_edge.copy()
                train_edge2.columns = ['v2', 'v1', 't2', 't1']
                train_edge = pd.concat((train_edge, train_edge2), axis=0, sort=False)
                train_edge = train_edge.drop_duplicates()

                test_edge2 = test_edge.copy()
                test_edge2.columns = ['v2', 'v1', 't2', 't1']
                test_edge = pd.concat((test_edge, test_edge2), axis=0, sort=False)
                test_edge = test_edge.drop_duplicates()

            result = evaluate(embedding, train_edge, test_edge, type1, type2, node_type, args.vector_f)
            print('Evaluate link prediction (Source type %s - Target type %s) Result: %.4f' % (type1, type2, result))

            result = evaluate(embedding, train_edge, test_edge, type2, type1, node_type, args.vector_f)
            print('Evaluate link prediction (Source type %s - Target type %s) Result: %.4f' % (type2, type1, result))
            
            


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='data')
    parser.add_argument('--dataset', type=str, default='dblp', choices=['douban_movie', 'aminer', 'blog', 'dblp', 'yelp'])
    parser.add_argument('--embedding', type=str, required=True)
    parser.add_argument('--vector_f', type=str, default='hadamard', choices=['hadamard', 'average', 'minus', 'abs_minus'])
    parser.add_argument('--result', type=str, default='')
    args = parser.parse_args()

    main(args)
