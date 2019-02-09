import os
import csv
import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score

from utils import load_data

def classify(embedding, df, train_size):
    count_df = df.groupby('L').count()
    exclude_label = count_df.nsmallest(8, 'A')
    #exclude_label = count_df.nsmallest(3, 'P')

    X_list, y_list = [], []
    cnt = 0
    min_size = 101000
    for label in count_df.index:
        if label in exclude_label.index:
            continue
        index = df[df['L']==label]['index'].values
        X_list.append(embedding[index, :])
        y_list.append(np.full(len(index), cnt))
        min_size = min([min_size, len(index)])
        cnt += 1

    micro_f1 = 0
    macro_f1 = 0
    for i in range(20):
        X = np.concatenate(X_list, axis=0)
        y = np.concatenate(y_list, axis=0)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-train_size, stratify=y)

        classifier = LogisticRegression(solver='liblinear', class_weight='balanced', multi_class='ovr').fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        micro_f1 += f1_score(y_test, y_pred, average='micro')
        macro_f1 += f1_score(y_test, y_pred, average='macro')

    return micro_f1/20, macro_f1/20

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='data')
    parser.add_argument('--dataset', type=str, default='dblp', choices=['dblp', 'yelp'])
    parser.add_argument('--embedding', type=str, default=os.path.join('output', 'deepwalk.npy'))
    parser.add_argument('--result', type=str, default=os.path.join('output', 'deepwalk_result.csv'))
    parser.add_argument('--method', type=str, default='classification', choices=['classification', 'clustering'])
    args = parser.parse_args()

    embedding = np.load(args.embedding)

    node_df_dict, _ = load_data(args)
    print(node_df_dict['V']['L'].unique())
    print(len(node_df_dict['V']['L'].unique()))
    assert False

    with open(args.result, 'w') as f:
        fieldnames = ['train_size', 'micro_f1', 'macro_f1']
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        writer.writeheader()
        for i in range(1, 10):
            train_size = i / 10
            #micro_f1, macro_f1 = classify(embedding, node_df_dict['A'], train_size)
            #micro_f1, macro_f1 = classify(embedding, node_df_dict['P'], train_size)
            micro_f1, macro_f1 = classify(embedding, node_df_dict['B'], train_size)
            writer.writerow({'train_size': train_size, 'micro_f1': micro_f1, 'macro_f1': macro_f1})
