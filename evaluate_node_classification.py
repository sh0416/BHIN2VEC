import os
import csv
import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import f1_score
import tqdm

from utils import load_data

def classify(X, y, train_size, trial=5):
    micro_f1 = 0
    macro_f1 = 0
    for i in range(trial):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-train_size, stratify=y)

        classifier = LogisticRegression(solver='liblinear', class_weight='balanced', multi_class='auto', max_iter=1000).fit(X_train, y_train)
        #model = SelectFromModel(classifier, prefit=True)
        #X_train = model.transform(X_train)
        #classifier = LogisticRegression(solver='lbfgs', class_weight='balanced', multi_class='auto', max_iter=1000).fit(X_train, y_train)

        #X_test = model.transform(X_test)
        y_pred = classifier.predict(X_test)

        micro_f1 += f1_score(y_test, y_pred, average='micro')
        macro_f1 += f1_score(y_test, y_pred, average='macro')

    return micro_f1/trial, macro_f1/trial


def main(args):
    # Load data
    node_type, edge_df, test_node_df, _ = load_data(args)
    embedding = np.load(args.embedding)

    for k in test_node_df.keys():
        idx = test_node_df[k][k].values
        for c in test_node_df[k].columns:
            print(c, k)
            if c == k:
                continue

            X = embedding[idx]
            y = test_node_df[k][c].values

            # Report value
            for i in range(1, 10):
                train_size = i / 10
                micro_f1, macro_f1 = classify(X, y, train_size, 10)
                print(k, c, train_size, micro_f1, macro_f1)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='data')
    parser.add_argument('--dataset', type=str, default='dblp', choices=['douban_movie', 'aminer', 'dblp', 'yelp'])
    parser.add_argument('--embedding', type=str, default='output')
    args = parser.parse_args()

    main(args)
