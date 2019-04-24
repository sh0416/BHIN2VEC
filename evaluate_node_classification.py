import os
import csv
import logging
import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import tqdm

from utils import load_data, add_argument, get_name


def classify(X_train, X_test, y_train, y_test, train_size, trial=5):
    micro_f1 = 0
    macro_f1 = 0
    X_train, _, y_train, _ = train_test_split(X_train, y_train, test_size=1-train_size, stratify=y_train)
    for i in range(trial):

        classifier = LogisticRegression(solver='liblinear', class_weight='balanced', multi_class='auto', max_iter=1000).fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        micro_f1 += f1_score(y_test, y_pred, average='micro')
        macro_f1 += f1_score(y_test, y_pred, average='macro')

    return micro_f1/trial, macro_f1/trial


def main(args):
    os.makedirs('node_classification_result', exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.FileHandler(os.path.join('node_classification_result', get_name(args)+'.log')))

    # Load data
    node_type, edge_df, node_df, _ = load_data(args)
    embedding = np.load(os.path.join('output', get_name(args)+'.npy'))

    for t, (min_value, max_value) in node_type.items():
        # 타겟 타입에 해당하는 노드만 걸러낸 뒤,
        target_node = node_df[(min_value <= node_df['v']) & (node_df['v'] <= max_value)]
        if len(target_node) == 0:
            continue

        X = embedding[target_node['v']]
        y = target_node['l'].values

        # 테스트 셋과 트레이닝 셋 스플릿
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

        # Report value
        for i in range(1, 10):
            train_size = i / 10
            micro_f1, macro_f1 = classify(X_train, X_test, y_train, y_test, train_size, 10)
            logger.info('Type %s, train_size %.1f, micro f1: %.4f, macro f1: %.4f' % (t, train_size, micro_f1, macro_f1))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    add_argument(parser)
    args = parser.parse_args()

    main(args)
