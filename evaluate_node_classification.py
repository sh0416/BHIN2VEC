import os
import csv
import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import tqdm

from utils import get_preprocessed_data

def classify(X, y, train_size, trial=5):
    micro_f1 = 0
    macro_f1 = 0
    for i in range(trial):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-train_size, stratify=y)

        classifier = LogisticRegression(solver='liblinear', class_weight='balanced', multi_class='ovr').fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        micro_f1 += f1_score(y_test, y_pred, average='micro')
        macro_f1 += f1_score(y_test, y_pred, average='macro')

    return micro_f1/trial, macro_f1/trial


def report(embedding, label, parameter, writer):
    for i in range(1, 10):
        train_size = i / 10
        micro_f1, macro_f1 = classify(embedding, label, train_size, 20)
        writer.writerow(parameter+[train_size, micro_f1, macro_f1])

def main(args):
    # Load data
    data = get_preprocessed_data(args, type_split=True)

    filelist = [f for f in os.listdir(args.embedding_root) if os.path.isfile(os.path.join(args.embedding_root, f))]
    report_filename = args.dataset+'_'+args.target_type+'_classification_'+args.result+'.csv'
    report_filepath = os.path.join('result', report_filename)

    os.makedirs('result', exist_ok=True)
    with open(report_filepath, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['model', 'd', 'l', 'k', 'm', 'train_size', 'micro_f1', 'macro_f1'])

        for filename in tqdm.tqdm(filelist, total=len(filelist), ascii=True):
            embedding = np.load(os.path.join(args.embedding_root, filename))

            parameter = filename.split('.')[0].split('_')

            # Load target interval
            interval = data['type_interval'][args.target_type]

            # Get target embedding and label
            embedding = embedding[interval[0]:interval[1]+1]
            label = data['class'][args.target_type]

            """
            print((label==9).nonzero()[0][0])
            embedding = np.delete(embedding, 2088, axis=0)
            label = np.delete(label, 2088, axis=0)
            """

            # Report value
            report(embedding, label, parameter, writer)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='data')
    parser.add_argument('--dataset', type=str, default='dblp', choices=['dblp', 'yelp'])
    parser.add_argument('--target_type', type=str, default='A')
    parser.add_argument('--embedding_root', type=str, default='output')
    parser.add_argument('--result', type=str, default='')
    args = parser.parse_args()

    main(args)
