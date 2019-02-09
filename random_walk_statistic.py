import os
import pickle
import logging
from collections import deque

import numpy as np
import scipy.sparse as sp

from utils import load_data, TypeChecker, deepwalk

def normalize_row(matrix: sp.coo_matrix):
    rowsum = matrix.sum(axis=1).A.ravel()
    rowsum[rowsum==0] = 1
    d_inverse = sp.diags(1/rowsum)
    matrix = d_inverse.dot(matrix)
    matrix.eliminate_zeros()
    return matrix

def create_adjacency_matrix(adj_list: dict, shape: (int, int)):
    row = [k for k, v in adj_list.items() for _ in v]
    col = [x for _, v in adj_list.items() for x in v]

    adj_matrix = sp.coo_matrix((np.ones(len(row)), (row, col)), shape=shape).tocsr()
    return adj_matrix

def class_normalized_random_walk(v: int, 
                                 adj_matrix_per_type: dict, 
                                 type_interval_dict: dict,
                                 type_vector: np.ndarray,
                                 outque_distance: int,
                                 l: int):
    type_num = len(adj_matrix_per_type.keys())
    # Initialize data structure
    inque_distance_list = np.asarray([0] * type_num)
    que = deque()

    # Synthesize random walk
    walk = [v] * l
    for i in range(1, l):
        # Mask value which is not inside queue
        prob = np.where(inque_distance_list==0, outque_distance, inque_distance_list)

        # Mask value which is not connected
        mask = np.zeros((type_num,))
        for type_idx, adj_matrix in adj_matrix_per_type.items():
            mask[type_idx] = 0 if adj_matrix[walk[i-1]].nnz == 0 else 1
        prob = prob * mask

        # Normalize probability
        prob = prob / prob.sum()

        # Sample next type
        next_type = np.random.choice(type_num, 1, p=prob)[0]

        # Update queue
        que.append(next_type)
        inque_distance_list += 1
        inque_distance_list[next_type] = 1
        size = inque_distance_list.max()
        while len(que) > size:
            que.popleft()

        # Using type vector to see multi-hop neighbor
        candidate_node = type_interval_dict[next_type][0] + adj_matrix_per_type[next_type][walk[i-1]].nonzero()[1]
        prob = type_vector[candidate_node, :].dot(inque_distance_list)
        walk[i] = np.random.choice(candidate_node, 1, p=prob/prob.sum())[0]
    return walk
    

def random_walk_2(v: int, indice: list, val: list, l: int):
    walk = [v] * l
    for i in range(1, l):
        walk[i] = np.random.choice(indice[walk[i-1]], 1, p=val[walk[i-1]])[0]
    return walk
            
        
if __name__=='__main__':

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler()
    logger.addHandler(handler)

    node_df_dict, edge_df_dict = load_data('preprocess')

    for k, v in edge_df_dict.items():
        print(k, len(v))

    # Type information
    type_num = len(node_df_dict)
    type_dict = {x: i for i, x in enumerate(node_df_dict.keys())}
    type_interval_dict = {type_dict[k]: (v['index'].min(), v['index'].max()) for k, v in node_df_dict.items()}
    logger.info("type number: " + str(type_num))
    logger.info("type index mapping: " + str(type_dict))
    logger.info("type interval information: " + str(type_interval_dict))

    # Node information
    node_num = sum([len(df) for df in node_df_dict.values()])
    logger.info("node number: " + str(node_num))
    logger.info("node number per type:")
    for k, df in node_df_dict.items():
        logger.info(k + ": " + str(len(df)))


    # Edge information
    with open('adj_dict.pickle', 'rb') as f:
        adj_dict = pickle.load(f)
        adj_list = {k: set().union(*v.values()) for k, v in adj_dict.items()}


    adj_matrix = create_adjacency_matrix(adj_list, (node_num, node_num))
    adj_matrix_per_type = {}
    for k, v in node_df_dict.items():
        idx_min, idx_max = v['index'].min(), v['index'].max()
        adj_matrix_per_type[type_dict[k]] = normalize_row(adj_matrix[:, idx_min:idx_max])
    adj_matrix = normalize_row(adj_matrix)

    for k in adj_matrix_per_type.keys():
        adj_matrix_per_type[k].eliminate_zeros()

    type_vector = np.zeros((node_num, 4))
    for i in range(node_num):
        type_vector[i, get_type(i, type_interval_dict)] = 1

    for _ in range(2):
        type_vector = 0.2 * adj_matrix.dot(type_vector) + 0.8 * type_vector

    from collections import Counter

    counter = Counter()
    for i in range(200):
        #walk = class_normalized_random_walk(1, adj_matrix_per_type, type_interval_dict, type_vector, 6, 80)
        #walk = deepwalk(1, adj_list, 80)
        walk = random_walk_2(1, adj_matrix_per_type, type_interval_dict, 80)
        walk_type = ''.join([str(get_type(x, type_interval_dict)) for x in walk])

        counter += Counter([str(get_type(x, type_interval_dict)) for x in walk])

    print(walk)
    print(walk_type)
    print(counter)

        
    """
    implicit_matrix = np.zeros((type_num, 11))

    node_dist = np.zeros((type_num,), dtype=np.int64)

    for i in range(node_num):
        walk = deepwalk(i, adj, 80)
        for j in range(len(walk)):
            node_type = key_map[typechecker(walk[j])]
            node_dist[node_type] += 1
            from_type = key_map[typechecker(walk[j])]
            for k in range(1, 11):
                to_type = key_map[typechecker(walk[j+k])]
                if from_type == to_type:
                    implicit_matrix[from_type, k] += 1

    print(implicit_matrix)
    print(node_dist)
    """

