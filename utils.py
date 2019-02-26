import os
import random
import pickle
import functools
import multiprocessing
from collections import deque, defaultdict

import tqdm
import numpy as np
import pandas as pd
import scipy.sparse as sp

import torch


def preprocess(node_df_dict, edge_df_dict):
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


def load_data(args):
    root = args.root
    node_df_dict = {}
    edge_df_dict = {}
    if args.dataset == 'dblp':
        node_df_dict['A'] = pd.read_csv(os.path.join(root, 'dblp', 'author.txt'), sep='\t', header=None, names=['index', 'A', 'L'])
        node_df_dict['P'] = pd.read_csv(os.path.join(root, 'dblp', 'paper.txt'), sep='\t', header=None, names=['index', 'P', 'L'])
        node_df_dict['T'] = pd.read_csv(os.path.join(root, 'dblp', 'topic.txt'), sep='\t', header=None, names=['index', 'T'])
        node_df_dict['V'] = pd.read_csv(os.path.join(root, 'dblp', 'venue.txt'), sep='\t', header=None, names=['index', 'V', 'L'])
        edge_df_dict['AP'] = pd.read_csv(os.path.join(root, 'dblp', 'write.txt'), sep='\t', header=None)
        edge_df_dict['VP'] = pd.read_csv(os.path.join(root, 'dblp', 'publish.txt'), sep='\t', header=None)
        edge_df_dict['PT'] = pd.read_csv(os.path.join(root, 'dblp', 'mention.txt'), sep='\t', header=None)
        edge_df_dict['PP'] = pd.read_csv(os.path.join(root, 'dblp', 'cite.txt'), sep='\t', header=None)
    elif args.dataset == 'yelp':
        node_df_dict['U'] = pd.read_csv(os.path.join(root, 'yelp', 'node', 'U.csv'), sep='\t')
        node_df_dict['B'] = pd.read_csv(os.path.join(root, 'yelp', 'node', 'B.csv'), sep='\t')
        node_df_dict['R'] = pd.read_csv(os.path.join(root, 'yelp', 'node', 'R.csv'), sep='\t')
        node_df_dict['W'] = pd.read_csv(os.path.join(root, 'yelp', 'node', 'W.csv'), sep='\t', keep_default_na=False)
        edge_df_dict['RU'] = pd.read_csv(os.path.join(root, 'yelp', 'edge', 'RU.csv'), sep='\t')
        edge_df_dict['RB'] = pd.read_csv(os.path.join(root, 'yelp', 'edge', 'RB.csv'), sep='\t')
        edge_df_dict['RW'] = pd.read_csv(os.path.join(root, 'yelp', 'edge', 'RW.csv'), sep='\t')
        edge_df_dict['UU'] = pd.read_csv(os.path.join(root, 'yelp', 'edge', 'UU.csv'), sep='\t')
    elif args.dataset == 'blog':
        node_df_dict['U'] = pd.read_csv(os.path.join(root, 'blog-catalog', 'nodes.csv'), header=None, names=['index'])
        node_df_dict['G'] = pd.read_csv(os.path.join(root, 'blog-catalog', 'groups.csv'), header=None, names=['index'])
        edge_df_dict['UU'] = pd.read_csv(os.path.join(root, 'blog-catalog', 'edges.csv'), sep=',', header=None)
        edge_df_dict['UG'] = pd.read_csv(os.path.join(root, 'blog-catalog', 'group-edges.csv'), sep=',', header=None)
        node_df, edge_df, test_edge_df = preprocess(node_df_dict, edge_df_dict)
    elif args.dataset == 'douban_movie':
        with open(os.path.join(root, 'douban_movie', 'node_type.pickle'), 'rb') as f:
            node_type = pickle.load(f)
        edge_df = pd.read_csv(os.path.join(root, 'douban_movie', 'edge.csv'), sep='\t', usecols=['v1', 'v2', 't1', 't2'])
        test_edge_df = {'UM': pd.read_csv(os.path.join(root, 'douban_movie', 'link_prediction', 'test_user_movie.csv'),
                                          sep='\t',
                                          usecols=['v1', 'v2', 't1', 't2']),
                        'UU': pd.read_csv(os.path.join(root, 'douban_movie', 'link_prediction', 'test_user_user.csv'),
                                          sep='\t',
                                          usecols=['v1', 'v2', 't1', 't2']),
                        'MA': pd.read_csv(os.path.join(root, 'douban_movie', 'link_prediction', 'test_movie_actor.csv'),
                                          sep='\t',
                                          usecols=['v1', 'v2', 't1', 't2']),
                        'MD': pd.read_csv(os.path.join(root, 'douban_movie', 'link_prediction', 'test_movie_director.csv'),
                                          sep='\t',
                                          usecols=['v1', 'v2', 't1', 't2'])}
        test_node_df = {'M': pd.read_csv(os.path.join(root, 'douban_movie', 'node_classification', 'movie_label.csv'), sep='\t')}

    return node_type, edge_df, test_node_df, test_edge_df


def convert_defaultdict_to_dict(x):
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


def create_adjacency_matrix(adj_list: dict, shape: (int, int)):
    row = [k for k, v in adj_list.items() for _ in v]
    col = [x for _, v in adj_list.items() for x in v]

    adj_matrix = sp.coo_matrix((np.ones(len(row)), (row, col)), shape=shape).tocsr()
    return adj_matrix


def deepwalk(v, adj, l):
    walk = [v] * l
    for idx in range(1, l):
        adj_list = adj[walk[idx-1]]
        walk[idx] = np.random.choice(list(adj_list), 1)[0]
    return walk


def metapath_walk(v, adj_dict, metapath, l, typechecker):
    walk = [v] * l
    type_idx = metapath.index(typechecker(v)) + 1
    for idx in range(1, l):
        next_type = metapath[type_idx%len(metapath)]
        type_idx += 1
        assert next_type in adj_dict[walk[idx-1]], '%s is not in %d' % (next_type, walk[idx-1])
        adj_list = adj_dict[walk[idx-1]][next_type]
        walk[idx] = np.random.choice(list(adj_list), 1)[0]
    return walk


def balanced_walk(v, adj, type_interval, type_str_dict, type_str_inverse_dict, type_, l):
    walk = [v] * l
    from_que = deque()
    to_prob = np.asarray([0] * 4)
    for i in range(1, l):
        cur_type = type_str_dict[get_type(walk[i-1], type_interval)]
        possible_type = list(map(type_str_dict.get, adj[walk[i-1]].keys()))

        # From que update
        from_que.append((cur_type, np.copy(type_[cur_type])))

        # Calculate probability
        prob = np.asarray([0] * 4, dtype=np.float32)
        for _, p in from_que:
            prob += p
        prob += to_prob

        nxt_type = np.random.choice(possible_type, 1, p=prob[possible_type]/prob[possible_type].sum())[0]
        walk[i] = np.random.choice(list(adj[walk[i-1]][type_str_inverse_dict[nxt_type]]), 1)[0]

        # To prob update
        to_prob[nxt_type] = 0
        for return_type, p in from_que:
            to_prob[return_type] += p[nxt_type]
            p[nxt_type] = 0

        while len(from_que) > 0:
            if np.all(from_que[0]==0):
                from_que.popleft()
            else:
                break
    return walk


class BalancedWalkFactory(object):
    def __init__(self,
                 possible_type_mask,
                 adjacent_node_num_per_type,
                 start_idx,
                 pad_packed_adj,
                 type_attention,
                 type_interval_dict,
                 type_str_dict):
        self.possible_type_mask = possible_type_mask
        self.adjacent_node_num_per_type = adjacent_node_num_per_type
        self.start_idx = start_idx
        self.pad_packed_adj = pad_packed_adj
        self.type_attention = type_attention
        self.type_interval_dict = type_interval_dict
        self.type_str_dict = type_str_dict

    def to(self, device):
        self.possible_type_mask = self.possible_type_mask.to(device)
        self.adjacent_node_num_per_type = self.adjacent_node_num_per_type.to(device)
        self.start_idx = self.start_idx.to(device)
        self.pad_packed_adj = self.pad_packed_adj.to(device)
        self.type_attention = self.type_attention.to(device)
        return self

    def _get(self, node, idx, type_):
        return self.pad_packed_adj[self.start_idx[node, type_]+idx, type_]

    def __call__(self, v, l):
        batch_size = v.shape[0]
        walk = v.new_zeros(l, batch_size, dtype=torch.long)
        walk[0, :] = v

        history = v.new_zeros(l, batch_size, 4, dtype=torch.float)
        return_history = v.new_full((l, batch_size), fill_value=-1, dtype=torch.long)
        to_prob = v.new_zeros(batch_size, 4, dtype=torch.float)

        cur_type = v.new_tensor([self.type_str_dict[get_type(x, self.type_interval_dict)] for x in walk[0, :]], dtype=torch.long)

        for i in range(1, l):
            history[i-1, :, :] = self.type_attention[cur_type]
            return_history[i-1, :] = cur_type
            prob = history.sum(dim=0)
            assert not torch.any(prob.sum(dim=1)==0)
            prob += to_prob
            assert not torch.any(prob.sum(dim=1)==0)
            prob *= self.possible_type_mask[walk[i-1,:]]

            assert not torch.any(prob<0)
            assert not torch.any(prob.sum(dim=1)==0)
            nxt_type = torch.multinomial(prob, 1)
            idx = torch.rand(batch_size).cuda()
            size = torch.gather(self.adjacent_node_num_per_type[walk[i-1,:]], 1, nxt_type).squeeze()

            idx = torch.floor(idx * size.float()).long()
            walk[i, :] = self._get(walk[i-1, :], idx, nxt_type.squeeze())
            cur_type = nxt_type.squeeze()

            to_prob.scatter_(dim=1, index=nxt_type, src=torch.tensor(0))
            for j in range(i):
                to_prob.scatter_add_(dim=1, index=return_history[[j], :].t(), 
                                 src=torch.gather(history[j, :, :], dim=1, index=nxt_type))
                history[j, :, :].scatter_(dim=1, index=nxt_type, src=torch.tensor(0))
        return walk

def node2vec_preprocess_f(v, adj, adj_d, p, q, result):
    g = sp.lil_matrix((1, adj.shape[1]*adj.shape[1]), dtype=np.float32)
    adj_list = adj[v].indices
    for t in adj_list:
        for x in adj_list:
            val = adj_d[t, x]
            if val & 0x1 == 1:
                g[0, t*adj.shape[1]+x] = 1/p
            elif val & 0x2 == 2:
                g[0, t*adj.shape[1]+x] = 1
            else:
                g[0, t*adj.shape[1]+x] = 1/q
    g = g.tocsr()
    result[v] = g


def node2vec_preprocess(adj, p, q):
    adj_0 = sp.eye(adj.shape[0])
    adj_1 = adj.copy()
    adj_1[adj_1!=0] = 2
    adj_2 = adj.dot(adj)
    adj_2[adj_2!=0] = 4
    adj_d = adj_0 + adj_1 + adj_2
    adj_d = adj_d.astype(np.int)

    result = [None] * adj.shape[0]
    node2vec_preprocess_f_one = functools.partial(node2vec_preprocess_f, 
            adj=adj, adj_d=adj_d, p=p, q=q, result=result)
    for i in tqdm.tqdm(range(adj.shape[0])):
        node2vec_preprocess_f_one(i)
    # If you want multiprocessing,
    #with multiprocessing.Pool() as pool:
    #    for _ in tqdm.tqdm(pool.imap_unordered(preprocess_adj_f_one, range(adj.shape[0])), total=adj.shape[0]):
    #        pass
    #    pool.close()
    #    pool.join()
    g = sp.vstack(result)
    return g


def node2vec_walk_f(v, adj, pre_adj, l, p, q):
    walk = [v] * l
    for idx in range(1, l):
        adj_list = adj[walk[idx-1]].indices
        if idx == 1:
            walk[idx] = np.random.choice(adj_list, 1)[0]
        else:
            t = walk[-2]
            x = walk[-1]
            prob = pre_adj[v, t*adj.shape[0]:(t+1)*adj.shape[0]].data
            walk[idx] = np.random.choice(adj_list, 1, p=prob/prob.sum())[0]
    return walk


def node2vec_walk(adj, p, q, r, l):
    # For large graph, preprocessing is memory killer.
    pre_adj = preprocess_adj(adj, p, q)
    sp.save_npz('pre_adj.npz', pre_adj)
    walks = [None] * (r*adj.shape[0])
    node2vec_walk_f_one = functools.partial(node2vec_walk_f, adj=adj, pre_adj=pre_adj, l=l, p=p, q=q)

    cnt = 0
    for _ in range(r):
        for v in range(adj.shape[0]):
            walks[cnt] = node2vec_walk_fun_one(v)
            cnt += 1
        # If you want multiprocessing,
        #with multiprocessing.Pool() as pool:
        #    for walk in tqdm.tqdm(pool.imap_unordered(node2vec_walk_fun_one, range(adj.shape[0])), total=adj.shape[0]):
        #        walks[cnt] = walk
        #        cnt += 1
    return walks


def calculate_entropy(arr):
    arr = arr/arr.sum()
    entropy = -(arr*np.log2(arr)).mean()
    assert 0 <= entropy and entropy < 1
    return entropy

def create_entropy_vector(adj):
    return np.asarray([calculate_entropy(np.asarray(adj[i].data)) for i in range(adj.shape[0])])

def create_confidence_vector():
    # Calculate confidence
    confidence_w = 0.5
    confidence_adj = total_adj
    confidence_adj = confidence_w*confidence_adj + (1-confidence_w)*total_adj.dot(confidence_adj)
    confidence_adj = confidence_w*confidence_adj + (1-confidence_w)*total_adj.dot(confidence_adj)

    confidence_adj = confidence_adj.tocsc()
    confidence_author_adj = confidence_adj[:, author_start:author_end]
    confidence_paper_adj = confidence_adj[:, paper_start:paper_end]
    confidence_topic_adj = confidence_adj[:, topic_start:topic_end]
    confidence_venue_adj = confidence_adj[:, venue_start:venue_end]

    sp.save_npz('confidence_author_adj.npz', confidence_author_adj)
    sp.save_npz('confidence_paper_adj.npz', confidence_paper_adj)
    sp.save_npz('confidence_topic_adj.npz', confidence_topic_adj)
    sp.save_npz('confidence_venue_adj.npz', confidence_venue_adj)
    print("Complete confidence adjacency matrix")

    # Calculate confidence_vector
    confidence_author_adj = sp.load_npz('confidence_author_adj.npz').tocsr()
    confidence_author_entropy = create_entropy_vector(confidence_author_adj)

    confidence_paper_adj = sp.load_npz('confidence_paper_adj.npz').tocsr()
    confidence_paper_entropy = create_entropy_vector(confidence_paper_adj)

    confidence_topic_adj = sp.load_npz('confidence_topic_adj.npz').tocsr()
    confidence_topic_entropy = create_entropy_vector(confidence_topic_adj)

    confidence_venue_adj = sp.load_npz('confidence_venue_adj.npz').tocsr()
    confidence_venue_entropy = create_entropy_vector(confidence_venue_adj)
        
    confidence_vector = np.vstack((confidence_author_entropy,
                                   confidence_paper_entropy,
                                   confidence_topic_entropy,
                                   confidence_venue_entropy)).T
    np.save('confidence_vector.npy', confidence_vector)
    print("Complete confidence vector")

def create_guidende_vector():
    # Create guidence vector
    guidence_vector = np.zeros((node_num, 4))
    start = 0
    for label, interval in enumerate([len(author_df), len(paper_df), len(topic_df), len(venue_df)]):
        guidence_vector[start:start+interval, label] = 1
        start = start + interval

    guidence_w = 0.5
    guidence_vector = guidence_w*guidence_vector + (1-guidence_w)*total_adj.dot(guidence_vector)
    guidence_vector = guidence_w*guidence_vector + (1-guidence_w)*total_adj.dot(guidence_vector)
    guidence_vector = guidence_w*guidence_vector + (1-guidence_w)*total_adj.dot(guidence_vector)

    np.save('guidence_vector.npy', guidence_vector)
    print("Complete guidence vector")

def pca(X, k=2):
    X = X - torch.mean(X, 0).expand_as(X)
    U, S, V = torch.svd(X.t())
    return torch.mm(X, U[:, :k])


def get_type(idx, type_interval_dict):
    for k, (idx_min, idx_max) in type_interval_dict.items():
        if idx_min <= idx and idx <= idx_max:
            return k
    raise Exception("Type unmatch")


