import os
import pickle
import functools
import multiprocessing
from collections import deque, defaultdict

import tqdm
import numpy as np
import pandas as pd
import scipy.sparse as sp

import torch

def pca(X, k=2):
    X = X - torch.mean(X, 0).expand_as(X)
    U, S, V = torch.svd(X.t())
    return torch.mm(X, U[:, :k])


def get_type(idx: int, type_interval_dict: dict):
    for k, (idx_min, idx_max) in type_interval_dict.items():
        if idx_min <= idx and idx <= idx_max:
            return k
    raise Exception("Type unmatch")


def load_data(args):
    root = args.root
    if args.dataset == 'dblp':
        author_df = pd.read_csv(os.path.join(root, 'dblp', 'author.txt'), 
                sep='\t', header=None, names=['index', 'A', 'L'])
        paper_df = pd.read_csv(os.path.join(root, 'dblp', 'paper.txt'), 
                sep='\t', header=None, names=['index', 'P', 'L'])
        topic_df = pd.read_csv(os.path.join(root, 'dblp', 'topic.txt'), 
                sep='\t', header=None, names=['index', 'T'])
        venue_df = pd.read_csv(os.path.join(root, 'dblp', 'venue.txt'), 
                sep='\t', header=None, names=['index', 'V', 'L'])

        write_df = pd.read_csv(os.path.join(root, 'dblp', 'write.txt'), 
                sep='\t', header=None, names=['A', 'P'])
        publish_df = pd.read_csv(os.path.join(root, 'dblp', 'publish.txt'), 
                sep='\t', header=None, names=['V', 'P'])
        mention_df = pd.read_csv(os.path.join(root, 'dblp', 'mention.txt'), 
                sep='\t', header=None, names=['P', 'T'])
        cite_df = pd.read_csv(os.path.join(root, 'dblp', 'cite.txt'), 
                sep='\t', header=None, names=['P1', 'P2'])
        node_df_dict = {'A': author_df, 'P': paper_df, 'T': topic_df, 'V': venue_df}
        edge_df_dict = {'AP': write_df, 'VP': publish_df, 'TP': mention_df, 'PP': cite_df}
    else:
        node_df_dict = {}
        edge_df_dict = {}
        node_df_dict['U'] = pd.read_csv(os.path.join(root, 'yelp', 'node', 'U.csv'), sep='\t')
        node_df_dict['B'] = pd.read_csv(os.path.join(root, 'yelp', 'node', 'B.csv'), sep='\t')
        node_df_dict['R'] = pd.read_csv(os.path.join(root, 'yelp', 'node', 'R.csv'), sep='\t')
        node_df_dict['W'] = pd.read_csv(os.path.join(root, 'yelp', 'node', 'W.csv'), sep='\t', keep_default_na=False)

        edge_df_dict['RU'] = pd.read_csv(os.path.join(root, 'yelp', 'edge', 'RU.csv'), sep='\t')
        edge_df_dict['RB'] = pd.read_csv(os.path.join(root, 'yelp', 'edge', 'RB.csv'), sep='\t')
        edge_df_dict['RW'] = pd.read_csv(os.path.join(root, 'yelp', 'edge', 'RW.csv'), sep='\t')
        edge_df_dict['UU'] = pd.read_csv(os.path.join(root, 'yelp', 'edge', 'UU.csv'), sep='\t')
    return node_df_dict, edge_df_dict


def get_preprocessed_data(args, type_split=False):
    fname = 'adj'
    fname = fname+'_type' if type_split else fname
    fname = fname+'.pickle'
    fname = os.path.join('preprocess', args.dataset, fname)

    # If exist, load data
    if os.path.exists(fname):
        with open(fname, 'rb') as f:
            return pickle.load(f)
    assert False
    result = dict()

    # Get node num
    node_df_dict, edge_df_dict = load_data(args)
    node_num = sum([len(v) for v in node_df_dict.values()])
    result['node_num'] = node_num

    # Type interval information
    type_interval = {k: (v['index'].min(), v['index'].max()) for k, v in node_df_dict.items()}
    type_size = {k: max_ - min_ + 1 for k, (min_ , max_) in type_interval.items()}
    result['type_interval'] = type_interval
    result['type_size'] = type_size

    # For visualization, get class idx for each type of node
    class_dict = {}
    for k, v in node_df_dict.items():
        if 'L' in v.columns:
            label_dict = {x: i for i, x in enumerate(v['L'].unique())}
            class_dict[k] = v['L'].apply(label_dict.get).values
    result['class'] = class_dict

    # Indice information
    adj_indice = create_adjacency_list_per_type(edge_df_dict)

    # Get degree distribution
    degree_dist = np.zeros((node_num,))
    for i in range(node_num):
        degree_dist[i] = sum([len(x) for x in adj_indice[i].values()])
    degree_dist = degree_dist / degree_dist.sum()
    result['degree'] = degree_dist

    # If type_split is false, union
    if type_split == False:
        adj_indice = {k: set().union(*v.values()) for k, v in adj_indice.items()}
    result['adj_indice'] = adj_indice

    # Save data
    os.makedirs(os.path.join('preprocess', args.dataset), exist_ok=True)
    with open(fname, 'wb') as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)

    return result


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


def create_adjacency_list_per_type(edge_df_dict):
    graph = defaultdict(lambda : defaultdict(set))
    for k, edge_df in edge_df_dict.items():
        if k[0] == k[1]:
            t0, t1 = k[0], k[1]
            l0, l1 = ['%s1'%k[0], '%s2'%k[0]]
        else:
            t0, t1 = k[0], k[1]
            l0, l1 = k[0], k[1]

        for _, row in tqdm.tqdm(edge_df.iterrows(), total=len(edge_df), ascii=True):
            graph[row[l0]][t1].add(row[l1])
            graph[row[l1]][t0].add(row[l0])
    return dict(graph)


class TypeChecker(object):
    def __init__(self, type_interval):
        self.type_interval = type_interval

    def __call__(self, v):
        for k, (min_idx, max_idx) in self.type_interval.items():
            if min_idx <= v and v <= max_idx:
                return k


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


def just_walk(v, adj_dict, l, typechecker, alpha, que_size):
    walk = [v] * l
    same_domain_count = 0
    que = deque()
    que.append(typechecker(walk[0]))
    for idx in range(1, l):
        cur_type = typechecker(walk[idx-1])
        possible_type = set(adj_dict[walk[idx-1]].keys())
        same_domain_count += 1
        if cur_type not in possible_type:
            stay = 0
        elif set(cur_type) == possible_type:
            stay = 1
        else:
            stay = 1 if np.random.rand(1) < np.power(alpha, same_domain_count) else 0
        if stay:
            walk[idx] = np.random.choice(list(adj_dict[walk[idx-1]][cur_type]), 1)[0]
        else:
            if len(possible_type-set(que)) == 0:
                target_domain = np.random.choice(list(possible_type), 1)[0]
            else:
                target_domain = np.random.choice(list(possible_type-set(que)), 1)[0]
            if len(que) == que_size:
                que.popleft()
            que.append(target_domain)
            walk[idx] = np.random.choice(list(adj_dict[walk[idx-1]][target_domain]), 1)[0]
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

