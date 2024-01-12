import networkx as nx
from tqdm import tqdm
import random
import numpy as np
from gensim.models import word2vec


def sampleByWeight(G, v, t, p, q):
    nbrs = list(G.neighbors(v))
    if len(nbrs) == 0:
        return False
    weights = [1] * len(nbrs)
    for i, x in enumerate(nbrs):
        if t == x:
            weights[i] = 1 / p
        elif not G.has_edge(t, x):
            weights[i] = 1 / q
    return random.choices(nbrs, weights=weights, k=1)[0]


def walkOneTime(G, start_node, walk_length, p, q):
    walk = [start_node]  # 起始节点
    for _ in range(walk_length):
        cur_node = walk[-1]
        nbrs = list(G.neighbors(cur_node))
        if len(nbrs) > 0:
            if len(walk) == 1:
                walk.append(random.choice(nbrs))
            else:
                prev = walk[-2]
                v = sampleByWeight(G, cur_node, prev, p, q)
                if not v:
                    break
                walk.append(v)
        else:
            break
    return walk


def getNode2vecWalkSeqs(G, p, q, walk_length, num_walks):
    """get sequences

    Args:
        G (_Graph_): _Graph_
        walk_length (_int_): _每个序列的长度_
        num_walks (_int_): _序列的个数_
    """
    seqs = []
    for _ in tqdm(range(num_walks)):
        start_node = np.random.choice(G.nodes)
        w = walkOneTime(G, start_node, walk_length, p, q)
        seqs.append(w)
    return seqs


def node2vecWalk(G, p, q, dimensions=10, walk_length=80, num_walks=10, min_count=3):
    seqs = getNode2vecWalkSeqs(G, p, q, walk_length=walk_length, num_walks=num_walks)
    # print(seqs)
    model = word2vec.Word2Vec(seqs, vector_size=dimensions, min_count=min_count)
    return model


g = nx.fast_gnp_random_graph(n=100, p=0.5, directed=True)
p = 0.5
q = 1.5
model = node2vecWalk(g, p, q, dimensions=10, walk_length=20, num_walks=100, min_count=3)
print(model.wv.most_similar(2, topn=5))
