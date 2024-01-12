import networkx as nx
from tqdm import tqdm
import numpy as np
from gensim.models import word2vec


def walkOneTime(G, start_node, walk_length):
    """单独生成一条序列

    Args:
        G (_type_): _图_
        start_node (_node_): _起始节点_
        walk_length (_int_): _序列长度_
    """
    walk = [str(start_node)]  # 起始节点
    for _ in range(walk_length):
        current_node = int(walk[-1])
        successors = list(G.successors(current_node))
        if len(successors) > 0:
            next_node = np.random.choice(successors, 1)
            walk.extend(list(map(str, next_node)))
        else:
            break
    return walk


def getDeepWalkSeqs(G, walk_length, num_walks):
    """get sequences

    Args:
        G (_Graph_): _Graph_
        walk_length (_int_): _每个序列的长度_
        num_walks (_int_): _序列的个数_
    """
    seqs = []
    for _ in tqdm(range(num_walks)):
        start_node = np.random.choice(G.nodes)
        w = walkOneTime(G, start_node, walk_length)
        seqs.append(w)
    return seqs


def deepwalk(G, dimensions=10, walk_length=80, num_walks=10, min_count=3):
    seqs = getDeepWalkSeqs(G, walk_length=walk_length, num_walks=num_walks)
    # print(seqs)
    model = word2vec.Word2Vec(seqs, vector_size=dimensions, min_count=min_count)
    return model


g = nx.fast_gnp_random_graph(n=100, p=0.5, directed=True)
model = deepwalk(g, dimensions=10, walk_length=20, num_walks=100, min_count=3)
print(model.wv.most_similar("2", topn=5))
