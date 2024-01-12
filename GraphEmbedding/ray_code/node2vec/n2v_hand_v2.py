import networkx as nx
from tqdm import tqdm
import random
import numpy as np
from gensim.models import word2vec
from alias_method_v2 import create_alias_table, alias_sample


def get_alias_edge(G, src, dst, p, q):
    unnormalized_probs = []
    for dst_nbr in sorted(G.neighbors(dst)):
        if dst_nbr == src:
            unnormalized_probs.append(G[dst][dst_nbr]["weight"] / p)
        elif G.has_edge(dst_nbr, src):
            unnormalized_probs.append(G[dst][dst_nbr]["weight"])
        else:
            unnormalized_probs.append(G[dst][dst_nbr]["weight"] / q)
    norm_const = sum(unnormalized_probs)
    K = len(unnormalized_probs)
    normalized_probs = [float(u_prob) * K / norm_const for u_prob in unnormalized_probs]
    return create_alias_table(normalized_probs)


def preprocess_trans_probs(G, p, q):
    alias_nodes = {}
    for node in G.nodes():
        unnormalized_probs = [
            G[node][nbr]["weight"] for nbr in sorted(G.neighbors(node))
        ]
        norm_const = sum(unnormalized_probs)
        K = len(unnormalized_probs)
        normalized_probs = [
            float(u_prob) * K / norm_const for u_prob in unnormalized_probs
        ]
        alias_nodes[node] = create_alias_table(normalized_probs)
    alias_edges = {}
    if G.is_directed():
        for edge in G.edges():
            alias_edges[edge] = get_alias_edge(G, edge[0], edge[1], p, q)
    else:
        for edge in G.edges():
            alias_edges[edge] = get_alias_edge(G, edge[0], edge[1], p, q)
            alias_edges[(edge[1], edge[0])] = get_alias_edge(G, edge[1], edge[0], p, q)
    return alias_nodes, alias_edges


def walkOneTime(G, alias_nodes, alias_edges, start_node, walk_length):
    walk = [start_node]  # 起始节点
    for _ in range(walk_length):
        cur = int(walk[-1])
        cur_nbrs = sorted(G.neighbors(cur))
        if len(cur_nbrs) > 0:
            if len(walk) == 1:
                walk.append(
                    cur_nbrs[alias_sample(alias_nodes[cur][0], alias_nodes[cur][1])]
                )
            else:
                prev = int(walk[-2])
                next_node = cur_nbrs[
                    alias_sample(
                        alias_edges[(prev, cur)][0], alias_edges[(prev, cur)][1]
                    )
                ]
                walk.append(next_node)
        else:
            break
    return walk


def getNode2vecWalkSeqs(G, alias_nodes, alias_edges, walk_length, num_walks):
    """get sequences

    Args:
        G (_Graph_): _Graph_
        walk_length (_int_): _每个序列的长度_
        num_walks (_int_): _序列的个数_
    """
    seqs = []
    for _ in tqdm(range(num_walks)):
        start_node = np.random.choice(G.nodes)
        w = walkOneTime(G, alias_nodes, alias_edges, start_node, walk_length)
        seqs.append(w)
    return seqs


def node2vecWalk(
    G,
    alias_nodes,
    alias_edges,
    dimensions=10,
    walk_length=80,
    num_walks=10,
    min_count=3,
):
    seqs = getNode2vecWalkSeqs(
        G, alias_nodes, alias_edges, walk_length=walk_length, num_walks=num_walks
    )
    # print(seqs)
    model = word2vec.Word2Vec(seqs, vector_size=dimensions, min_count=min_count)
    return model


p = 0.5
q = 1.5
g = nx.fast_gnp_random_graph(n=100, p=0.5, directed=True)
weights = {(u, v): random.random() for u, v in g.edges()}
nx.set_edge_attributes(g, values=weights, name="weight")
alias_nodes, alias_edges = preprocess_trans_probs(g, p, q)
model = node2vecWalk(
    g,
    alias_nodes,
    alias_edges,
    dimensions=10,
    walk_length=20,
    num_walks=100,
    min_count=3,
)
print(model.wv.most_similar(2, topn=5))
