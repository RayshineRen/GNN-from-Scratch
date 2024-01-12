import networkx as nx

G = nx.read_edgelist(
    "/home/ray/code/python/GNN-from-Scratch/GraphEmbedding/data/wiki/Wiki_edgelist.txt",
    create_using=nx.DiGraph(),
    nodetype=None,
    data=[("weight", int)],
)
