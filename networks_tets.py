import matplotlib.pyplot as plt

from netgraph import Graph  # pip install netgraph OR conda install -c conda-forge netgraph

# right triangle
edge_length = {
    (0, 1): 0.8,
    (1, 2): 0.4,
    (2, 0): 0.9,
    (0, 3): 0.5,
}
edges = list(edge_length.keys())

fig, ax = plt.subplots()
Graph(edges, node_layout="geometric", node_layout_kwargs=dict(edge_length=edge_length), node_labels=True, ax=ax)
ax.set_aspect("equal")
plt.show()
