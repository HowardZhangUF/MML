import networkx as nx
import matplotlib.pyplot as plt

# Positions of nodes relative to the 2D coordinates
node_positions = {
    0: (-1.5, 1),   # Node 0
    1: (0, 1),      # Node 1
    2: (2.1, 1),      # Node 2
    3: (-1.5, -1),  # Node 3
    4: (0, -1),     # Node 4
    5: (0.4, 0),    # Node 5
    6: (1.7, 0),    # Node 6
    7: (2.1, -1),     # Node 7
}


# Adjacency matrix indicating connections between the nodes
trans_matrix = [
    # 0    1    2    3    4    5    6    7
     [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # Node 0
     [0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0],  # Node 1
     [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Node 2
     [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Node 3
     [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],  # Node 4
     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],  # Node 5
     [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5],  # Node 6
     [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # Node 7
]

# Create a directed graph
G = nx.DiGraph()

# Add nodes with positions
for node, pos in node_positions.items():
    G.add_node(node, pos=pos)

# Add edges based on adjacency matrix
for i, row in enumerate(trans_matrix):
    for j, val in enumerate(row):
        if val > 0:
            G.add_edge(i, j, weight=val)

# Extract positions for drawing
pos = nx.get_node_attributes(G, 'pos')

# Draw the graph
plt.figure(figsize=(8, 6))
nx.draw(G, pos, with_labels=True, node_size=700, node_color='lightgrey', edge_color='black', arrows=True, arrowstyle='-|>', arrowsize=20)

# Draw edge labels (transition probabilities)
edge_labels = {(i, j): f'{val:.2f}' for i, row in enumerate(trans_matrix) for j, val in enumerate(row) if val > 0}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

occ_widths = [0.6, 0.6, 0.6, 0.6]
occ_centers = [[-1.5, 1], [-0.75, 0.], [1.1, 1.1], [1.7, 0.]]
# draw occlusions with red squares. Use centers and occ_widths to draw the squares
for ii, occ in enumerate(occ_centers):
    # use occ_widths to draw the squares. Only draw the border of the square
    # remember the the line would be at center +/- 0.5*occ_widths
    plt.plot([occ[0] - 0.5*occ_widths[ii], occ[0] + 0.5*occ_widths[ii], occ[0] + 0.5*occ_widths[ii], occ[0] - 0.5*occ_widths[ii], occ[0] - 0.5*occ_widths[ii]],
             [occ[1] - 0.5*occ_widths[ii], occ[1] - 0.5*occ_widths[ii], occ[1] + 0.5*occ_widths[ii], occ[1] + 0.5*occ_widths[ii], occ[1] - 0.5*occ_widths[ii]], 'r')

# plot two small arrows in the origin to indicate the direction of the x and y axis
plt.arrow(0, 0, 0.1, 0, head_width=0.05, head_length=0.1, fc='k', ec='k')
plt.arrow(0, 0, 0, 0.1, head_width=0.05, head_length=0.1, fc='k', ec='k')

plt.title("Directed Graph Visualization")
plt.axis('equal')
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.show()