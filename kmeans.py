# CMPG 313 Project: K-Means Clustering of Dense Network with Energy Analysis
# Author: Zanokuhle Jola
# This script generates a dense network, applies K-Means clustering, 
# and visualizes node clusters with energy levels in 2D and 3D.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.interpolate import griddata
import matplotlib.cm as cm

# Set random seed for reproducibility
np.random.seed(42)

# -------------------------------
# 1. Create a dense network (600 nodes)
# -------------------------------
sizes = [85, 95, 80, 90, 75, 90, 85]  # total 600 nodes
p_in = [0.30, 0.28, 0.32, 0.29, 0.31, 0.27, 0.30]  # intra-community edges
p_out = 0.05  # inter-community edges

n_blocks = len(sizes)
probs = np.full((n_blocks, n_blocks), p_out, dtype=float)
for i in range(n_blocks):
    probs[i, i] = p_in[i]

G = nx.stochastic_block_model(sizes, probs, seed=42)

# -------------------------------
# 2. Generate node positions
# -------------------------------
base_pos = nx.spring_layout(G, seed=42, k=0.11, iterations=300)
nodes = list(G.nodes())
xy = np.array([base_pos[n] for n in nodes])

# Normalize coordinates to 0–1
xy[:, 0] = (xy[:, 0] - xy[:, 0].min()) / (xy[:, 0].max() - xy[:, 0].min())
xy[:, 1] = (xy[:, 1] - xy[:, 1].min()) / (xy[:, 1].max() - xy[:, 1].min())

# Community centers to make clusters overlap
community_centers = np.array([
    [0.25, 0.45],
    [0.38, 0.68],
    [0.52, 0.74],
    [0.68, 0.55],
    [0.68, 0.28],
    [0.48, 0.20],
    [0.22, 0.22]
])

membership = []
for block_id, sz in enumerate(sizes):
    membership.extend([block_id] * sz)
membership = np.array(membership)

new_xy = np.zeros_like(xy)
for block_id in range(len(sizes)):
    idx = np.where(membership == block_id)[0]
    local = xy[idx].copy()
    local[:, 0] -= local[:, 0].mean()
    local[:, 1] -= local[:, 1].mean()
    local *= 0.18  # compress nodes to overlap
    local += np.random.normal(0, 0.025, local.shape)  # add jitter
    local += community_centers[block_id]  # move to target center
    new_xy[idx] = local

x = new_xy[:, 0] * 100
y = new_xy[:, 1] * 100

# -------------------------------
# 3. Assign random energy values to nodes
# -------------------------------
energy = np.random.randint(10, 101, size=len(nodes))
df = pd.DataFrame({
    "Node": [f"N{n}" for n in nodes],
    "X": x,
    "Y": y,
    "Energy": energy
})

# -------------------------------
# 4. Apply K-Means clustering
# -------------------------------
k = 3  # number of clusters (change to 7 if needed)
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(df[["X", "Y"]])
centroids = kmeans.cluster_centers_

# -------------------------------
# 5. Create graph with node names
# -------------------------------
G2 = nx.Graph()
for u, v in G.edges():
    G2.add_edge(f"N{u}", f"N{v}")

coord_map = dict(zip(df["Node"], zip(df["X"], df["Y"])))

cluster_colors = {
    0: "#e41a1c", 1: "#377eb8", 2: "#984ea3",
    3: "#999999", 4: "#a65628", 5: "#ff7f00", 6: "#f781bf"
}

# -------------------------------
# 6. Plot 2D dense network
# -------------------------------
plt.figure(figsize=(12, 12), facecolor="#efefef")
ax = plt.gca()
ax.set_facecolor("#efefef")

nx.draw_networkx_edges(
    G2, pos=coord_map, edge_color="black", width=0.35, alpha=0.10
)

for cluster_id in sorted(df["Cluster"].unique()):
    cluster_nodes = df[df["Cluster"] == cluster_id]["Node"].tolist()
    nx.draw_networkx_nodes(
        G2, pos=coord_map, nodelist=cluster_nodes,
        node_color=cluster_colors[cluster_id],
        node_size=95, alpha=0.82,
        linewidths=0.25, edgecolors="white"
    )

plt.scatter(
    centroids[:, 0], centroids[:, 1],
    s=430, c="black", marker="X",
    linewidths=1.5, label="Centroids"
)
plt.title("Dense K-Means Clustered Network with Centroids", fontsize=15)
plt.axis("off")
plt.legend()
plt.show()

# -------------------------------
# 7. Plot 3D scatter of node energy
# -------------------------------
fig = plt.figure(figsize=(11, 9))
ax = fig.add_subplot(111, projection="3d")

for cluster_id in sorted(df["Cluster"].unique()):
    cdf = df[df["Cluster"] == cluster_id]
    ax.scatter(cdf["X"], cdf["Y"], cdf["Energy"], s=12, alpha=0.85, label=f"Cluster {cluster_id}")

# Centroid energies
centroid_energy = [df[df["Cluster"] == c]["Energy"].mean() for c in sorted(df["Cluster"].unique())]
ax.scatter(centroids[:, 0], centroids[:, 1], centroid_energy, s=250, c="black", marker="X", label="Centroids")

ax.set_title("3D Clustered Network with Node Energy")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Energy")
ax.legend()
plt.show()

# -------------------------------
# 8. Smooth 3D energy surface
# -------------------------------
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection="3d")

# Interpolate energy for smooth surface
grid_x, grid_y = np.meshgrid(
    np.linspace(df["X"].min(), df["X"].max(), 150),
    np.linspace(df["Y"].min(), df["Y"].max(), 150)
)
grid_z = griddata((df["X"], df["Y"]), df["Energy"], (grid_x, grid_y), method='cubic')

# Plot surface
surf = ax.plot_surface(grid_x, grid_y, grid_z, cmap=cm.viridis, linewidth=0, antialiased=True, alpha=0.85)
fig.colorbar(surf, ax=ax, shrink=0.6, aspect=12, label="Energy Level")

# Overlay nodes and centroids
ax.scatter(df["X"], df["Y"], df["Energy"], s=5, alpha=0.4)
ax.scatter(centroids[:, 0], centroids[:, 1], centroid_energy, s=250, c="red", marker="X", label="Centroids")

ax.set_title("Smooth 3D Energy Surface with K-Means Clusters")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Energy")
ax.legend()
plt.show()
