"""
Ring, Random & Small-World Networks
====================================
Replicates the network visualisations from Ch 9 (Smaldino).

1. Regular-k ring  (Watts–Strogatz with p=0)
2. Erdős–Rényi random graph
3. Barabási–Albert scale-free graph  (PMF + CCDF on log-log axes)
"""

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import networkx as nx
from collections import Counter

# ── 1. Regular-k Ring Network ───────────────────────────────────────────────

g1 = nx.watts_strogatz_graph(11, 8, 0)

fig1, ax1 = plt.subplots(figsize=(8, 6))
pos1 = nx.circular_layout(g1)
nx.draw(g1, pos1, ax=ax1, with_labels=True,
        node_size=300, node_color="lightblue",
        edge_color="gray", width=1.0)
ax1.set_title("Regular-k Ring  (n=11, k=8, p=0)")
fig1.tight_layout()
plt.show(block=False)

# ── 2. Erdős–Rényi Network ─────────────────────────────────────────────────

g2 = nx.erdos_renyi_graph(100, 0.1)
degrees2 = [d for _, d in g2.degree()]

fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(14, 6))

# circular layout
pos2 = nx.circular_layout(g2)
nx.draw(g2, pos2, ax=ax2a, node_size=30, node_color="steelblue",
        edge_color="gray", width=0.4, with_labels=False)
ax2a.set_title("ER Graph  (n=100, p=0.1)")

# degree histogram
ax2b.hist(degrees2, bins=range(0, max(degrees2) + 2),
          align="left", edgecolor="black", color="steelblue")
ax2b.set_xlabel("Degree")
ax2b.set_ylabel("Count")
ax2b.set_title("Degree Distribution")

fig2.tight_layout()
plt.show(block=False)

print(f"max-deg:{max(degrees2)}, min-deg:{min(degrees2)}, "
      f"#links:{g2.number_of_edges()}")

# ── 3. Barabási–Albert (Scale-free) Network ────────────────────────────────

g3 = nx.barabasi_albert_graph(1000, 1)
degrees3 = np.array([d for _, d in g3.degree()])

degree_counts = Counter(degrees3)
unique_degrees = np.array(sorted(degree_counts.keys()))
frequencies = np.array([degree_counts[k] for k in unique_degrees])
pmf = frequencies / frequencies.sum()

# --- Log-log PMF of node degrees ---
fig3, ax3 = plt.subplots(figsize=(7, 5))
ax3.scatter(unique_degrees, pmf, s=30, zorder=3, label="Empirical P(k)")

x_line = np.linspace(unique_degrees.min(), unique_degrees.max(), 100)
y_line = 2.0 / (x_line ** 3)
ax3.plot(x_line, y_line, linewidth=2, color="tab:orange",
         label=r"Reference: $2k^{-3}$")

ax3.set_xscale("log")
ax3.set_yscale("log")
ax3.set_xlabel("Degree")
ax3.set_ylabel("Probability")
ax3.set_xlim(1, 1e2)
ax3.set_ylim(10**-3.1, 1.0)
ax3.set_yticks([1e-3, 1e-2, 1e-1, 1])
ax3.legend(loc="upper right")
ax3.set_title("Log-Log PMF of Node Degrees (BA, n=1000, m=1)")
fig3.tight_layout()
plt.show(block=False)

# --- Log-log CCDF: P(K >= k) ---
ccdf = np.cumsum(pmf[::-1])[::-1]

fig4, ax4 = plt.subplots(figsize=(7, 5))
ax4.plot(unique_degrees, ccdf, marker="o", markersize=4, linewidth=1)
ax4.set_xscale("log")
ax4.set_yscale("log")
ax4.set_xlabel("Degree")
ax4.set_ylabel("CCDF")
ax4.set_ylim(10**-3.1, 1.0)
ax4.set_yticks([1e-3, 1e-2, 1e-1, 1])
ax4.set_title("Log-Log CCDF of Node Degrees")
fig4.tight_layout()
plt.show(block=False)

plt.show()


