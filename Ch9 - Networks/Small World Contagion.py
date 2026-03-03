"""
Small-World Contagion  (Fig 9.12)
==================================
Simple vs complex contagion on Watts–Strogatz small-world graphs.

* simulate_contagion(): SIR-like spread on a WS graph
* Parameter sweep over rewiring probability (log-spaced)
* Parallel via ProcessPoolExecutor
* Two plots: (a) ticks to full contagion  (b) percent spread
"""

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import os, itertools

N_WORKERS = min(10, os.cpu_count() or 4)

# ── 1. Contagion simulation ────────────────────────────────────────────────

def simulate_contagion(complex_contagion: bool,
                       prob_infection: float,
                       rewiring_prob: float,
                       num_nodes: int = 500,
                       k: int = 4,
                       rng: np.random.Generator | None = None):
    """Simulate contagion on a Watts–Strogatz graph.

    Returns (percent_infected, ticks).
    """
    if rng is None:
        rng = np.random.default_rng()

    # --- build WS graph as adjacency list ---
    adj = _watts_strogatz(num_nodes, k, rewiring_prob, rng)

    infected = np.zeros(num_nodes, dtype=np.bool_)

    seed = rng.integers(num_nodes)
    nbrs = adj[seed]
    if len(nbrs) == 0:
        return (0.0, 0)
    seed2 = nbrs[rng.integers(len(nbrs))]

    infected[seed] = True
    infected[seed2] = True

    prev_count = 0
    curr_count = int(infected.sum())
    ticks = 0

    while curr_count != prev_count and curr_count < num_nodes:
        prev_count = curr_count
        infected_nodes = np.where(infected)[0].copy()
        rng.shuffle(infected_nodes)

        for u in infected_nodes:
            for v in adj[u]:
                if not infected[v]:
                    if not complex_contagion or _infected_nbr_count(adj[v], infected) > 1:
                        if rng.random() <= prob_infection:
                            infected[v] = True

        curr_count = int(infected.sum())
        ticks += 1

    percent = 100.0 * curr_count / num_nodes
    return (percent, ticks)


def _infected_nbr_count(neighbors, infected):
    c = 0
    for w in neighbors:
        if infected[w]:
            c += 1
    return c


def _watts_strogatz(n, k, p, rng):
    """Build a Watts–Strogatz graph as a list-of-lists adjacency structure."""
    half_k = k // 2
    # start with ring lattice
    edge_set = set()
    for i in range(n):
        for j in range(1, half_k + 1):
            nbr = (i + j) % n
            edge_set.add((min(i, nbr), max(i, nbr)))

    # rewire
    for i in range(n):
        for j in range(1, half_k + 1):
            nbr = (i + j) % n
            edge = (min(i, nbr), max(i, nbr))
            if edge in edge_set and rng.random() < p:
                edge_set.discard(edge)
                while True:
                    new_nbr = rng.integers(n)
                    if new_nbr != i:
                        new_edge = (min(i, new_nbr), max(i, new_nbr))
                        if new_edge not in edge_set:
                            edge_set.add(new_edge)
                            break

    adj = [[] for _ in range(n)]
    for u, v in edge_set:
        adj[u].append(v)
        adj[v].append(u)
    return adj


# ── 2. Sweep worker ────────────────────────────────────────────────────────

def _sweep(args):
    """Run *trials* replications for one (complex_contagion, rewiring_prob) combo."""
    cc, rw, p_infect, trials = args
    rng = np.random.default_rng()
    perc_list = np.empty(trials)
    ticks_list = np.empty(trials, dtype=np.int64)
    for i in range(trials):
        perc, t = simulate_contagion(cc, p_infect, rw, rng=rng)
        perc_list[i] = perc
        ticks_list[i] = t
    return {
        "complex_contagion": cc,
        "rewiring_prob": rw,
        "avg_percent": perc_list.mean(),
        "perc_list": perc_list,
        "avg_ticks": ticks_list.mean(),
        "ticks_list": ticks_list,
    }


# ── 3. Main ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # --- parameters ---
    rw_log = sorted(set(
        [0.0] + [m * 10.0**k for k in range(-3, 1) for m in range(1, 10)
                 if m * 10.0**k <= 1.0]
    ))
    complex_vals = [False, True]
    p_infect = 1.0
    trials = 100

    jobs = [(cc, rw, p_infect, trials)
            for cc in complex_vals for rw in rw_log]

    total = len(jobs)
    print(f"Running {total} combos × {trials} trials across {N_WORKERS} workers …")

    results = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        for i, res in enumerate(pool.map(_sweep, jobs), 1):
            results.append(res)
            tag = "complex" if res["complex_contagion"] else "simple"
            print(f"  done {i}/{total}  {tag}  rw={res['rewiring_prob']:.4f}")

    print("All trials complete.")

    # ── custom x-axis mapping (linear near 0, log beyond rw0) ──
    rw0 = 1e-3
    delta = 0.08
    x1 = np.log10(rw0)
    x0 = x1 - delta

    def xmap(rw):
        if rw <= rw0:
            return x0 + (rw / rw0) * (x1 - x0)
        return np.log10(rw)

    xtick_vals = [1e-3, 1e-2, 1e-1, 1.0]
    xtick_pos = [xmap(v) for v in xtick_vals]
    xtick_labels = ["0.001", "0.010", "0.100", "1.000"]

    case_colors = {False: "purple", True: "orange"}
    case_labels = {False: "simple", True: "complex"}
    case_markers = {False: "*", True: "o"}
    plot_n = 100
    rng_plot = np.random.default_rng(42)

    # ── Plot 9.12(a): ticks (simple only, matching Julia code) ──
    fig1, ax1 = plt.subplots(figsize=(9, 5.2))
    for cc in [False]:
        subset = sorted(
            [r for r in results if r["complex_contagion"] == cc],
            key=lambda r: r["rewiring_prob"],
        )
        rw_line = [xmap(r["rewiring_prob"]) for r in subset]
        mean_line = [r["avg_ticks"] for r in subset]
        ax1.plot(rw_line, mean_line, lw=4,
                 color=case_colors[cc], label=case_labels[cc])

        xs, ys = [], []
        for r in subset:
            tl = r["ticks_list"]
            nkeep = min(plot_n, len(tl))
            idx = rng_plot.choice(len(tl), nkeep, replace=False)
            xval = xmap(r["rewiring_prob"])
            xs.extend([xval] * nkeep)
            ys.extend(tl[idx].tolist())
        ax1.scatter(xs, ys, s=16, alpha=0.18,
                    color=case_colors[cc], marker=case_markers[cc])

    ax1.set_xticks(xtick_pos)
    ax1.set_xticklabels(xtick_labels)
    ax1.set_xlim(x0 - 0.02, np.log10(1.0) + 0.05)
    ax1.set_xlabel("rewiring probability")
    ax1.set_ylabel("Ticks")
    ax1.legend(loc="upper right")
    fig1.tight_layout()
    plt.show(block=False)

    # ── Plot 9.12(b): percent spread (simple & complex) ──
    fig2, ax2 = plt.subplots(figsize=(9, 6.2))
    for cc in [False, True]:
        subset = sorted(
            [r for r in results if r["complex_contagion"] == cc],
            key=lambda r: r["rewiring_prob"],
        )
        rw_line = [xmap(r["rewiring_prob"]) for r in subset]
        mean_line = [r["avg_percent"] for r in subset]
        ax2.plot(rw_line, mean_line, lw=4,
                 color=case_colors[cc], label=case_labels[cc])

        xs, ys = [], []
        for r in subset:
            pl = r["perc_list"]
            nkeep = min(plot_n, len(pl))
            idx = rng_plot.choice(len(pl), nkeep, replace=False)
            xval = xmap(r["rewiring_prob"])
            xs.extend([xval] * nkeep)
            ys.extend(pl[idx].tolist())
        ax2.scatter(xs, ys, s=16, alpha=0.18,
                    color=case_colors[cc], marker=case_markers[cc])

    ax2.set_xticks(xtick_pos)
    ax2.set_xticklabels(xtick_labels)
    ax2.set_xlim(x0 - 0.02, np.log10(1.0) + 0.05)
    ax2.set_xlabel("rewiring probability")
    ax2.set_ylabel("percent spread")
    ax2.legend(loc="right")
    fig2.tight_layout()
    plt.show(block=False)

    plt.show()

