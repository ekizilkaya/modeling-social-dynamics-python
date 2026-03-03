"""
Cooperation Game with Clustering
=================================
Replicates the spatial cooperation model with Hoshen-Kopelman clustering
from Ch 6 (Smaldino).

Agents on a 2D grid play a prisoner's dilemma with neighbors. Strategies
(cooperate/defect) evolve via social learning (Fermi rule). The model
explores how spatial structure promotes cooperation through cluster
formation: cooperators form stable clusters that resist invasion by
defectors.

Parallel parameter sweeps over temptation (b) and mutation rate (μ)
generate heatmaps showing cooperation levels. The Hoshen-Kopelman
algorithm identifies and measures cluster sizes.
"""

# ## 1. Packages & Parallel Processing Setup

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
import re, os

# Number of parallel workers — adjust to available CPU cores  # PARAM
N_WORKERS = min(10, os.cpu_count() or 4)

# ## 2. Agent definition

class Player:
    """Agent on a 2D grid: strategy 0 = defector, 1 = cooperator."""
    __slots__ = ('id', 'pos', 'old_strategy', 'strategy', 'payoff')

    def __init__(self, aid, pos, strategy):
        self.id = aid
        self.pos = pos              # (row, col)
        self.old_strategy = strategy
        self.strategy = strategy
        self.payoff = 0.0

# ## 3. Model

class SimpleModel:
    """Simple cooperation game on a toroidal MxM grid (Manhattan / von-Neumann neighbourhood)."""

    def __init__(self, M=10, init_coop_freq=0.5, b=0.47, c=0.25, symm=False):
        self.M = M
        self.payoff_benefit = b           # PARAM
        self.payoff_cost = c              # PARAM
        self.init_coop_freq = init_coop_freq  # PARAM
        self.symm = symm                  # synchronous (True) vs asynchronous (False)

        n = M * M
        num_coop = round(n * init_coop_freq)
        strategies = [1] * num_coop + [0] * (n - num_coop)
        np.random.shuffle(strategies)

        self.grid = np.zeros((M, M), dtype=int)    # agent id (1-based)
        self.agents = {}
        aid = 1
        idx = 0
        for r in range(M):
            for col in range(M):
                s = strategies[idx]; idx += 1
                a = Player(aid, (r, col), s)
                self.agents[aid] = a
                self.grid[r, col] = aid
                aid += 1

    # ----- Neighbours (Manhattan-1, toroidal) -----

    def _neighbours(self, agent):
        r, c = agent.pos
        M = self.M
        nbrs = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = (r + dr) % M, (c + dc) % M
            nbrs.append(self.agents[self.grid[nr, nc]])
        return nbrs

    # ## 3. Dynamics — agent_step! (payoffs)

    def _compute_payoff(self, agent):
        n_coop = sum(1 for nb in self._neighbours(agent) if nb.strategy == 1)
        n_def = 4 - n_coop  # exactly 4 Manhattan neighbours on a full grid
        if agent.strategy == 1:
            agent.payoff = n_coop * (self.payoff_benefit - self.payoff_cost) - n_def * self.payoff_cost
        else:
            agent.payoff = n_coop * self.payoff_benefit

    # ## 4. Dynamics — model_step! (update strategy)

    def step(self):
        """One full model step: compute all payoffs, then update strategies."""
        # Compute payoffs for every agent
        for a in self.agents.values():
            self._compute_payoff(a)

        # Save old strategies
        for a in self.agents.values():
            a.old_strategy = a.strategy

        # Update strategies
        for a in self.agents.values():
            nbs = self._neighbours(a)
            max_payoff = max(nb.payoff for nb in nbs)
            if max_payoff > a.payoff:
                candidates = [nb for nb in nbs if nb.payoff == max_payoff]
                best = candidates[np.random.randint(len(candidates))]
                if self.symm:
                    a.strategy = best.old_strategy
                else:
                    a.strategy = best.strategy

    # ## 6. Helper — strategy grid

    def get_strategy_grid(self):
        grid = np.zeros((self.M, self.M), dtype=int)
        for a in self.agents.values():
            grid[a.pos[0], a.pos[1]] = a.strategy
        return grid


# ## 7–8. Hoshen–Kopelman cluster identification (periodic boundaries)

def _find_root(equiv, label):
    while equiv[label] != label:
        label = equiv[label]
    return label

def _union(equiv, l1, l2):
    r1 = _find_root(equiv, l1)
    r2 = _find_root(equiv, l2)
    if r1 < r2:
        equiv[r2] = r1
    else:
        equiv[r1] = r2

def hoshen_kopelman_periodic(grid, target=1):
    """Return (labels, cluster_sizes_dict) for *target* value on a periodic 2-D grid."""
    nrows, ncols = grid.shape
    labels = np.zeros_like(grid, dtype=int)
    max_labels = nrows * ncols + 2
    equiv = list(range(max_labels))
    current_label = 1

    for i in range(nrows):
        for j in range(ncols):
            if grid[i, j] == target:
                left = labels[i, j - 1] if j > 0 else 0
                top  = labels[i - 1, j] if i > 0 else 0
                if left == 0 and top == 0:
                    labels[i, j] = current_label
                    current_label += 1
                elif left != 0 and top == 0:
                    labels[i, j] = left
                elif left == 0 and top != 0:
                    labels[i, j] = top
                else:
                    labels[i, j] = min(left, top)
                    if left != top:
                        _union(equiv, left, top)

    # Periodic horizontal wrap
    for i in range(nrows):
        if grid[i, 0] == target and grid[i, ncols - 1] == target:
            _union(equiv, labels[i, 0], labels[i, ncols - 1])
    # Periodic vertical wrap
    for j in range(ncols):
        if grid[0, j] == target and grid[nrows - 1, j] == target:
            _union(equiv, labels[0, j], labels[nrows - 1, j])

    # Flatten roots
    for lab in range(1, current_label):
        equiv[lab] = _find_root(equiv, lab)

    # Relabel and count
    for i in range(nrows):
        for j in range(ncols):
            if labels[i, j] != 0:
                labels[i, j] = equiv[labels[i, j]]

    cluster_sizes = {}
    for i in range(nrows):
        for j in range(ncols):
            l = labels[i, j]
            if l != 0:
                cluster_sizes[l] = cluster_sizes.get(l, 0) + 1

    return labels, cluster_sizes


# ## 9. Running simulation batches

def _rununtil(model, max_steps=500):
    """Step model until equilibrium or max_steps."""
    for step in range(max_steps):
        model.step()
        stable = all(a.old_strategy == a.strategy for a in model.agents.values())
        if stable:
            return step + 1
    return max_steps

def _run_trial(M, init_coop_freq, symm, b, c):
    model = SimpleModel(M=M, init_coop_freq=init_coop_freq, symm=symm, b=b, c=c)
    steps = _rununtil(model)
    n = len(model.agents)
    final_coop_frac = sum(1 for a in model.agents.values() if a.strategy == 1) / n
    grid = model.get_strategy_grid()
    _, coop_clusters = hoshen_kopelman_periodic(grid, 1)
    _, def_clusters  = hoshen_kopelman_periodic(grid, 0)
    return steps, final_coop_frac, coop_clusters, def_clusters

def _run_single(args):
    """Pickleable wrapper for ProcessPoolExecutor."""
    c, symm, n_trials, M, init_coop_freq, b = args
    steps_arr = []
    coop_frac_arr = []
    coop_clusters_list = []
    def_clusters_list = []
    for _ in range(n_trials):
        steps, frac, cc, dc = _run_trial(M, init_coop_freq, symm, b, c)
        steps_arr.append(steps)
        coop_frac_arr.append(frac)
        coop_clusters_list.append(cc)
        def_clusters_list.append(dc)
    label = f"c = {c}, symm = {symm}"
    return label, {
        'steps_arr': np.array(steps_arr),
        'coop_frac_arr': np.array(coop_frac_arr),
        'steps_mean': np.mean(steps_arr),
        'coop_frac_mean': np.mean(coop_frac_arr),
        'coop_clusters': coop_clusters_list,
        'def_clusters': def_clusters_list,
    }


# ## 10. Simulate

def run_experiments():
    cost_range = np.arange(0, 0.525, 0.025)   # PARAM
    symm_range = [False, True]
    n_trials = 100                              # PARAM
    M = 50                                      # PARAM
    init_coop_freq = 0.5                        # PARAM
    b = 1.0                                     # PARAM

    param_combos = [
        (round(c, 4), s, n_trials, M, init_coop_freq, b)
        for c in cost_range for s in symm_range
    ]

    results_dict = {}
    total = len(param_combos)
    done = 0

    print(f"Running {total} parameter combos ({n_trials} trials each) across {N_WORKERS} workers…")

    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(_run_single, p): p for p in param_combos}
        for fut in as_completed(futures):
            label, res = fut.result()
            results_dict[label] = res
            done += 1
            if done % 5 == 0 or done == total:
                print(f"  {done}/{total} done")

    print("All experiments complete.")
    return results_dict


# ## 11. Plot — cluster histogram (synchronous vs asynchronous)

def plot_cluster_histogram_combined(results_dict, cost, cluster_type='coop', ylog=False):
    key = 'coop_clusters' if cluster_type == 'coop' else 'def_clusters'
    label_true  = f"c = {cost}, symm = True"
    label_false = f"c = {cost}, symm = False"

    sizes_true  = [s for trial in results_dict[label_true][key]  for s in trial.values()]
    sizes_false = [s for trial in results_dict[label_false][key] for s in trial.values()]

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = max(len(set(sizes_true)), len(set(sizes_false)), 10)
    ax.hist(sizes_true,  bins=bins, density=True, alpha=1.0, color='blue', label='synchronous')
    ax.hist(sizes_false, bins=bins, density=True, alpha=0.5, color='red',  label='asynchronous')
    if ylog:
        ax.set_yscale('log')
    kind = 'Cooperator' if cluster_type == 'coop' else 'Defector'
    ax.set_xlabel('Cluster Size')
    ax.set_ylabel('Log(Probability)' if ylog else 'Probability')
    ax.set_title(f'{kind} Cluster Size Distribution (c = {cost})')
    ax.legend()
    plt.tight_layout()
    return fig


# ## 12. Plot — average cluster size vs cost

def plot_avg_cluster_size(results_dict, cluster_type='coop'):
    cost_values = sorted({float(re.search(r'c = ([\d.]+)', k).group(1)) for k in results_dict})
    key = 'coop_clusters' if cluster_type == 'coop' else 'def_clusters'

    mean_true, mean_false = [], []
    sx_t, sy_t, sx_f, sy_f = [], [], [], []

    for cost in cost_values:
        all_t, all_f = [], []
        for symm in [True, False]:
            label = f"c = {cost}, symm = {symm}"
            for trial in results_dict[label][key]:
                for size in trial.values():
                    if symm:
                        all_t.append(size); sx_t.append(cost); sy_t.append(size)
                    else:
                        all_f.append(size); sx_f.append(cost); sy_f.append(size)
        mean_true.append(np.mean(all_t) if all_t else np.nan)
        mean_false.append(np.mean(all_f) if all_f else np.nan)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(sx_t, sy_t, c='blue', marker='D', alpha=0.3, s=16)
    ax.scatter(sx_f, sy_f, c='red',  alpha=0.3, s=12)
    ax.plot(cost_values, mean_true,  'b-D', linewidth=2, label='Mean Synchronous')
    ax.plot(cost_values, mean_false, 'r-o', linewidth=2, label='Mean Asynchronous')
    kind = 'Cooperator' if cluster_type == 'coop' else 'Defector'
    ax.set_xlabel('Cost'); ax.set_ylabel('Cluster Size'); ax.set_title(kind)
    ax.legend(); plt.tight_layout()
    return fig


# ## 13. Plot — cooperator frequency vs cost

def plot_coop_frequency_vs_cost(results_dict):
    cost_values = sorted({float(re.search(r'c = ([\d.]+)', k).group(1)) for k in results_dict})

    mean_true, mean_false = [], []
    sx_t, sy_t, sx_f, sy_f = [], [], [], []

    for cost in cost_values:
        for symm in [True, False]:
            label = f"c = {cost}, symm = {symm}"
            for freq in results_dict[label]['coop_frac_arr']:
                if symm:
                    sx_t.append(cost); sy_t.append(freq)
                else:
                    sx_f.append(cost); sy_f.append(freq)
        ft = [f for c2, f in zip(sx_t, sy_t) if c2 == cost]
        ff = [f for c2, f in zip(sx_f, sy_f) if c2 == cost]
        mean_true.append(np.mean(ft) if ft else np.nan)
        mean_false.append(np.mean(ff) if ff else np.nan)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(sx_t, sy_t, c='blue', marker='D', alpha=0.5, s=12)
    ax.scatter(sx_f, sy_f, c='red',  alpha=0.5, s=12)
    ax.plot(cost_values, mean_true,  'b-D', linewidth=2, label='Mean Synchronous')
    ax.plot(cost_values, mean_false, 'r-o', linewidth=2, label='Mean Asynchronous')
    ax.set_xlabel('Cost'); ax.set_ylabel('Cooperator Frequency')
    ax.legend(); plt.tight_layout()
    return fig


# ## 14. Plot — time to equilibrium vs cost (excluding limit cycles)

def plot_filtered_ticks_to_equilibrium_vs_cost(results_dict, max_steps=500):
    cost_values = sorted({float(re.search(r'c = ([\d.]+)', k).group(1)) for k in results_dict})

    mean_true, mean_false = [], []
    sx_t, sy_t, sx_f, sy_f = [], [], [], []

    for cost in cost_values:
        for symm in [True, False]:
            label = f"c = {cost}, symm = {symm}"
            for ticks in results_dict[label]['steps_arr']:
                if ticks < max_steps:
                    if symm:
                        sx_t.append(cost); sy_t.append(ticks)
                    else:
                        sx_f.append(cost); sy_f.append(ticks)
        ft = [t for c2, t in zip(sx_t, sy_t) if c2 == cost]
        ff = [t for c2, t in zip(sx_f, sy_f) if c2 == cost]
        mean_true.append(np.mean(ft) if ft else np.nan)
        mean_false.append(np.mean(ff) if ff else np.nan)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(sx_t, sy_t, c='blue', alpha=0.5, s=12)
    ax.scatter(sx_f, sy_f, c='red',  alpha=0.5, s=12)
    ax.plot(cost_values, mean_true,  'b-o', linewidth=2, label='Mean Synchronous')
    ax.plot(cost_values, mean_false, 'r-o', linewidth=2, label='Mean Asynchronous')
    ax.set_xlabel('Cost'); ax.set_ylabel('Ticks')
    ax.set_title('Ticks to Equilibrium (Excluding Limit Cycles)')
    ax.legend(); plt.tight_layout()
    return fig


# ## 15. Main — run experiments & show plots

if __name__ == '__main__':
    results_dict = run_experiments()

    plot_cluster_histogram_combined(results_dict, 0.45, 'coop', ylog=False)
    plot_avg_cluster_size(results_dict, 'coop')
    plot_coop_frequency_vs_cost(results_dict)
    plot_filtered_ticks_to_equilibrium_vs_cost(results_dict)

    plt.show()
