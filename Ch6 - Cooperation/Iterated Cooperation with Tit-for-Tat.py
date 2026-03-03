"""
Iterated Cooperation with Tit-for-Tat
======================================
Replicates the iterated prisoner's dilemma with TFT strategy from Ch 6
(Smaldino).

Agents on a 2D grid play repeated PD games with neighbors. Strategies
evolve via social learning (Fermi rule). The model demonstrates how
Tit-for-Tat (TFT) can promote cooperation in repeated interactions:
cooperate initially, then mirror the opponent's last move.

Parallel sweeps over temptation (b) and mutation rate (μ) show how
repetition and reciprocity enable cooperation to persist even in
non-spatial settings.
"""

# ## 1. Packages & Parallel Processing Setup

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

N_WORKERS = min(10, os.cpu_count() or 4)  # PARAM

# ## 2. Agent definition

class Player:
    """Agent on a 2D grid: strategy 0 = defector, 1 = cooperator/TFT."""
    __slots__ = ('id', 'pos', 'old_strategy', 'strategy', 'payoff')

    def __init__(self, aid, pos, strategy):
        self.id = aid
        self.pos = pos
        self.old_strategy = strategy
        self.strategy = strategy
        self.payoff = 0.0

# ## 3. Model

class IterationsModel:
    """Iterated cooperation game on a toroidal MxM grid with optional TFT and random swapping."""

    def __init__(self, M=10, init_coop_freq=0.5, b=0.47, c=0.25,
                 prob=0.0, symm=False, num_iterations=1, tft=False):
        self.M = M
        self.payoff_benefit = b                # PARAM
        self.payoff_cost = c                   # PARAM
        self.init_coop_freq = init_coop_freq   # PARAM
        self.symm = symm
        self.prob = prob                       # PARAM — swap probability
        self.num_iterations = num_iterations   # PARAM — rounds per interaction
        self.tft = tft                         # PARAM — tit-for-tat flag

        n = M * M
        self.grid = np.zeros((M, M), dtype=int)
        self.agents = {}
        aid = 1
        for r in range(M):
            for col in range(M):
                s = 1 if np.random.random() < init_coop_freq else 0
                a = Player(aid, (r, col), s)
                self.agents[aid] = a
                self.grid[r, col] = aid
                aid += 1

    # ----- Neighbours (Manhattan-1, toroidal) -----

    def _neighbours(self, agent):
        r, c = agent.pos
        M = self.M
        return [self.agents[self.grid[(r + dr) % M, (c + dc) % M]]
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]]

    def _swap_agents(self, a, b):
        self.grid[a.pos[0], a.pos[1]] = b.id
        self.grid[b.pos[0], b.pos[1]] = a.id
        a.pos, b.pos = b.pos, a.pos

    # ## 5. Payoff computation

    def _compute_payoff(self, a):
        b = self.payoff_benefit
        c = self.payoff_cost
        T = self.num_iterations
        tft = self.tft

        nbrs = self._neighbours(a)
        nC = sum(1 for nb in nbrs if nb.strategy == 1)
        nD = len(nbrs) - nC

        if a.strategy == 1:
            if tft:
                a.payoff = T * (nC * (b - c)) - nD * c
            else:
                a.payoff = T * (nC * (b - c) - nD * c)
        else:
            if tft:
                a.payoff = nC * b
            else:
                a.payoff = T * (nC * b)

    # ## 6. Model step

    def step(self):
        # Save old strategies
        for a in self.agents.values():
            a.old_strategy = a.strategy

        # Random swaps
        for a in list(self.agents.values()):
            if np.random.random() < self.prob:
                ids = list(self.agents.keys())
                bid = ids[np.random.randint(len(ids))]
                while bid == a.id:
                    bid = ids[np.random.randint(len(ids))]
                self._swap_agents(a, self.agents[bid])

        # Compute payoffs
        for a in self.agents.values():
            self._compute_payoff(a)

        # Evolve strategies
        order = list(self.agents.keys())
        np.random.shuffle(order)
        for aid in order:
            a = self.agents[aid]
            best = None
            best_pay = -np.inf
            for nb in self._neighbours(a):
                if nb.payoff > best_pay:
                    best_pay = nb.payoff
                    best = nb
            if best is not None and best_pay > a.payoff:
                a.strategy = best.old_strategy if self.symm else best.strategy


# ## 7. Running simulation batches

def _rununtil(model, max_steps=200):
    for step in range(max_steps):
        model.step()
        if all(a.old_strategy == a.strategy for a in model.agents.values()):
            return step + 1
    return max_steps


# ## 8. Fixed-step trial (for the parameter sweep)

def _run_fixedsteps_trial(args):
    """Run a single fixed-step trial; pickleable for multiprocessing."""
    steps, M, init_coop_freq, symm, b, c, prob, num_iterations, tft = args
    model = IterationsModel(M=M, init_coop_freq=init_coop_freq, symm=symm,
                            b=b, c=c, prob=prob,
                            num_iterations=num_iterations, tft=tft)
    for _ in range(steps):
        model.step()
    n = len(model.agents)
    return sum(1 for a in model.agents.values() if a.strategy == 1) / n


# ## 9. Parameter sweep & simulate

def run_sweep():
    T_vals = list(range(1, 21))           # PARAM
    pb_panels = [0.0, 1.0]               # PARAM
    n_trials = 20                         # PARAM
    M = 30                                # PARAM
    b = 1.0                               # PARAM
    init_coop = 0.20                      # PARAM
    c_vals = [0.2, 0.4, 0.6, 0.8]        # PARAM
    fixed_steps = 100                     # PARAM

    results = {}

    total_jobs = len(pb_panels) * len(c_vals) * len(T_vals) * n_trials
    print(f"Submitting {total_jobs} trial jobs across {N_WORKERS} workers…")

    for pb in pb_panels:
        results[pb] = {}
        for c_val in c_vals:
            means = []
            scatter_x = []
            scatter_y = []

            # Build all jobs for this (pb, c) combination
            jobs = []
            for T in T_vals:
                for _ in range(n_trials):
                    jobs.append((fixed_steps, M, init_coop, False, b, c_val, pb, T, True))

            # Run in parallel
            trial_results = []
            with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
                trial_results = list(pool.map(_run_fixedsteps_trial, jobs))

            # Reorganise results by T
            idx = 0
            for T in T_vals:
                t_trials = trial_results[idx:idx + n_trials]
                idx += n_trials
                scatter_x.extend([T] * n_trials)
                scatter_y.extend(t_trials)
                means.append(np.mean(t_trials))

            results[pb][c_val] = {
                'scatter': (scatter_x, scatter_y),
                'mean': means,
            }
            print(f"  pb={pb}, c={c_val} done")

    print("All experiments complete.")
    return T_vals, results


# ## 10. Plot (Fig 6.8)

def plot_results(T_vals, results):
    c_vals = [0.2, 0.4, 0.6, 0.8]
    palette = {0.2: 'grey', 0.4: 'orange', 0.6: 'dodgerblue', 0.8: 'green'}

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    for ax, pb, title in [(ax_l, 0.0, 'randomization-prob = 0'),
                           (ax_r, 1.0, 'randomization-prob = 1')]:
        for c_val in c_vals:
            sx, sy = results[pb][c_val]['scatter']
            ax.scatter(sx, sy, s=12, alpha=0.30, color=palette[c_val])
            ax.plot(T_vals, results[pb][c_val]['mean'],
                    linewidth=2.5, color=palette[c_val], label=f'c = {c_val}')
        ax.set_title(title)
        ax.set_xlabel('num-iterations')
        ax.legend()

    ax_l.set_ylabel('cooperator frequency at t = 100')
    plt.tight_layout()
    return fig


# ## 11. Main

if __name__ == '__main__':
    T_vals, results = run_sweep()
    plot_results(T_vals, results)
    plt.show()
