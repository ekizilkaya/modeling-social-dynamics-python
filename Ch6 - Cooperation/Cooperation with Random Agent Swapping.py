"""
Cooperation with Random Agent Swapping
=======================================
Replicates the cooperation model with random partner swapping from Ch 6
(Smaldino).

Agents on a 2D grid play prisoner's dilemma with neighbors, but each step
a random agent is swapped with another random agent, disrupting spatial
structure. Strategies (cooperate/defect) evolve via social learning
(Fermi rule).

This model contrasts with spatial clustering: without stable neighbors,
cooperation cannot form protective clusters, leading to defector
dominance. Parallel sweeps over temptation (b) and mutation rate (μ)
confirm that random mixing undermines cooperation.
"""

# ## 1. Packages & Parallel Processing Setup

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
import re, os

N_WORKERS = min(10, os.cpu_count() or 4)  # PARAM

# ## 2. Agent definition

class Player:
    """Agent on a 2D grid: strategy 0 = defector, 1 = cooperator."""
    __slots__ = ('id', 'pos', 'old_strategy', 'strategy', 'payoff')

    def __init__(self, aid, pos, strategy):
        self.id = aid
        self.pos = pos
        self.old_strategy = strategy
        self.strategy = strategy
        self.payoff = 0.0

# ## 3. Model

class RandomModel:
    """Cooperation game on a toroidal MxM grid with random agent-swapping."""

    def __init__(self, M=10, init_coop_freq=0.5, b=0.47, c=0.25, prob=0.0, symm=False):
        self.M = M
        self.payoff_benefit = b            # PARAM
        self.payoff_cost = c               # PARAM
        self.init_coop_freq = init_coop_freq  # PARAM
        self.symm = symm
        self.prob = prob                   # PARAM — swap probability

        n = M * M
        num_coop = round(n * init_coop_freq)
        strategies = [1] * num_coop + [0] * (n - num_coop)
        np.random.shuffle(strategies)

        self.grid = np.zeros((M, M), dtype=int)
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

    def _swap_agents(self, a, b):
        """Swap grid positions of two agents."""
        self.grid[a.pos[0], a.pos[1]] = b.id
        self.grid[b.pos[0], b.pos[1]] = a.id
        a.pos, b.pos = b.pos, a.pos

    # ## 4. Dynamics — agent_step! (swap places)

    def _agent_step(self, a):
        if np.random.random() < self.prob:
            ids = list(self.agents.keys())
            bid = ids[np.random.randint(len(ids))]
            while bid == a.id:
                bid = ids[np.random.randint(len(ids))]
            self._swap_agents(a, self.agents[bid])

    # ## 5. Dynamics — model_step! (payoff & update strategy)

    def step(self):
        ids = list(self.agents.keys())
        np.random.shuffle(ids)
        agents_shuffled = [self.agents[i] for i in ids]

        # Agent step: random swaps
        for a in agents_shuffled:
            self._agent_step(a)

        # Save old strategies
        for a in agents_shuffled:
            a.old_strategy = a.strategy

        # Compute payoffs
        for a in agents_shuffled:
            nbrs = self._neighbours(a)
            n_coop = sum(1 for nb in nbrs if nb.strategy == 1)
            n_def = len(nbrs) - n_coop
            if a.strategy == 1:
                a.payoff = n_coop * (self.payoff_benefit - self.payoff_cost) - n_def * self.payoff_cost
            else:
                a.payoff = n_coop * self.payoff_benefit

        # Update strategies
        for a in agents_shuffled:
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


# ## 7. Running simulation batches

def _rununtil(model, max_steps=200):
    for step in range(max_steps):
        model.step()
        if all(a.old_strategy == a.strategy for a in model.agents.values()):
            return step + 1
    return max_steps

def _run_single(args):
    """Pickleable function for ProcessPoolExecutor."""
    pb, symm, n_trials, M, init_coop_freq, b, c = args
    steps_arr = []
    coop_frac_arr = []
    for _ in range(n_trials):
        model = RandomModel(M=M, init_coop_freq=init_coop_freq, symm=symm, b=b, c=c, prob=pb)
        steps = _rununtil(model)
        n = len(model.agents)
        frac = sum(1 for a in model.agents.values() if a.strategy == 1) / n
        steps_arr.append(steps)
        coop_frac_arr.append(frac)
    label = f"pb = {pb}, symm = {symm}"
    return label, {
        'steps_arr': np.array(steps_arr),
        'coop_frac_arr': np.array(coop_frac_arr),
        'steps_mean': np.mean(steps_arr),
        'coop_frac_mean': np.mean(coop_frac_arr),
    }


# ## 8. Simulate

def run_experiments():
    prob_range = np.arange(0, 0.105, 0.005)  # PARAM
    symm_range = [False, True]
    n_trials = 100   # PARAM
    M = 30           # PARAM
    init_coop_freq = 0.5  # PARAM
    b = 1.0          # PARAM
    c = 0.2          # PARAM

    param_combos = [
        (round(pb, 4), s, n_trials, M, init_coop_freq, b, c)
        for pb in prob_range for s in symm_range
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


# ## 9. Plot — cooperator frequency vs swap probability

def plot_coop_frequency_vs_prob(results_dict):
    pb_values = sorted({float(re.search(r'pb = ([\d.]+)', k).group(1)) for k in results_dict})

    mean_true, mean_false = [], []
    sx_t, sy_t, sx_f, sy_f = [], [], [], []

    for pb in pb_values:
        for symm in [True, False]:
            label = f"pb = {pb}, symm = {symm}"
            for freq in results_dict[label]['coop_frac_arr']:
                if symm:
                    sx_t.append(pb); sy_t.append(freq)
                else:
                    sx_f.append(pb); sy_f.append(freq)
        ft = [f for x, f in zip(sx_t, sy_t) if x == pb]
        ff = [f for x, f in zip(sx_f, sy_f) if x == pb]
        mean_true.append(np.mean(ft) if ft else np.nan)
        mean_false.append(np.mean(ff) if ff else np.nan)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(sx_t, sy_t, c='blue', marker='D', alpha=0.4, s=16)
    ax.scatter(sx_f, sy_f, c='red',  alpha=0.4, s=12)
    ax.plot(pb_values, mean_true,  'b-D', linewidth=2, label='Mean Synchronous')
    ax.plot(pb_values, mean_false, 'r-o', linewidth=2, label='Mean Asynchronous')
    ax.set_xlabel('Probability')
    ax.set_ylabel('Cooperator Frequency')
    ax.legend()
    plt.tight_layout()
    return fig


# ## 10. Main

if __name__ == '__main__':
    results_dict = run_experiments()
    plot_coop_frequency_vs_prob(results_dict)
    plt.show()