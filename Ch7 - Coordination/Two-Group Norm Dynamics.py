"""
Two-Group Norm Dynamics
=======================
Replicates the two-group evolutionary model from Ch 7 (Smaldino).

Agents are arranged in a 2D grid, divided into two groups (e.g., by color
or region). Each agent has a binary "norm" trait (0 or 1) that evolves via
social learning (Fermi rule). Payoffs depend on the norm and group context,
with possible inter-group interactions.

The model explores how norms can diverge between groups, leading to
cultural differentiation. Parallel parameter sweeps generate heatmaps
showing fixation probabilities for different initial frequencies and
interaction strengths (Fig 7.10 & 7.11).
"""

# ## 1. Packages & Parallel Processing Setup

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import os

N_WORKERS = min(10, os.cpu_count() or 4)  # PARAM

# ## 2. Agent definition

class Player:
    """Agent on a 2D grid with a binary norm (1 or 0) and group membership."""
    __slots__ = ('id', 'pos', 'old_norm', 'norm', 'groupID', 'payoff')

    def __init__(self, aid, pos, norm, groupID):
        self.id = aid
        self.pos = pos
        self.old_norm = norm
        self.norm = norm
        self.groupID = groupID   # 0 = Group A, 1 = Group B
        self.payoff = 0.0

# ## 3. Model

class TwoGroupsModel:
    """Two-group norm-evolution model on a non-periodic MxM grid (Manhattan)."""

    def __init__(self, M=30, init_norm1_groupA=0.6, init_norm1_groupB=0.4,
                 norm1_prosocial_benefit=1.0, norm1_coord_benefit=1.0,
                 norm1_rarity_cost=1.0, prob_outgroup_observation=0.0, symm=False):
        self.M = M
        self.norm1_prosocial_benefit = norm1_prosocial_benefit  # g
        self.norm1_coord_benefit = norm1_coord_benefit          # δ
        self.norm1_rarity_cost = norm1_rarity_cost              # h
        self.prob_outgroup_observation = prob_outgroup_observation  # m
        self.symm = symm

        half_x = M // 2
        self.agents = {}
        self.groupA_ids = []
        self.groupB_ids = []
        aid = 1
        for y in range(M):
            for x in range(M):
                gid = 0 if x < half_x else 1
                init_prob = init_norm1_groupA if gid == 0 else init_norm1_groupB
                s = 1 if np.random.random() < init_prob else 0
                a = Player(aid, (y, x), s, gid)
                self.agents[aid] = a
                if gid == 0:
                    self.groupA_ids.append(aid)
                else:
                    self.groupB_ids.append(aid)
                aid += 1

    # ## 4. Dynamics — model_step!

    def step(self):
        agents = self.agents
        n = len(agents)
        ids = list(agents.keys())

        if self.symm:
            for a in agents.values():
                a.old_norm = a.norm

        g = self.norm1_prosocial_benefit
        delta = self.norm1_coord_benefit
        h = self.norm1_rarity_cost

        # --- Payoffs for Group A ---
        gA = self.groupA_ids
        if gA:
            sA = sum(agents[i].norm for i in gA)
            lenA = len(gA)
            p1A = sA / lenA
            norm1_payA = p1A * (1 + g + delta) + (1 - p1A) * (1 - h)
            norm2_payA = p1A * (1 + g) + (1 - p1A)
            for i in gA:
                a = agents[i]
                a.payoff = norm1_payA if a.norm == 1 else norm2_payA

        # --- Payoffs for Group B ---
        gB = self.groupB_ids
        if gB:
            sB = sum(agents[i].norm for i in gB)
            lenB = len(gB)
            p1B = sB / lenB
            norm1_payB = p1B * (1 + g + delta) + (1 - p1B) * (1 - h)
            norm2_payB = p1B * (1 + g) + (1 - p1B)
            for i in gB:
                a = agents[i]
                a.payoff = norm1_payB if a.norm == 1 else norm2_payB

        m = self.prob_outgroup_observation

        # --- Social learning ---
        order = list(ids)
        np.random.shuffle(order)
        for idx in order:
            a = agents[idx]
            my_group = a.groupID
            other_group = (my_group + 1) % 2 if np.random.random() < m else my_group
            group_ids = gA if other_group == 0 else gB

            if not group_ids:
                continue
            if other_group == my_group and len(group_ids) == 1:
                continue

            while True:
                cand_id = group_ids[np.random.randint(len(group_ids))]
                if other_group != my_group or cand_id != a.id:
                    break

            observed = agents[cand_id]
            payoff_diff = observed.payoff - a.payoff
            prob_copy = 1.0 / (1.0 + np.exp(-payoff_diff))

            if np.random.random() < prob_copy:
                a.norm = observed.old_norm if self.symm else observed.norm


# ## 5. Running simulation batches

def _run_trajectory(args):
    """Run a single trajectory; pickleable for multiprocessing."""
    (M, init_norm1_groupA, init_norm1_groupB, g, delta, h, m_val,
     symm, max_steps) = args

    model = TwoGroupsModel(
        M=M, init_norm1_groupA=init_norm1_groupA,
        init_norm1_groupB=init_norm1_groupB,
        norm1_prosocial_benefit=g, norm1_coord_benefit=delta,
        norm1_rarity_cost=h, prob_outgroup_observation=m_val, symm=symm,
    )
    gA = model.groupA_ids
    gB = model.groupB_ids

    groupA_freq = [sum(model.agents[i].norm for i in gA) / len(gA) if gA else 0.0]
    groupB_freq = [sum(model.agents[i].norm for i in gB) / len(gB) if gB else 0.0]

    for _ in range(max_steps):
        model.step()
        groupA_freq.append(sum(model.agents[i].norm for i in gA) / len(gA) if gA else 0.0)
        groupB_freq.append(sum(model.agents[i].norm for i in gB) / len(gB) if gB else 0.0)

    return groupA_freq, groupB_freq


def _run_batch(args):
    """Run multiple trials for one parameter combo."""
    tag, h, g, m_val, n_trials, M, init_A, init_B, delta, symm, max_steps = args
    results = []
    for _ in range(n_trials):
        traj = _run_trajectory((M, init_A, init_B, g, delta, h, m_val, symm, max_steps))
        results.append(traj)
    return tag, h, g, m_val, results


# ## 7. Parameter sweep settings

# Fig 7.10
h_vals = [0.25, 0.5, 0.75, 1.0]        # PARAM
m_range = np.arange(0.0, 0.082, 0.002)  # PARAM
n_trials_710 = 50                        # PARAM
max_steps_710 = 500                      # PARAM

# Fig 7.11
g_range = np.arange(0.0, 6.5, 0.5)      # PARAM
m_vals = [0.014, 0.016, 0.018]          # PARAM
n_trials_711 = 100                       # PARAM
max_steps_711 = 1000                     # PARAM

delta_val = 1.0   # PARAM
M_val = 30        # PARAM
init_A = 0.8      # PARAM
init_B = 0.15     # PARAM
symm = False       # PARAM


# ## 8. Simulate

def run_experiments():
    param_cases = []

    # Fig 7.10
    for h in h_vals:
        for m_val in m_range:
            param_cases.append(('fig710', h, 1.0, round(m_val, 4),
                                n_trials_710, M_val, init_A, init_B,
                                delta_val, symm, max_steps_710))
    # Fig 7.11
    for m_val in m_vals:
        for g in g_range:
            param_cases.append(('fig711', 0.5, round(g, 4), m_val,
                                n_trials_711, M_val, init_A, init_B,
                                delta_val, symm, max_steps_711))

    total = len(param_cases)
    print(f"Running {total} parameter combos across {N_WORKERS} workers…")

    results_710 = {}
    results_711 = {}
    done = 0

    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        for tag, h, g, m_val, trajs in pool.map(_run_batch, param_cases):
            if tag == 'fig710':
                results_710.setdefault(h, {})[m_val] = trajs
            else:
                results_711.setdefault(m_val, {})[g] = trajs
            done += 1
            if done % 20 == 0 or done == total:
                print(f"  {done}/{total} done")

    print("All experiments complete.")
    return results_710, results_711


# ## 9. Plotting

def plot_fig710(results_710, h_vals, m_range, ax):
    colors = plt.cm.tab10(np.linspace(0, 1, len(h_vals)))
    for i, h in enumerate(h_vals):
        inner = results_710[h]
        m_all, B_all = [], []
        B_mean = []
        for m_val in m_range:
            m_val = round(m_val, 4)
            trajs = inner[m_val]
            finals = [gB[-1] for (_, gB) in trajs]
            m_all.extend([m_val] * len(finals))
            B_all.extend(finals)
            B_mean.append(np.mean(finals))

        ax.scatter(m_all, B_all, alpha=0.25, s=12, color=colors[i], linewidths=0)
        ax.plot(m_range, B_mean, linewidth=2.5, color=colors[i], label=f'h = {h}')

    ax.set_xlabel('prob outgroup observation, m')
    ax.set_ylabel('mean frequency of norm 1 in group B')
    ax.legend()


def plot_fig711(results_711, m_vals, g_range, ax):
    colors = plt.cm.tab10(np.linspace(0, 1, len(m_vals)))
    for i, m_val in enumerate(m_vals):
        inner = results_711[m_val]
        g_all, B_all = [], []
        B_mean = []
        for g in g_range:
            g = round(g, 4)
            trajs = inner[g]
            finals = [gB[-1] for (_, gB) in trajs]
            g_all.extend([g] * len(finals))
            B_all.extend(finals)
            B_mean.append(np.mean(finals))

        ax.scatter(g_all, B_all, alpha=0.25, s=12, color=colors[i], linewidths=0)
        ax.plot(g_range, B_mean, linewidth=2.5, color=colors[i], label=f'm = {m_val}')

    ax.set_xlabel('prosocial benefit, g')
    ax.set_ylabel('mean frequency of norm 1 in group B')
    ax.legend(loc='right')


# ## 10. Main

if __name__ == '__main__':
    results_710, results_711 = run_experiments()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8))
    fig.subplots_adjust(hspace=0.35)

    plot_fig710(results_710, h_vals, m_range, ax1)
    plot_fig711(results_711, m_vals, g_range, ax2)

    plt.tight_layout()
    plt.show()
