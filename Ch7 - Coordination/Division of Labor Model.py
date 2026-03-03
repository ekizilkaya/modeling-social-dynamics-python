"""
Division of Labor Model
=======================
Replicates the evolutionary model of task specialization from Ch 7
(Smaldino).

Agents are arranged in a 2D grid and belong to groups. Each agent has a
"norm" trait (0 or 1) determining which task they specialize in. Payoffs
depend on group composition: mixed groups (both norms) get higher rewards
due to division of labor, but pure groups are stable.

Evolution proceeds via social learning (Fermi rule) and occasional
mutation. The model explores how division of labor can emerge and persist
under group-level selection pressures (Fig 7.12).
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
    __slots__ = ('id', 'pos', 'old_norm', 'norm', 'groupID', 'payoff')

    def __init__(self, aid, pos, norm, groupID):
        self.id = aid
        self.pos = pos
        self.old_norm = norm
        self.norm = norm
        self.groupID = groupID
        self.payoff = 0.0


# ## 3. Model

class DivLaborModel:
    """Division-of-labor model with two groups on a non-periodic MxM grid."""

    def __init__(self, M=30, init_norm1_groupA=0.6, init_norm1_groupB=0.4,
                 surplus_benefit=7.0, norm1_share=0.7,
                 prob_outgroup_interaction=0.9, prob_outgroup_observation=0.0,
                 asymm=False):
        self.M = M
        self.surplus_benefit = surplus_benefit          # G
        self.norm1_share = norm1_share                  # gamma
        self.prob_outgroup_interaction = prob_outgroup_interaction  # d
        self.prob_outgroup_observation = prob_outgroup_observation  # m
        self.asymm = asymm

        half_x = M // 2
        self.agents = {}
        self.group0_ids = []
        self.group1_ids = []
        aid = 1
        for y in range(M):
            for x in range(M):
                gid = 0 if x < half_x else 1
                init_prob = init_norm1_groupA if gid == 0 else init_norm1_groupB
                s = 1 if np.random.random() < init_prob else 0
                a = Player(aid, (y, x), s, gid)
                self.agents[aid] = a
                if gid == 0:
                    self.group0_ids.append(aid)
                else:
                    self.group1_ids.append(aid)
                aid += 1

    # --- Helper: update payoffs for one group ---
    @staticmethod
    def _update_group(group_ids, out_ids, agents, prob_out, pay_high, pay_low):
        if not group_ids:
            return
        s_in = sum(agents[i].norm for i in group_ids)
        p_in = s_in / len(group_ids)

        p_out = 0.0
        if out_ids:
            s_out = sum(agents[i].norm for i in out_ids)
            p_out = s_out / len(out_ids)

        norm1_pay = (prob_out * ((1 - p_out) * (1 + pay_high) + p_out) +
                     (1 - prob_out) * ((1 - p_in) * (1 + pay_high) + p_in))
        norm2_pay = (prob_out * ((1 - p_out) + p_out * (1 + pay_low)) +
                     (1 - prob_out) * ((1 - p_in) + p_in * (1 + pay_low)))

        for i in group_ids:
            a = agents[i]
            a.payoff = norm1_pay if a.norm == 1 else norm2_pay

    # --- Step ---
    def step(self):
        agents = self.agents
        n = len(agents)
        ids = list(agents.keys())

        if self.asymm:
            for a in agents.values():
                a.old_norm = a.norm

        prob_out = self.prob_outgroup_interaction
        surplus = self.surplus_benefit
        share = self.norm1_share
        pay_high = share * surplus
        pay_low = (1 - share) * surplus

        g0 = self.group0_ids
        g1 = self.group1_ids

        self._update_group(g0, g1, agents, prob_out, pay_high, pay_low)
        self._update_group(g1, g0, agents, prob_out, pay_high, pay_low)

        m = self.prob_outgroup_observation

        order = list(ids)
        np.random.shuffle(order)
        for idx in order:
            a = agents[idx]
            my_group = a.groupID
            other_group = (my_group + 1) % 2 if np.random.random() < m else my_group
            group_ids = g0 if other_group == 0 else g1

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
                a.norm = observed.old_norm if self.asymm else observed.norm


# ## 4. Batch helpers (pickleable for multiprocessing)

def _run_batch(args):
    """Run n_trials trajectories for one parameter combo."""
    (tag, x_val, d_val, gamma_val, G_val, m_val, init_A, init_B,
     asymm, M, n_trials, max_steps) = args

    results_A = []
    results_B = []
    for _ in range(n_trials):
        model = DivLaborModel(
            M=M, init_norm1_groupA=init_A, init_norm1_groupB=init_B,
            surplus_benefit=G_val, norm1_share=gamma_val,
            prob_outgroup_interaction=d_val,
            prob_outgroup_observation=m_val, asymm=asymm,
        )
        g0 = model.group0_ids
        g1 = model.group1_ids

        for _ in range(max_steps):
            model.step()

        fA = sum(model.agents[i].norm for i in g0) / len(g0) if g0 else 0.0
        fB = sum(model.agents[i].norm for i in g1) / len(g1) if g1 else 0.0
        results_A.append(fA)
        results_B.append(fB)

    return tag, x_val, results_A, results_B


# ## 5. Parameter sweep settings

d_range = np.arange(0.40, 0.91, 0.02)      # PARAM  left panel x-axis
gamma_range = np.arange(0.50, 1.01, 0.02)   # PARAM  right panel x-axis

G_val = 3.0          # PARAM  surplus_benefit
m_val = 0.03         # PARAM  prob_outgroup_observation
gamma_fix = 0.5      # PARAM  gamma fixed when sweeping d
d_fix = 0.8          # PARAM  d fixed when sweeping gamma
n_trials = 50        # PARAM
M_val = 30           # PARAM
max_steps = 300      # PARAM
init_A = 0.6         # PARAM
init_B = 0.4         # PARAM
asymm = False        # PARAM


# ## 6. Plot helper

def plot_panel(ax, results_dict, xs, xlabel_str):
    x_all, A_all, B_all = [], [], []
    A_mean, B_mean = [], []

    for x in xs:
        x_r = round(x, 4)
        A_vec, B_vec = results_dict[x_r]
        x_all.extend([x_r] * len(A_vec))
        A_all.extend(A_vec)
        B_all.extend(B_vec)
        A_mean.append(np.mean(A_vec))
        B_mean.append(np.mean(B_vec))

    ax.scatter(x_all, A_all, alpha=0.25, s=12, color='tab:blue', linewidths=0)
    ax.scatter(x_all, B_all, alpha=0.25, s=12, color='tab:orange', linewidths=0)
    ax.plot(xs, A_mean, lw=3, color='tab:blue', label='type A')
    ax.plot(xs, B_mean, lw=3, color='tab:orange', label='type B')
    ax.set_xlabel(xlabel_str)
    ax.set_ylabel('mean frequency of norm 1')
    ax.legend()


# ## 7. Main

if __name__ == '__main__':
    param_cases = []

    # Left panel: vary d, hold gamma fixed
    for d in d_range:
        param_cases.append(('d', round(d, 4), round(d, 4), gamma_fix,
                            G_val, m_val, init_A, init_B, asymm,
                            M_val, n_trials, max_steps))

    # Right panel: vary gamma, hold d fixed
    for g in gamma_range:
        param_cases.append(('gamma', round(g, 4), d_fix, round(g, 4),
                            G_val, m_val, init_A, init_B, asymm,
                            M_val, n_trials, max_steps))

    total = len(param_cases)
    print(f"Running {total} parameter combos across {N_WORKERS} workers...")

    results_d = {}
    results_gamma = {}
    done = 0

    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        for tag, x_val, A_vec, B_vec in pool.map(_run_batch, param_cases):
            if tag == 'd':
                results_d[x_val] = (A_vec, B_vec)
            else:
                results_gamma[x_val] = (A_vec, B_vec)
            done += 1
            if done % 10 == 0 or done == total:
                print(f"  {done}/{total} done")

    print("All experiments complete.")

    # Plot Figure 7.12
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    plot_panel(ax1, results_d, d_range, 'prob outgroup interaction, d')
    plot_panel(ax2, results_gamma, gamma_range, 'asymmetric division of labor, gamma')

    plt.tight_layout()
    plt.show()


