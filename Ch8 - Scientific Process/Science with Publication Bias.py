"""
Science with Publication Bias — Canonization of False Facts
============================================================
Replicates the publication-bias model from Ch 8 (Smaldino).

* Bayesian updating with true/false positives & negatives
* Publication bias: null results suppressed with probability ρ
* Canonization: belief crosses threshold τ  →  hypothesis accepted/rejected
* Parameter sweep over power, tau, alpha, pub_bias  →  2×2 panel plot
* Single-run trajectory example
"""

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import itertools

# ── 1. Bayesian helpers ─────────────────────────────────────────────────────

def bayes_positive(prob_true: float, power: float, alpha: float) -> float:
    """Posterior after a *positive* test result."""
    num = power * prob_true
    den = num + alpha * (1.0 - prob_true)
    return num / den


def bayes_negative(prob_true: float, power: float, alpha: float) -> float:
    """Posterior after a *negative* test result."""
    num = (1.0 - power) * prob_true
    den = num + (1.0 - alpha) * (1.0 - prob_true)
    return num / den


# ── 2. Update with publication bias ─────────────────────────────────────────

def update_false_with_pub(prob_true, power, alpha, pub_bias, rng):
    """Update when the hypothesis is actually FALSE."""
    if rng.random() < alpha:                      # false positive → always published
        return bayes_positive(prob_true, power, alpha)
    else:                                          # true negative
        if rng.random() < (1.0 - pub_bias):        # published null
            return bayes_negative(prob_true, power, alpha)
        else:                                       # unpublished null  → no update
            return prob_true


def update_true_with_pub(prob_true, power, alpha, pub_bias, rng):
    """Update when the hypothesis is actually TRUE."""
    if rng.random() < power:                       # true positive → always published
        return bayes_positive(prob_true, power, alpha)
    else:                                           # false negative
        if rng.random() < (1.0 - pub_bias):         # published null
            return bayes_negative(prob_true, power, alpha)
        else:                                        # unpublished null → no update
            return prob_true


def update_with_pub(prob_true, true_hypothesis, power, alpha, pub_bias, rng):
    if true_hypothesis:
        return update_true_with_pub(prob_true, power, alpha, pub_bias, rng)
    else:
        return update_false_with_pub(prob_true, power, alpha, pub_bias, rng)


# ── 3. Canonization dynamics ────────────────────────────────────────────────

def run_canonization(initial_prior, true_hypothesis, power, alpha,
                     pub_bias, tau, max_steps=10_000, rng=None):
    """Return True if hypothesis is canonised as true, False otherwise."""
    if rng is None:
        rng = np.random.default_rng()
    prob_true = initial_prior
    for _ in range(max_steps):
        if prob_true >= tau:
            return True
        if prob_true <= 1.0 - tau:
            return False
        prob_true = update_with_pub(prob_true, true_hypothesis,
                                    power, alpha, pub_bias, rng)
    return False


def run_canonization_traj(initial_prior, true_hypothesis, power, alpha,
                          pub_bias, tau, max_steps=10_000, rng=None):
    """Like run_canonization but also returns the trajectory."""
    if rng is None:
        rng = np.random.default_rng()
    prob_true = initial_prior
    traj = [prob_true]
    for _ in range(max_steps):
        if prob_true >= tau:
            return traj, True
        if prob_true <= 1.0 - tau:
            return traj, False
        prob_true = update_with_pub(prob_true, true_hypothesis,
                                    power, alpha, pub_bias, rng)
        traj.append(prob_true)
    return traj, False


# ── 4. Parameter sweep (one (power, tau, alpha) combo) ──────────────────────

def _sweep_one(args):
    """Worker for one (power, tau, alpha) triple across all pub_biases."""
    power, tau, alpha, pub_biases, initial_prior, n_runs = args
    rng = np.random.default_rng()
    probs = []
    for rho in pub_biases:
        count_true = 0
        for _ in range(n_runs):
            if run_canonization(initial_prior, False, power, alpha,
                                rho, tau, rng=rng):
                count_true += 1
        probs.append(count_true / n_runs)
    return (power, tau, alpha), probs


# ── 5. Main ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # --- simulation parameters ---
    initial_prior = 0.5
    powers = [0.8, 0.6]
    taus = [0.9, 0.999]
    alphas = [0.05, 0.10, 0.15, 0.20, 0.25]
    pub_biases = np.arange(0.0, 1.01, 0.02)
    n_runs = 1000

    # build argument list
    jobs = []
    for power in powers:
        for tau in taus:
            for alpha in alphas:
                jobs.append((power, tau, alpha, pub_biases,
                             initial_prior, n_runs))

    total = len(jobs)
    print(f"Running {total} parameter combos × {len(pub_biases)} "
          f"pub-bias values × {n_runs} runs each …")

    # parallel sweep
    results = {}
    with ProcessPoolExecutor() as pool:
        for i, (key, probs) in enumerate(pool.map(_sweep_one, jobs), 1):
            results[key] = probs
            print(f"  done {i}/{total}  "
                  f"power={key[0]}, tau={key[1]}, alpha={key[2]}")

    # --- 2×2 panel plot ---
    markers = ["o", "s", "^", "D", "v"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    panel_specs = [
        (0, 0, 0.8, 0.9),
        (0, 1, 0.6, 0.9),
        (1, 0, 0.8, 0.999),
        (1, 1, 0.6, 0.999),
    ]

    for row, col, power, tau in panel_specs:
        ax = axes[row, col]
        for j, alpha in enumerate(alphas):
            y = results[(power, tau, alpha)]
            ax.scatter(pub_biases, y, s=18,
                       marker=markers[j % len(markers)],
                       label=f"α = {alpha}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("publication bias (ρ)")
        ax.set_ylabel("prob canonising false fact as true")
        ax.set_title(f"1 − β = {power}")
        ax.text(0.98, 0.05, f"τ = {tau}",
                transform=ax.transAxes, ha="right", fontsize=8)
        ax.legend(loc="upper left", fontsize=7, framealpha=0.8)

    fig.tight_layout()
    plt.show(block=False)

    # --- single-run trajectory ---
    print("\n── Single-run trajectory ──")
    traj_prior = 0.2
    true_hypothesis = True
    traj_power = 0.8
    traj_alpha = 0.05
    traj_pub_bias = 0.2
    traj_tau = 0.99
    traj_max_steps = 200

    traj, canonized = run_canonization_traj(
        traj_prior, true_hypothesis, traj_power, traj_alpha,
        traj_pub_bias, traj_tau, max_steps=traj_max_steps,
    )
    times = np.arange(len(traj))

    fig2, ax2 = plt.subplots(figsize=(7, 4))
    ax2.plot(times, traj)
    ax2.set_xlabel("time")
    ax2.set_ylabel("Pr(true)")
    ax2.set_title("Probability of True Hypothesis")
    fig2.tight_layout()
    plt.show(block=False)

    print(f"Canonised as true?  {canonized}")

    plt.show()
