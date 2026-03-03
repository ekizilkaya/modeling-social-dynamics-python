"""
Coordination with Asymmetric Payoffs  –  Numba-accelerated
===========================================================
Replicates the coordination-game sweep from Ch 7 (Smaldino).

The entire per-trial simulation loop (init → step × N → return mean norm)
is compiled to machine code with ``@numba.njit``.  All trials are run in
parallel via ``numba.prange`` — no ProcessPoolExecutor overhead.
"""

import numpy as np
import numba as nb
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# ── 1. Numba-jitted simulation core ────────────────────────────────────────

@nb.njit(cache=True)
def _run_trial_jit(M, init_norm1, prosocial, coord, rarity, steps):
    """One trial: build grid, step until fixation or *steps*, return mean norm."""
    N = M * M
    n1 = int(round(init_norm1 * N))

    # --- build & shuffle initial norms ---
    norms = np.zeros(N, dtype=np.int8)
    for i in range(n1):
        norms[i] = 1
    for i in range(N - 1, 0, -1):               # Fisher-Yates
        j = np.random.randint(0, i + 1)
        norms[i], norms[j] = norms[j], norms[i]

    payoffs = np.empty(N, dtype=np.float64)
    order = np.arange(N)

    for _ in range(steps):
        # frequency of norm 1
        p1 = 0.0
        for i in range(N):
            p1 += norms[i]
        p1 /= N
        if p1 == 0.0 or p1 == 1.0:
            break

        norm1_pay = p1 * (1.0 + prosocial + coord) + (1.0 - p1) * (1.0 - rarity)
        norm2_pay = p1 * (1.0 + prosocial) + (1.0 - p1) * 1.0

        for i in range(N):
            payoffs[i] = norm1_pay if norms[i] == 1 else norm2_pay

        # shuffle agent order (Fisher-Yates)
        for i in range(N - 1, 0, -1):
            j = np.random.randint(0, i + 1)
            order[i], order[j] = order[j], order[i]

        # social learning – Fermi rule
        for k in range(N):
            idx = order[k]
            partner = np.random.randint(0, N - 1)
            if partner >= idx:
                partner += 1
            delta = payoffs[partner] - payoffs[idx]
            prob_copy = 1.0 / (1.0 + np.exp(-delta))
            if np.random.random() < prob_copy:
                norms[idx] = norms[partner]

    s = 0.0
    for i in range(N):
        s += norms[i]
    return s / N


@nb.njit(parallel=True, cache=True)
def run_all_trials(M, p0_vals, n_trials, prosocial, coord, rarity, steps):
    """Run all (p0 × n_trials) in parallel threads via prange."""
    n_p0 = len(p0_vals)
    total = n_p0 * n_trials
    results = np.empty(total, dtype=np.float64)
    for i in nb.prange(total):
        p0_idx = i // n_trials
        results[i] = _run_trial_jit(M, p0_vals[p0_idx],
                                    prosocial, coord, rarity, steps)
    return results


# ── 2. Main ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # --- parameters ---
    coord_val     = 1.0    # δ  norm1_coord_benefit
    rarity_val    = 0.5    # h  norm1_rarity_cost
    prosocial_val = 1.0    # g  norm1_prosocial_benefit
    p_star = rarity_val / (coord_val + rarity_val)

    p0_vals    = np.arange(0.18, 0.505, 0.01)
    n_trials   = 100
    M_val      = 31
    steps_val  = 100
    threshold  = 0.99

    total = len(p0_vals) * n_trials

    # warm-up JIT (tiny dummy run — compilation cost paid once)
    print("JIT compiling …", end=" ", flush=True)
    _run_trial_jit(3, 0.5, 0.0, 0.0, 0.0, 1)
    run_all_trials(3, np.array([0.5]), 1, 0.0, 0.0, 0.0, 1)
    print("done.")

    # --- run sweep ---
    print(f"Running {total} trials ({len(p0_vals)} p0 × {n_trials}) …",
          flush=True)

    results = run_all_trials(M_val, p0_vals, n_trials,
                             prosocial_val, coord_val, rarity_val, steps_val)

    print("All trials complete.")

    # --- organise results ---
    results_2d = results.reshape(len(p0_vals), n_trials)  # (n_p0, n_trials)

    prop_win = np.mean(results_2d >= threshold, axis=1)

    # scatter data
    x_dots = np.repeat(p0_vals, n_trials)
    y_dots = (results >= threshold).astype(np.float64)

    # --- plot ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x_dots, y_dots, color="orange", alpha=0.2, s=16,
               linewidths=0, zorder=2)
    ax.plot(p0_vals, prop_win, color="black", linewidth=3, zorder=3)
    ax.axvline(p_star, linestyle="--", color="gray", zorder=1)
    ax.set_xlabel("initial frequency of norm 1")
    ax.set_ylabel("proportion of runs norm 1 wins")
    ax.set_ylim(-0.1, 1.1)
    ax.set_xlim(p0_vals[0], p0_vals[-1])
    plt.tight_layout()
    plt.show()

