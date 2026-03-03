"""
Bayesian Updating of Hypothesis Beliefs
=========================================
Replicates the Bayesian-updating simulation from Ch 8 (Smaldino).

A scientist starts with a prior probability that a hypothesis is true and
repeatedly runs experiments.  Each experiment yields a positive or negative
result governed by the statistical **power** (1 − β) and the **false-positive
rate** (α).  After every result the belief is updated via Bayes' rule.

The script runs multiple independent trajectories for both a true and a
false hypothesis, then plots belief over time (Fig 8.1 / 8.2 style),
showing how evidence accumulates and beliefs converge — or sometimes
mislead — depending on power and α.
"""

# ## 1. Packages

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


# ## 2. Bayesian update

def bayes_step(prob_true, true_hypothesis, power, false_positive_rate):
    if true_hypothesis:
        positive_result = np.random.random() < power
    else:
        positive_result = np.random.random() < false_positive_rate

    if positive_result:
        # positive result → update with power / false-positive rate
        prob_true = (power * prob_true) / (
            power * prob_true + false_positive_rate * (1 - prob_true))
    else:
        # negative result → update with (1-power) / (1-false_positive_rate)
        prob_true = ((1 - power) * prob_true) / (
            (1 - power) * prob_true + (1 - false_positive_rate) * (1 - prob_true))

    return prob_true


# ## 3. Helper – running simulation

def run_timeseries(initial_prior, true_hypothesis, power, false_positive_rate, n_steps):
    prob_true = initial_prior
    data = [prob_true]  # time 0

    for _ in range(n_steps):
        prob_true = bayes_step(prob_true, true_hypothesis, power, false_positive_rate)
        data.append(prob_true)

    return data


# ## 4. Parameters, simulate, and plot

initial_prior = 0.2          # PARAM
true_hypothesis = True       # PARAM
power = 0.8                  # PARAM
false_positive_rate = 0.05   # PARAM
n_steps = 18                 # PARAM

if __name__ == '__main__':
    data = run_timeseries(initial_prior, true_hypothesis,
                          power, false_positive_rate, n_steps)
    times = np.arange(n_steps + 1)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(times, data)
    ax.set_title('Probability of True Hypothesis')
    ax.set_xlabel('time')
    ax.set_ylabel('Pr(true)')
    plt.tight_layout()
    plt.show()
