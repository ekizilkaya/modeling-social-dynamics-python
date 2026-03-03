"""
Natural Selection of Bad Science (NSOBS)
=========================================
Replicates the evolutionary model of scientific laboratories from Ch 8
(Smaldino & McElreath, 2016).

Each "lab" agent has two evolvable traits — statistical **power** (1 − β)
and **effort** (sample size).  Labs conduct studies, accumulate publication
payoff, and reproduce via tournament selection (highest payoff copied,
oldest removed).  Mutation nudges power/effort; because publishing is
rewarded regardless of truth, selection pressure can drive power down and
false-positive rates up — the "natural selection of bad science."

Two scenarios are plotted (Fig 8.9):
  (a) Effort mutates, power fixed   → effort declines, FPR / FDR rise.
  (b) Power mutates, effort fixed   → power declines, FPR / FDR rise.
"""

# ## 1. Packages

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


# ## 2. Agent definition

class Lab:
    __slots__ = ('id', 'power', 'effort', 'false_pos_rate', 'pub_payoff', 'age')

    def __init__(self, aid, power, effort, pub_payoff=0.0, age=0):
        self.id = aid
        self.power = power
        self.effort = effort
        self.false_pos_rate = power / (1 + (1 - power) * effort)
        self.pub_payoff = pub_payoff
        self.age = age


# ## 3. Model

class ScienceModel:
    """Evolutionary model of scientific labs with power/effort traits."""

    def __init__(self, num_labs=100, effort_influence=0.2,
                 reproduction_samplesize=10.0, base_rate=0.1,
                 effort_init=75.0, power_init=0.8,
                 prob_publish_neg_result=0.0, payoff_neg_result=0.0,
                 mutation_rate_effort=0.1, mutation_rate_power=0.01,
                 mutation_SD_effort=1.0, mutation_SD_power=0.01):
        self.effort_influence = effort_influence
        self.reproduction_samplesize = reproduction_samplesize
        self.base_rate = base_rate
        self.prob_publish_neg_result = prob_publish_neg_result
        self.payoff_neg_result = payoff_neg_result
        self.mutation_rate_effort = mutation_rate_effort
        self.mutation_rate_power = mutation_rate_power
        self.mutation_SD_effort = mutation_SD_effort
        self.mutation_SD_power = mutation_SD_power

        self._next_id = 1
        self.agents = {}
        for _ in range(num_labs):
            self._add_lab(power_init, effort_init)

    def _add_lab(self, power, effort, pub_payoff=0.0, age=0):
        lab = Lab(self._next_id, power, effort, pub_payoff, age)
        self.agents[self._next_id] = lab
        self._next_id += 1
        return lab

    # --- Science production ---
    @staticmethod
    def _do_science(agent, model):
        if np.random.random() < 1.0 - model.effort_influence * np.log10(agent.effort):
            actual_truth = np.random.random() < model.base_rate

            if actual_truth:
                if np.random.random() < agent.power:
                    agent.pub_payoff += 1  # true positive
                else:
                    if np.random.random() < model.prob_publish_neg_result:
                        agent.pub_payoff += model.payoff_neg_result
            else:
                if np.random.random() < agent.false_pos_rate:
                    agent.pub_payoff += 1  # false positive
                else:
                    if np.random.random() < model.prob_publish_neg_result:
                        agent.pub_payoff += model.payoff_neg_result

    # --- Mutation ---
    def _mutate(self, agent):
        if np.random.random() < self.mutation_rate_power:
            agent.power += np.random.randn() * self.mutation_SD_power
            agent.power = np.clip(agent.power, 0.0, 1.0)

        if np.random.random() < self.mutation_rate_effort:
            agent.effort += np.random.randn() * self.mutation_SD_effort
            agent.effort = np.clip(agent.effort, 1.0, 100.0)

        agent.false_pos_rate = agent.power / (1 + (1 - agent.power) * agent.effort)

    # --- Evolution (death + birth) ---
    def _evolution(self):
        sample_size = int(round(self.reproduction_samplesize))
        labs = list(self.agents.values())
        if not labs:
            return
        k = min(sample_size, len(labs))
        if k <= 0:
            return

        # Death: sample k labs, remove the oldest (random tie-break)
        death_sample = list(np.random.choice(labs, size=k, replace=False))
        max_age = max(a.age for a in death_sample)
        oldest = [a for a in death_sample if a.age == max_age]
        to_die = oldest[np.random.randint(len(oldest))]
        del self.agents[to_die.id]

        # Birth: sample k labs (from updated population), copy the fanciest
        labs = list(self.agents.values())
        k = min(sample_size, len(labs))
        if k <= 0:
            return
        birth_sample = list(np.random.choice(labs, size=k, replace=False))
        max_pub = max(a.pub_payoff for a in birth_sample)
        fanciest = [a for a in birth_sample if a.pub_payoff == max_pub]
        parent = fanciest[np.random.randint(len(fanciest))]

        offspring = self._add_lab(parent.power, parent.effort, pub_payoff=0.0, age=0)
        self._mutate(offspring)

    # --- False discovery rate ---
    def false_discovery_rate(self):
        labs = list(self.agents.values())
        pow_mean = np.mean([a.power for a in labs])
        fpr_mean = np.mean([a.false_pos_rate for a in labs])

        br = self.base_rate
        pneg = self.prob_publish_neg_result

        total_pubs = (br * (pow_mean + (1 - pow_mean) * pneg) +
                      (1 - br) * (fpr_mean + (1 - fpr_mean) * pneg))
        false_pubs = (br * (1 - pow_mean) * pneg +
                      (1 - br) * fpr_mean)
        return false_pubs / total_pubs if total_pubs > 0 else 0.0

    # --- One model step ---
    def step(self):
        # Agent step: each lab does science (random order)
        ids = list(self.agents.keys())
        np.random.shuffle(ids)
        for aid in ids:
            if aid in self.agents:
                self._do_science(self.agents[aid], self)

        # Model step: evolution, age, fdr
        self._evolution()
        for a in self.agents.values():
            a.age += 1


# ## 4. Helper – running simulations

def run_trial(effort_influence, mutation_rate_effort, mutation_rate_power,
              max_steps=80000):
    model = ScienceModel(effort_influence=effort_influence,
                         mutation_rate_effort=mutation_rate_effort,
                         mutation_rate_power=mutation_rate_power)

    ts_time = np.empty(max_steps)
    ts_effort = np.empty(max_steps)
    ts_power = np.empty(max_steps)
    ts_fpr = np.empty(max_steps)
    ts_fdr = np.empty(max_steps)

    for t in range(max_steps):
        model.step()
        labs = list(model.agents.values())
        ts_time[t] = t + 1
        ts_effort[t] = np.mean([a.effort for a in labs])
        ts_power[t] = np.mean([a.power for a in labs])
        ts_fpr[t] = np.mean([a.false_pos_rate for a in labs])
        ts_fdr[t] = model.false_discovery_rate()

        if (t + 1) % 10000 == 0:
            print(f"  step {t + 1}/{max_steps}")

    return ts_time, ts_effort, ts_power, ts_fpr, ts_fdr


# ## 5. Helper – build plot

def sample_run_plot(axes, effort_influence=0.2, mutation_rate_effort=0.1,
                    mutation_rate_power=0.01, max_steps=80000):
    print(f"Running: effort_inf={effort_influence}, "
          f"mut_effort={mutation_rate_effort}, mut_power={mutation_rate_power}, "
          f"steps={max_steps}")

    ts_time, ts_effort, ts_power, ts_fpr, ts_fdr = run_trial(
        effort_influence, mutation_rate_effort, mutation_rate_power,
        max_steps=max_steps)

    # Thin for faster plotting
    maxpts = 10000  # PARAM
    step = max(1, len(ts_time) // maxpts)
    idx = slice(None, None, step)

    t = ts_time[idx]
    pwr = ts_power[idx]
    fpr = ts_fpr[idx]
    fd = ts_fdr[idx]
    eff = ts_effort[idx]

    ax1, ax2 = axes

    ax1.plot(t, pwr, label='power')
    ax1.plot(t, fpr, label='false pos')
    ax1.plot(t, fd, label='false disc')
    ax1.set_ylim(0, 1)
    ax1.set_xlabel('time')
    ax1.set_title('Power and false positives')
    ax1.legend(loc='lower right')

    ax2.plot(t, eff, color='tab:blue')
    ax2.set_ylim(0, 100)
    ax2.set_xlabel('time')
    ax2.set_ylabel('effort')
    ax2.set_title('Effort')


# ## 6. Simulate and plot

if __name__ == '__main__':
    # Fig 8.9 top
    fig1, (ax1a, ax1b) = plt.subplots(2, 1, figsize=(9, 8))
    fig1.subplots_adjust(hspace=0.35)
    sample_run_plot((ax1a, ax1b),
                    effort_influence=0.2,       # PARAM
                    mutation_rate_effort=0.1,    # PARAM
                    mutation_rate_power=0.00,    # PARAM
                    max_steps=150000)            # PARAM
    fig1.suptitle('Fig 8.9 top: effort mutates, power fixed', y=1.01)
    fig1.tight_layout()

    # Fig 8.9 bottom
    fig2, (ax2a, ax2b) = plt.subplots(2, 1, figsize=(9, 8))
    fig2.subplots_adjust(hspace=0.35)
    sample_run_plot((ax2a, ax2b),
                    effort_influence=0.2,       # PARAM
                    mutation_rate_effort=0.0,    # PARAM
                    mutation_rate_power=0.01,    # PARAM
                    max_steps=80000)             # PARAM
    fig2.suptitle('Fig 8.9 bottom: power mutates, effort fixed', y=1.01)
    fig2.tight_layout()

    plt.show()
