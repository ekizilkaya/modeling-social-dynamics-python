"""
Schelling Segregation GUI
==========================
Interactive GUI for the Schelling segregation model from Ch 3 (Smaldino).

Agents on a 2D grid belong to two groups (e.g., ethnic groups). Each agent
has a "similarity threshold" — they are "happy" if at least that fraction
of neighbors share their group. Unhappy agents move to empty spots.

The GUI animates segregation emergence in real-time. Sliders adjust the
similarity threshold and initial group ratio. Buttons reset/initialize
the grid. The model demonstrates how mild preferences for similarity
can lead to extreme segregation.
"""

# ## 1. Packages

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button
from matplotlib.colors import ListedColormap

# ## 2. Agent definition

class SchellingAgent:
    """Agent on a 2D grid with mood and group membership."""
    def __init__(self, aid, pos, group):
        self.id = aid
        self.pos = pos          # (row, col) on the grid
        self.mood = False       # All agents start unhappy
        self.group = group      # 1 = red, 2 = blue

# ## 3. Model initialization

class SchellingModel:
    """Schelling segregation model on a toroidal MxM grid (Chebyshev / Moore neighbourhood)."""

    def __init__(self, M=60, min_to_be_happy=0.5, rho=0.8,
                 proportion_red=0.5, lonely_agents_unhappy=True):
        self.M = M                                          # PARAM - grid size
        self.min_to_be_happy = min_to_be_happy              # PARAM
        self.rho = rho                                      # PARAM - density
        self.proportion_red = proportion_red                # PARAM
        self.lonely_agents_unhappy = lonely_agents_unhappy  # PARAM

        self.total_agents = int(round(rho * M * M))
        self.average_similarity = 0.0
        self.similarity_history = []

        # Grid: 0 = empty, agent.id otherwise (ids start at 1)
        self.grid = np.zeros((M, M), dtype=int)
        self.agents = {}          # id -> SchellingAgent
        self._next_id = 1

        # Place agents on random unique positions
        all_positions = [(r, c) for r in range(M) for c in range(M)]
        np.random.shuffle(all_positions)

        for i in range(self.total_agents):
            pos = all_positions[i]
            group = 1 if np.random.random() < proportion_red else 2
            agent = SchellingAgent(self._next_id, pos, group)
            self.agents[agent.id] = agent
            self.grid[pos[0], pos[1]] = agent.id
            self._next_id += 1

    # ----- Space helpers (toroidal, Chebyshev / Moore) -----

    def _neighbors(self, agent):
        """Return list of neighbouring agents (Moore neighbourhood, radius 1, toroidal)."""
        r, c = agent.pos
        nbrs = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr = (r + dr) % self.M
                nc = (c + dc) % self.M
                aid = self.grid[nr, nc]
                if aid != 0:
                    nbrs.append(self.agents[aid])
        return nbrs

    def _empty_cells(self):
        """Return list of (row, col) tuples for all empty cells."""
        return list(zip(*np.where(self.grid == 0)))

    def _move_to_random_empty(self, agent):
        """Move agent to a random empty cell."""
        empties = self._empty_cells()
        if not empties:
            return
        new_pos = empties[np.random.randint(len(empties))]
        # Vacate old position
        self.grid[agent.pos[0], agent.pos[1]] = 0
        # Occupy new position
        agent.pos = new_pos
        self.grid[new_pos[0], new_pos[1]] = agent.id

    # ## 4. Dynamics — agent_step! / model_step!

    def _step_agent(self, agent):
        """Advance a single agent: check neighbours, update mood, move if unhappy."""
        nbrs = self._neighbors(agent)

        # Handle case where agent has no neighbours
        if len(nbrs) == 0:
            if self.lonely_agents_unhappy:
                agent.mood = False               # Lonely agents are unhappy
                self._move_to_random_empty(agent)
            else:
                agent.mood = True                # Lonely agents are content
            return

        # Fraction of neighbours that share the same group
        same = sum(1 for n in nbrs if n.group == agent.group)
        frac = same / len(nbrs)

        if frac >= self.min_to_be_happy:
            agent.mood = True                    # Agent is happy
        else:
            agent.mood = False                   # Agent is unhappy
            self._move_to_random_empty(agent)    # Move to random empty cell

    def step(self):
        """One model step: advance all agents in random order, record similarity."""
        ids = list(self.agents.keys())
        np.random.shuffle(ids)
        for aid in ids:
            self._step_agent(self.agents[aid])
        self.average_similarity = self._calculate_average_similarity()
        self.similarity_history.append(self.average_similarity)

    # ## 5. Helper — measurement

    def _calculate_average_similarity(self):
        """Average fraction of same-group neighbours across all agents."""
        sims = []
        for a in self.agents.values():
            nbrs = self._neighbors(a)
            if len(nbrs) == 0:
                sims.append(1.0)
            else:
                same = sum(1 for n in nbrs if n.group == a.group)
                sims.append(same / len(nbrs))
        return np.mean(sims) if sims else 0.0

    # ----- Grid image for visualisation -----

    def grid_image(self):
        """Return an MxM array: 0 = empty (black), 1 = group 1 (red), 2 = group 2 (blue)."""
        img = np.zeros((self.M, self.M), dtype=int)
        for a in self.agents.values():
            img[a.pos[0], a.pos[1]] = a.group
        return img


# ## 6. GUI setup

# Colour map: 0→black (empty), 1→red, 2→blue
cmap = ListedColormap(['black', 'red', 'royalblue'])

# --- Default parameters ---
DEFAULT_M = 60                      # PARAM
DEFAULT_RHO = 0.8                   # PARAM
DEFAULT_PROPORTION_RED = 0.5        # PARAM
DEFAULT_LONELY_UNHAPPY = True       # PARAM

model = SchellingModel(
    M=DEFAULT_M, rho=DEFAULT_RHO,
    min_to_be_happy=0.1,
    proportion_red=DEFAULT_PROPORTION_RED,
    lonely_agents_unhappy=DEFAULT_LONELY_UNHAPPY
)

# --- Figure layout ---
fig = plt.figure(figsize=(14, 8))
fig.patch.set_facecolor('#2b2b2b')

ax_grid = fig.add_axes([0.02, 0.25, 0.52, 0.70])
ax_data = fig.add_axes([0.60, 0.25, 0.37, 0.70])

def draw_grid(ax, model):
    ax.clear()
    ax.imshow(model.grid_image(), cmap=cmap, vmin=0, vmax=2,
              interpolation='nearest', origin='upper')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f'Step {len(model.similarity_history)}', color='white')

def draw_data(ax, model):
    ax.clear()
    ax.plot(model.similarity_history, color='orange', linewidth=1.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('Avg Similarity')
    ax.set_title('Average Similarity')
    ax.set_ylim(0, 1)

draw_grid(ax_grid, model)
draw_data(ax_data, model)

# --- Slider ---
ax_happy = fig.add_axes([0.15, 0.13, 0.70, 0.03])
happy_slider = Slider(ax_happy, 'Min Happy', 0.1, 0.9, valinit=0.1, valstep=0.01)

# --- Buttons ---
ax_play  = fig.add_axes([0.15, 0.04, 0.12, 0.05])
ax_step  = fig.add_axes([0.29, 0.04, 0.12, 0.05])
ax_reset = fig.add_axes([0.43, 0.04, 0.12, 0.05])

play_btn  = Button(ax_play,  'Play')
step_btn  = Button(ax_step,  'Step')
reset_btn = Button(ax_reset, 'Reset')

playing = [False]

def _sync(_val):
    model.min_to_be_happy = happy_slider.val

happy_slider.on_changed(_sync)

def _redraw():
    draw_grid(ax_grid, model)
    draw_data(ax_data, model)
    fig.canvas.draw_idle()

def _animate(_frame):
    if playing[0]:
        model.step()
        _redraw()

ani = animation.FuncAnimation(fig, _animate, interval=50, cache_frame_data=False)

# ## 7. Launch GUI & Controls

def _on_play(_event):
    playing[0] = not playing[0]
    play_btn.label.set_text('Pause' if playing[0] else 'Play')
    fig.canvas.draw_idle()

def _on_step(_event):
    model.step()
    _redraw()

def _on_reset(_event):
    global model
    playing[0] = False
    play_btn.label.set_text('Play')
    model = SchellingModel(
        M=DEFAULT_M, rho=DEFAULT_RHO,
        min_to_be_happy=happy_slider.val,
        proportion_red=DEFAULT_PROPORTION_RED,
        lonely_agents_unhappy=DEFAULT_LONELY_UNHAPPY
    )
    _redraw()

play_btn.on_clicked(_on_play)
step_btn.on_clicked(_on_step)
reset_btn.on_clicked(_on_reset)

plt.show()
