"""
Opinion Dynamics GUI: Positive, BC, NBC
========================================
Interactive GUI for opinion dynamics models from Ch 5 (Smaldino).

Agents on a 2D grid have continuous opinions (0-1). Three update rules:
- **Positive**: Agents adopt opinions closer to their own (consensus).
- **BC (Bounded Confidence)**: Only adopt if difference < threshold ε.
- **NBC (Negative Bounded Confidence)**: Adopt if difference > threshold ε.

The GUI allows real-time parameter adjustment (ε, update rule) and
visualizes opinion distributions over time. Sliders control ε, buttons
reset/initialize, and radio buttons switch between Positive/BC/NBC.
"""

# ## 1. Packages

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button, RadioButtons
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.cm as cm

# ## 2. Agent definition

class OpinionAgent:
    """Agent on a 2D grid with a continuous opinion in [-1, 1]."""
    def __init__(self, aid, pos, opinion):
        self.id = aid
        self.pos = pos              # (row, col)
        self.opinion = opinion      # float in [-1, 1]

# ## 3. Model initialization

class OpinionModel:
    """Positive / Bounded-Confidence / Negative-BC opinion dynamics on a toroidal grid."""

    MODE_NAMES = ['positive', 'bounded', 'negative']

    def __init__(self, M=10, local_int=False, learning_rate=0.5,
                 consensus_threshold=0.05, interaction_mode='positive',
                 confidence_threshold=0.10):
        self.M = M                                          # PARAM grid size
        self.local_int = local_int                          # PARAM local interactions
        self.learning_rate = learning_rate                  # PARAM
        self.consensus_threshold = consensus_threshold      # PARAM
        self.interaction_mode = interaction_mode            # 'positive', 'bounded', 'negative'
        self.confidence_threshold = confidence_threshold    # PARAM
        self.consensus = False
        self.time_to_consensus = 0
        self.step_count = 0

        # Grid is fully filled: one agent per cell
        self.n = M * M
        self.agents = {}
        self.grid = np.zeros((M, M), dtype=int)  # agent id (1-based), 0 = empty

        aid = 1
        for r in range(M):
            for c in range(M):
                opinion = 2.0 * np.random.random() - 1.0   # Uniform in [-1, 1]
                a = OpinionAgent(aid, (r, c), opinion)
                self.agents[aid] = a
                self.grid[r, c] = aid
                aid += 1

        # History: list of opinion snapshots per step (for spaghetti plot)
        self.opinion_history = []    # list of dicts {aid: opinion}
        self._record()

    # ----- Space helpers (toroidal, Manhattan) -----

    def _manhattan_neighbours(self, agent):
        """Return agents in the Manhattan-1 neighbourhood (4 neighbours) on the torus."""
        r, c = agent.pos
        nbrs = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr = (r + dr) % self.M
            nc = (c + dc) % self.M
            aid = self.grid[nr, nc]
            if aid != 0:
                nbrs.append(self.agents[aid])
        return nbrs

    # ----- Interaction mechanism -----

    def _interact(self, a, b):
        x1, x2 = a.opinion, b.opinion
        lr = self.learning_rate
        d = self.confidence_threshold
        mode = self.interaction_mode
        delta = abs(x1 - x2)

        if mode == 'positive':
            x1n = x1 + lr * (x2 - x1)
            x2n = x2 + lr * (x1 - x2)

        elif mode == 'bounded':
            if delta >= d:
                return  # No interaction
            x1n = x1 + lr * (x2 - x1)
            x2n = x2 + lr * (x1 - x2)

        elif mode == 'negative':
            if delta < d:
                x1n = x1 + lr * (x2 - x1)
                x2n = x2 + lr * (x1 - x2)
            elif delta > d:
                if x1 > x2:
                    x1n = x1 + lr * (x1 - x2) * (1 - x1) * 0.5
                    x2n = x2 + lr * (x2 - x1) * (1 + x2) * 0.5
                else:
                    x1n = x1 + lr * (x1 - x2) * (1 + x1) * 0.5
                    x2n = x2 + lr * (x2 - x1) * (1 - x2) * 0.5
            else:
                return  # delta == d: no change
        else:
            return

        a.opinion = x1n
        b.opinion = x2n

    # ----- Clique detection -----

    def _clique_ranges(self):
        d = self.confidence_threshold
        xs = sorted(a.opinion for a in self.agents.values())
        n = len(xs)
        if n == 0:
            return []
        if n == 1:
            return [0.0]

        ranges = []
        cmin = xs[0]
        cmax = xs[0]
        prev = xs[0]
        for k in range(1, n):
            x = xs[k]
            if (x - prev) > d:
                ranges.append(cmax - cmin)
                cmin = x
                cmax = x
            else:
                cmax = x
            prev = x
        ranges.append(cmax - cmin)
        return ranges

    # ----- Record history -----

    def _record(self):
        self.opinion_history.append({a.id: a.opinion for a in self.agents.values()})

    # ----- Model step -----

    def step(self, n_sub=1):
        """Perform n_sub sub-steps (pair interactions) per call, then record."""
        for _ in range(n_sub):
            n = self.n
            if n <= 1:
                return

            if self.local_int:
                # Pick random agent, interact with random Manhattan neighbour
                a = self.agents[np.random.randint(1, n + 1)]
                nbrs = self._manhattan_neighbours(a)
                if nbrs:
                    b = nbrs[np.random.randint(len(nbrs))]
                    self._interact(a, b)
            else:
                # Pick two distinct random agents globally
                i = np.random.randint(1, n + 1)
                j = np.random.randint(1, n)
                if j >= i:
                    j += 1
                self._interact(self.agents[i], self.agents[j])

            # Consensus check
            if not self.consensus:
                ranges = self._clique_ranges()
                if ranges and all(r < self.consensus_threshold for r in ranges):
                    self.time_to_consensus = self.step_count
                    self.consensus = True

            self.step_count += 1

        self._record()

    # ----- Visualisation helpers -----

    def grid_image(self):
        """Return MxM float array of opinions for imshow."""
        img = np.zeros((self.M, self.M))
        for a in self.agents.values():
            img[a.pos[0], a.pos[1]] = a.opinion
        return img


# ## 4. GUI setup

# Colour map: blue(-1) → white(0) → red(+1)  (RdBu reversed)
opinion_cmap = plt.get_cmap('RdBu_r')
norm = Normalize(vmin=-1, vmax=1)

# --- Default parameters ---
DEFAULT_M = 10                          # PARAM
DEFAULT_LR = 0.50
DEFAULT_THR = 0.10
DEFAULT_CONS = 0.05
DEFAULT_MODE = 'positive'
DEFAULT_LOCAL = False

model = OpinionModel(
    M=DEFAULT_M, learning_rate=DEFAULT_LR,
    confidence_threshold=DEFAULT_THR,
    consensus_threshold=DEFAULT_CONS,
    interaction_mode=DEFAULT_MODE,
    local_int=DEFAULT_LOCAL,
)

# --- Figure ---
fig = plt.figure(figsize=(16, 9))
fig.patch.set_facecolor('#f0f0f0')

ax_grid  = fig.add_axes([0.03, 0.30, 0.30, 0.60])
ax_lines = fig.add_axes([0.38, 0.30, 0.35, 0.60])

# --- Draw helpers ---

def draw_grid(ax, mdl):
    ax.clear()
    img = mdl.grid_image()
    ax.imshow(img, cmap=opinion_cmap, vmin=-1, vmax=1,
              interpolation='nearest', origin='upper')
    ax.set_xticks([]); ax.set_yticks([])
    status = f'  (consensus @ {mdl.time_to_consensus})' if mdl.consensus else ''
    ax.set_title(f'Grid  —  step {mdl.step_count}{status}', fontsize=10)


def draw_spaghetti(ax, mdl):
    ax.clear()
    hist = mdl.opinion_history
    if not hist:
        return
    ids = sorted(hist[0].keys())
    steps = list(range(len(hist)))
    for aid in ids:
        vals = [h[aid] for h in hist]
        c = opinion_cmap(norm(vals[-1]))
        ax.plot(steps, vals, color=c, linewidth=0.6, alpha=0.7)
    ax.set_ylim(-1, 1)
    ax.set_xlabel('Step')
    ax.set_ylabel('Opinion')
    ax.set_title('All opinions over time')


draw_grid(ax_grid, model)
draw_spaghetti(ax_lines, model)

# --- Interaction mode radio ---
ax_radio = fig.add_axes([0.78, 0.55, 0.18, 0.20])
radio = RadioButtons(ax_radio, ('positive', 'bounded', 'negative'), active=0)
ax_radio.set_title('Mode', fontsize=9)

def _on_mode(label):
    model.interaction_mode = label
radio.on_clicked(_on_mode)

# --- Sliders ---
slider_defs = [
    ('Learning rate',    0.20, 0.0, 1.0, 0.01, DEFAULT_LR),
    ('Confidence thr.',  0.15, 0.0, 1.0, 0.01, DEFAULT_THR),
    ('Consensus thr.',   0.10, 0.0, 1.0, 0.01, DEFAULT_CONS),
    ('SPU (steps/upd)',  0.05, 1,   50,  1,    1),
]
sliders = {}
for label, y, vmin, vmax, vstep, vinit in slider_defs:
    ax_s = fig.add_axes([0.15, y, 0.55, 0.025])
    sliders[label] = Slider(ax_s, label, vmin, vmax, valinit=vinit, valstep=vstep)

def _sync_sliders(_val):
    model.learning_rate        = sliders['Learning rate'].val
    model.confidence_threshold = sliders['Confidence thr.'].val
    model.consensus_threshold  = sliders['Consensus thr.'].val
for s in list(sliders.values())[:3]:
    s.on_changed(_sync_sliders)

# --- Local-interaction toggle (a 1-option radio acting as checkbox) ---
ax_local = fig.add_axes([0.78, 0.42, 0.18, 0.08])
local_chk = RadioButtons(ax_local, ('Global', 'Local'),
                          active=(1 if DEFAULT_LOCAL else 0))
ax_local.set_title('Interaction scope', fontsize=9)

def _on_local(label):
    model.local_int = (label == 'Local')
local_chk.on_clicked(_on_local)

# --- Buttons ---
ax_play  = fig.add_axes([0.78, 0.30, 0.08, 0.05])
ax_step  = fig.add_axes([0.87, 0.30, 0.08, 0.05])
ax_reset = fig.add_axes([0.78, 0.23, 0.08, 0.05])

play_btn  = Button(ax_play,  'Play')
step_btn  = Button(ax_step,  'Step')
reset_btn = Button(ax_reset, 'Reset')

playing = [False]

def _redraw():
    draw_grid(ax_grid, model)
    draw_spaghetti(ax_lines, model)
    fig.canvas.draw_idle()

def _animate(_frame):
    if playing[0]:
        spu = int(sliders['SPU (steps/upd)'].val)
        model.step(n_sub=spu)
        _redraw()
        # Auto-stop on consensus
        if model.consensus:
            playing[0] = False
            play_btn.label.set_text('Play')
            fig.canvas.draw_idle()

ani = animation.FuncAnimation(fig, _animate, interval=50, cache_frame_data=False)

# ## 5. Launch GUI & controls

def _on_play(_event):
    playing[0] = not playing[0]
    play_btn.label.set_text('Pause' if playing[0] else 'Play')
    fig.canvas.draw_idle()

def _on_step(_event):
    spu = int(sliders['SPU (steps/upd)'].val)
    model.step(n_sub=spu)
    _redraw()

def _on_reset(_event):
    global model
    playing[0] = False
    play_btn.label.set_text('Play')
    model = OpinionModel(
        M=DEFAULT_M,
        learning_rate=sliders['Learning rate'].val,
        confidence_threshold=sliders['Confidence thr.'].val,
        consensus_threshold=sliders['Consensus thr.'].val,
        interaction_mode=model.interaction_mode,
        local_int=model.local_int,
    )
    _redraw()

play_btn.on_clicked(_on_play)
step_btn.on_clicked(_on_step)
reset_btn.on_clicked(_on_reset)

plt.show()
