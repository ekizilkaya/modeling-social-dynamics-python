"""
Disease Spread GUI: SI, SIS, SIR Models
========================================
Interactive GUI for compartmental disease models from Ch 4 (Smaldino).

Agents move in continuous 2D space and interact via proximity. Three
epidemic models:
- **SI**: Susceptible → Infected (permanent).
- **SIS**: Susceptible ↔ Infected (recovery without immunity).
- **SIR**: Susceptible → Infected → Recovered (immune).

The GUI animates spread in real-time with adjustable parameters:
infection radius, transmission probability, recovery rate. Sliders
control rates, buttons reset/initialize, and the animation shows
state changes over time.
"""

# ## 1. Packages

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection

# ## 2. Agent definition

class Agent:
    """Agent in 2D continuous space with infection state."""
    def __init__(self, aid, pos, vel, infected=False, recovered=False,
                 transmission_rate=0.1, recovery_rate=0.01, speed=1.0, scale=0.7):
        self.id = aid
        self.pos = np.array(pos, dtype=float)    # [x, y]
        self.vel = np.array(vel, dtype=float)    # unit velocity [vx, vy]
        self.infected = infected
        self.recovered = recovered
        self.transmission_rate = transmission_rate
        self.recovery_rate = recovery_rate
        self.speed = speed
        self.scale = scale

# ## 3. Helper — random unit vector

def random_unit_vec():
    """Random unit velocity in 2D."""
    v = np.random.random(2) - 0.5
    n = np.linalg.norm(v)
    if n > 0:
        v /= n
    else:
        v = np.array([1.0, 0.0])
    return v

# ## 4. Model definition & dynamics

class SIModel:
    """SI / SIS / SIR model on continuous 2D toroidal space."""

    DISEASE_NAMES = {1: 'SI', 2: 'SIS', 3: 'SIR'}

    def __init__(self, n_agents=500, transmission_rate=0.10, recovery_rate=0.01,
                 spontaneous_infect=0.0, speed=1.0, scale=0.7,
                 extent=(100, 100), turning_angle=360.0,
                 disease_model='SI', initial_infected=1):
        self.extent = np.array(extent, dtype=float)     # PARAM
        self.turning_angle = turning_angle               # PARAM (degrees)
        self.speed = speed                               # PARAM
        self.transmission_rate = transmission_rate       # PARAM
        self.recovery_rate = recovery_rate               # PARAM
        self.spontaneous_infect = spontaneous_infect     # PARAM
        self.scale = scale
        self.disease_model_index = {'SI': 1, 'SIS': 2, 'SIR': 3}[disease_model]  # 1=SI, 2=SIS, 3=SIR
        self.max_infected_fraction = 0.0
        self.n_agents = n_agents

        # History for plots
        self.susceptible_history = []
        self.infected_history = []
        self.recovered_history = []
        self.max_inf_history = []

        # Create agents
        self.agents = {}
        for i in range(1, n_agents + 1):
            pos = np.random.random(2) * self.extent
            vel = random_unit_vec()
            infected = (i <= initial_infected)
            a = Agent(i, pos, vel, infected=infected, recovered=False,
                      transmission_rate=transmission_rate, recovery_rate=recovery_rate,
                      speed=speed, scale=scale)
            self.agents[i] = a

    # ----- Space helpers -----

    def _toroidal_dist(self, p1, p2):
        """Shortest distance on a torus."""
        delta = np.abs(p1 - p2)
        delta = np.minimum(delta, self.extent - delta)
        return np.linalg.norm(delta)

    def _nearby_agents(self, agent, radius=1.0):
        """All agents within toroidal radius (excluding self)."""
        return [a for a in self.agents.values()
                if a.id != agent.id and self._toroidal_dist(agent.pos, a.pos) < radius]

    def _wrap(self, pos):
        return pos % self.extent

    # ----- agent_step! -----

    def _step_agent(self, a):
        dm = self.disease_model_index
        susceptible = (not a.infected) and not (dm == 3 and a.recovered)

        if susceptible:
            k = sum(1 for n in self._nearby_agents(a, 1.0) if n.infected)
            p_inf = 1.0 - ((1.0 - self.transmission_rate) ** k) * (1.0 - self.spontaneous_infect)
            if np.random.random() < p_inf:
                a.infected = True

        # Move agent
        theta_max = np.deg2rad(self.turning_angle)
        d_theta = (np.random.random() - np.random.random()) * theta_max
        phi = np.arctan2(a.vel[1], a.vel[0]) + d_theta
        a.vel = np.array([np.cos(phi), np.sin(phi)])
        a.pos = self._wrap(a.pos + a.vel * self.speed)

    # ----- model_step! -----

    def _model_step(self):
        dm = self.disease_model_index
        infected_ids = [a.id for a in self.agents.values() if a.infected]

        if dm == 2:  # SIS — recovery back to susceptible
            for aid in infected_ids:
                a = self.agents[aid]
                if a.infected and np.random.random() < self.recovery_rate:
                    a.infected = False
                    a.recovered = False
        elif dm == 3:  # SIR — recovery to recovered
            for aid in infected_ids:
                a = self.agents[aid]
                if a.infected and np.random.random() < self.recovery_rate:
                    a.infected = False
                    a.recovered = True

        inf_frac = sum(1 for a in self.agents.values() if a.infected) / self.n_agents
        if inf_frac > self.max_infected_fraction:
            self.max_infected_fraction = inf_frac

    # ----- Full step -----

    def step(self):
        ids = list(self.agents.keys())
        np.random.shuffle(ids)
        for aid in ids:
            self._step_agent(self.agents[aid])
        self._model_step()
        # Record history
        n = self.n_agents
        self.susceptible_history.append(
            sum(1 for a in self.agents.values() if not a.infected and not a.recovered) / n)
        self.infected_history.append(
            sum(1 for a in self.agents.values() if a.infected) / n)
        self.recovered_history.append(
            sum(1 for a in self.agents.values() if a.recovered) / n)
        self.max_inf_history.append(self.max_infected_fraction)


# ## 5. GUI visualisation helpers

def make_triangle(pos, vel, scale):
    """Triangle vertices at *pos*, pointing along *vel*."""
    angle = np.arctan2(vel[1], vel[0])
    s = scale
    verts = np.array([[-s, -s],
                      [2 * s, 0],
                      [-s, s]])
    c, sn = np.cos(angle), np.sin(angle)
    R = np.array([[c, -sn],
                  [sn,  c]])
    return verts @ R.T + pos


def agent_color(a):
    """Red=infected, blue=recovered, grey=susceptible."""
    if a.infected:
        return (1.0, 0.0, 0.0)
    elif a.recovered:
        return (0.2, 0.4, 1.0)
    else:
        return (0.15, 0.15, 0.15)


def draw_world(ax, model):
    """Render all agents as coloured triangles."""
    ax.clear()
    ax.set_xlim(0, model.extent[0])
    ax.set_ylim(0, model.extent[1])
    ax.set_aspect('equal')
    ax.set_facecolor('white')
    ax.set_xticks([]); ax.set_yticks([])
    dm_name = SIModel.DISEASE_NAMES[model.disease_model_index]
    ax.set_title(f'{dm_name}  —  Step {len(model.infected_history)}', fontsize=10)

    patches, colors = [], []
    for a in model.agents.values():
        verts = make_triangle(a.pos, a.vel, a.scale)
        patches.append(MplPolygon(verts, closed=True))
        colors.append(agent_color(a))
    pc = PatchCollection(patches, match_original=False, edgecolors='none')
    pc.set_facecolors(colors)
    ax.add_collection(pc)


def draw_data(ax, model):
    """Plot S / I / R fractions and max-infected over time."""
    ax.clear()
    steps = range(len(model.infected_history))
    ax.plot(steps, model.susceptible_history, color='grey',   label='Susceptible', linewidth=1.2)
    ax.plot(steps, model.infected_history,    color='red',    label='Infected',    linewidth=1.2)
    ax.plot(steps, model.recovered_history,   color='royalblue', label='Recovered', linewidth=1.2)
    ax.plot(steps, model.max_inf_history,     color='orange', label='Max infected', linewidth=1, linestyle='--')
    ax.set_xlabel('Step')
    ax.set_ylabel('Fraction')
    ax.set_title('S / I / R fractions')
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right', fontsize=7)


# ## 6. GUI setup

model = SIModel(n_agents=500, disease_model='SI')   # PARAM

fig = plt.figure(figsize=(15, 9))
fig.patch.set_facecolor('#f0f0f0')

ax_world = fig.add_axes([0.02, 0.30, 0.50, 0.65])
ax_data  = fig.add_axes([0.57, 0.30, 0.40, 0.65])

draw_world(ax_world, model)
draw_data(ax_data, model)

# --- Sliders ---
slider_specs = [
    ('Disease (1=SI 2=SIS 3=SIR)', 0.22, 1, 3, 1, 1),
    ('Speed',             0.18, 0.1, 5.0, 0.1, 1.0),
    ('Transmission',      0.14, 0.0, 1.0, 0.01, 0.10),
    ('Recovery',          0.10, 0.0, 1.0, 0.01, 0.01),
    ('Spontaneous Inf.',  0.06, 0.0, 1.0, 0.01, 0.0),
    ('Turning Angle',     0.02, 0.0, 360.0, 15.0, 360.0),
]

sliders = {}
for label, y, vmin, vmax, vstep, vinit in slider_specs:
    ax_s = fig.add_axes([0.18, y, 0.50, 0.025])
    sliders[label] = Slider(ax_s, label, vmin, vmax, valinit=vinit, valstep=vstep)

# --- Buttons ---
ax_play  = fig.add_axes([0.74, 0.06, 0.08, 0.04])
ax_step  = fig.add_axes([0.83, 0.06, 0.08, 0.04])
ax_reset = fig.add_axes([0.74, 0.01, 0.08, 0.04])

play_btn  = Button(ax_play,  'Play')
step_btn  = Button(ax_step,  'Step')
reset_btn = Button(ax_reset, 'Reset')

playing = [False]

# --- Sync sliders → model ---
def _sync(_val):
    model.disease_model_index = int(sliders['Disease (1=SI 2=SIS 3=SIR)'].val)
    model.speed              = sliders['Speed'].val
    model.transmission_rate  = sliders['Transmission'].val
    model.recovery_rate      = sliders['Recovery'].val
    model.spontaneous_infect = sliders['Spontaneous Inf.'].val
    model.turning_angle      = sliders['Turning Angle'].val

for s in sliders.values():
    s.on_changed(_sync)

def _redraw():
    draw_world(ax_world, model)
    draw_data(ax_data, model)
    fig.canvas.draw_idle()

def _animate(_frame):
    if playing[0]:
        model.step()
        _redraw()

ani = animation.FuncAnimation(fig, _animate, interval=50, cache_frame_data=False)

# ## 7. Launch GUI & controls

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
    dm_idx = int(sliders['Disease (1=SI 2=SIS 3=SIR)'].val)
    dm_name = SIModel.DISEASE_NAMES[dm_idx]
    model = SIModel(
        n_agents=500,
        disease_model=dm_name,
        speed=sliders['Speed'].val,
        transmission_rate=sliders['Transmission'].val,
        recovery_rate=sliders['Recovery'].val,
        spontaneous_infect=sliders['Spontaneous Inf.'].val,
        turning_angle=sliders['Turning Angle'].val,
    )
    _redraw()

play_btn.on_clicked(_on_play)
step_btn.on_clicked(_on_step)
reset_btn.on_clicked(_on_reset)

plt.show()

