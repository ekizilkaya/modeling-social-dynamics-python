"""
Particle Simulation GUI
========================
Interactive GUI for the particle collision model from Ch 2 (Smaldino).

Particles move in continuous 2D space with elastic collisions. Each
particle has position, velocity, and radius. Collisions conserve
momentum and energy. The GUI animates particle motion in real-time
with adjustable parameters: number of particles, speed, radius.

Sliders control particle count, speed, and radius. Buttons reset/
initialize the simulation. Demonstrates emergent patterns from simple
physical rules, like clustering or phase separation.
"""

# ## 1. Packages

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button

# ## 2. Agent definition

# Define the Particle agent over a continuous 2D space
class Particle:
    """A particle agent with position, velocity, scale, and color."""
    def __init__(self, pid, pos, vel, scale, color):
        self.id = pid
        self.pos = np.array(pos, dtype=float)   # Position [x, y]
        self.vel = np.array(vel, dtype=float)   # Velocity unit vector [vx, vy]
        self.scale = scale                       # Visual scale/size
        self.color = color                       # Drawing color (R, G, B)

# ## 3. Model initialization & 4. Dynamics — agent step

class ParticleModel:
    """Particle model with continuous 2D toroidal space."""

    def __init__(self, n_particle=100, speed=1.0, whimsy=0.0, scale=0.7,
                 extent=(100, 100), flocking=0.0, vision_radius=5.0):
        # Continuous 2D toroidal space
        self.extent = np.array(extent, dtype=float)  # Periodic boundary conditions  # PARAM
        # Global model settings
        self.speed = speed                      # Movement speed
        self.whimsy = whimsy                    # Random turn parameter (degrees)
        self.flocking = flocking                # Alignment strength [0, 1]
        self.vision_radius = vision_radius      # Neighbor detection radius
        self.collisions = 0                     # Total collision count
        self.collision_history = []
        self.particles = []

        # Add each particle with random position, velocity, and color
        for i in range(n_particle):
            pos = np.random.random(2) * self.extent
            theta = 2 * np.pi * np.random.random()              # Random angle for velocity
            vel = np.array([np.cos(theta), np.sin(theta)])       # Initial velocity as unit vector
            color = tuple(np.random.random(3))                   # Random RGB color
            self.particles.append(Particle(i, pos, vel, scale, color))

    # ----- Space helpers -----

    def _toroidal_dist(self, p1, p2):
        """Shortest distance on a torus."""
        delta = np.abs(p1 - p2)
        delta = np.minimum(delta, self.extent - delta)
        return np.linalg.norm(delta)

    def _neighbors(self, particle, radius):
        """All particles within toroidal radius (excluding self)."""
        return [p for p in self.particles
                if p.id != particle.id
                and self._toroidal_dist(particle.pos, p.pos) < radius]

    def _wrap(self, pos):
        """Wrap position into toroidal space."""
        return pos % self.extent

    # ----- Dynamics (agent_step!) -----

    def _step_particle(self, particle):
        """Advance a single particle: flocking → whimsy → move → collisions."""

        # --- Flocking: align velocity with nearby neighbors ---
        if self.flocking > 0:
            nbs = self._neighbors(particle, self.vision_radius)
            if nbs:
                acc_vel = np.zeros(2)
                for nb in nbs:
                    acc_vel += nb.vel
                # Compute average heading
                avg_vel = acc_vel / len(nbs)
                nv = np.linalg.norm(avg_vel)
                if nv > 0:
                    direction = avg_vel / nv                            # Unit mean direction
                    # Blend current vel with flock direction
                    blend = (1 - self.flocking) * particle.vel + self.flocking * direction
                    bn = np.linalg.norm(blend)
                    if bn > 0:
                        particle.vel = blend / bn                       # Update heading

        # --- Whimsy: random turn ---
        d_theta = ((np.random.random() * self.whimsy
                     - np.random.random() * self.whimsy)
                    * (np.pi / 180))                                    # Random turn in radians
        theta0 = np.arctan2(particle.vel[1], particle.vel[0])          # Current heading angle
        theta1 = theta0 + d_theta                                      # New heading angle
        particle.vel = np.array([np.cos(theta1), np.sin(theta1)])      # Apply turn

        # --- Move particle forward ---
        particle.pos = self._wrap(particle.pos + particle.vel * self.speed)

        # --- Collision detection & scattering ---
        neighbors = self._neighbors(particle, 1.0)
        if neighbors:
            # Scatter self
            th = np.random.randint(0, 360) * (np.pi / 180)
            particle.vel = np.array([np.cos(th), np.sin(th)])
            particle.pos = self._wrap(particle.pos + particle.vel * 0.5)

            # Scatter all neighbors in-radius
            for nb in neighbors:
                th_n = np.random.randint(0, 360) * (np.pi / 180)
                nb.vel = np.array([np.cos(th_n), np.sin(th_n)])
                nb.pos = self._wrap(nb.pos + nb.vel * 0.5)

            self.collisions += 1

    def step(self):
        """One model step: advance all particles in random order."""
        for idx in np.random.permutation(len(self.particles)):
            self._step_particle(self.particles[idx])
        self.collision_history.append(self.collisions)

# ## 5. GUI visualization helper

def make_triangle(pos, vel, scale):
    """Triangle vertices at *pos*, pointing along *vel*."""
    angle = np.arctan2(vel[1], vel[0])
    s = scale
    # Back-left, front, back-right  (same geometry as the Julia version)
    verts = np.array([[-s, -s],
                      [2 * s, 0],
                      [-s, s]])
    c, sn = np.cos(angle), np.sin(angle)
    R = np.array([[c, -sn],
                  [sn,  c]])
    return verts @ R.T + pos


def draw_world(ax, model):
    """Render all particles as coloured triangles."""
    ax.clear()
    ax.set_xlim(0, model.extent[0])
    ax.set_ylim(0, model.extent[1])
    ax.set_aspect('equal')
    ax.set_facecolor('black')
    ax.set_xticks([])
    ax.set_yticks([])

    patches, colors = [], []
    for p in model.particles:
        verts = make_triangle(p.pos, p.vel, p.scale)
        patches.append(MplPolygon(verts, closed=True))
        colors.append(p.color)

    pc = PatchCollection(patches, match_original=False, edgecolors='none')
    pc.set_facecolors(colors)
    ax.add_collection(pc)

# ## 6. Launch GUI & controls

# --- Initialize model ---
model = ParticleModel(
    n_particle=100,    # PARAM
    scale=0.7,         # PARAM
    extent=(100, 100)  # PARAM
)

# --- Figure layout ---
fig = plt.figure(figsize=(14, 8))
fig.patch.set_facecolor('#2b2b2b')

# Main axes: world + data plot
ax_world = fig.add_axes([0.02, 0.25, 0.55, 0.70])
ax_data  = fig.add_axes([0.62, 0.25, 0.35, 0.70])

draw_world(ax_world, model)
ax_data.set_xlabel('Step')
ax_data.set_ylabel('Collisions')
ax_data.set_title('Cumulative Collisions')

# --- Parameter sliders ---
ax_speed    = fig.add_axes([0.12, 0.16, 0.30, 0.03])
ax_whimsy   = fig.add_axes([0.12, 0.11, 0.30, 0.03])
ax_flocking = fig.add_axes([0.58, 0.16, 0.30, 0.03])
ax_vision   = fig.add_axes([0.58, 0.11, 0.30, 0.03])

speed_slider    = Slider(ax_speed,    'Speed',    0.1, 2.0,   valinit=1.0, valstep=0.1)
whimsy_slider   = Slider(ax_whimsy,   'Whimsy',   0.0, 359.0, valinit=0.0, valstep=10.0)
flocking_slider = Slider(ax_flocking, 'Flocking', 0.0, 1.0,   valinit=0.0, valstep=0.1)
vision_slider   = Slider(ax_vision,   'Vision R',  1.0, 20.0,  valinit=5.0, valstep=1.0)

# --- Buttons ---
ax_play  = fig.add_axes([0.12, 0.03, 0.12, 0.05])
ax_step  = fig.add_axes([0.26, 0.03, 0.12, 0.05])
ax_reset = fig.add_axes([0.40, 0.03, 0.12, 0.05])

play_btn  = Button(ax_play,  'Play')
step_btn  = Button(ax_step,  'Step')
reset_btn = Button(ax_reset, 'Reset')

playing = [False]  # mutable flag for play/pause toggle

# --- Sync sliders -> model properties ---
def _sync(_val):
    model.speed         = speed_slider.val
    model.whimsy        = whimsy_slider.val
    model.flocking      = flocking_slider.val
    model.vision_radius = vision_slider.val

for s in (speed_slider, whimsy_slider, flocking_slider, vision_slider):
    s.on_changed(_sync)

# --- Redraw helper ---
def _redraw():
    draw_world(ax_world, model)
    ax_data.clear()
    ax_data.plot(model.collision_history, color='orange', linewidth=1.5)
    ax_data.set_xlabel('Step')
    ax_data.set_ylabel('Collisions')
    ax_data.set_title('Cumulative Collisions')
    fig.canvas.draw_idle()

# --- Animation loop (runs while Play is toggled on) ---
def _animate(_frame):
    if playing[0]:
        model.step()
        _redraw()

ani = animation.FuncAnimation(fig, _animate, interval=50, cache_frame_data=False)

# --- Button callbacks ---
def _on_play(_event):
    playing[0] = not playing[0]
    ax_play.texts.clear() if hasattr(ax_play, 'texts') else None
    play_btn.label.set_text('Pause' if playing[0] else 'Play')
    fig.canvas.draw_idle()

def _on_step(_event):
    model.step()
    _redraw()

def _on_reset(_event):
    global model
    playing[0] = False
    play_btn.label.set_text('Play')
    model = ParticleModel(
        n_particle=100, scale=0.7, extent=(100, 100),
        speed=speed_slider.val, whimsy=whimsy_slider.val,
        flocking=flocking_slider.val, vision_radius=vision_slider.val
    )
    _redraw()

play_btn.on_clicked(_on_play)
step_btn.on_clicked(_on_step)
reset_btn.on_clicked(_on_reset)

plt.show()
