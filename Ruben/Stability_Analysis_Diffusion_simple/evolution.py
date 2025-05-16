import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from matplotlib.animation import FuncAnimation, PillowWriter

# ——— Parameter wie gehabt ———
D = 1.0
L = 1.0
N = 100
h = L/(N-1)
dt = 0.4*h**2/D
Tfinal = 0.1
nsteps = int(Tfinal/dt)
x = np.linspace(0, L, N)

# ——— Matrix A mit Neumann-BC ———
diag = -2.0*np.ones(N)
off  = 1.0*np.ones(N-1)
A = (np.diag(diag) + np.diag(off,1) + np.diag(off,-1)) / h**2
A[0,0], A[0,1]         = -1/h**2,  1/h**2
A[-1,-2], A[-1,-1]     =  1/h**2, -1/h**2

# ——— Downloads-Ordner ermitteln und Unterordner anlegen ———
home       = os.path.expanduser("~")
out_dir    = os.path.join(home, "Downloads", "DiffusionResults")
os.makedirs(out_dir, exist_ok=True)

# ——— GIF erzeugen ———
delta = 0.1 * np.sin(np.pi * x / L)
fig, ax = plt.subplots()
line, = ax.plot(x, delta)
ax.set_xlim(0, L)
ax.set_ylim(1.1*delta.min(), 1.1*delta.max())
ax.set_xlabel('x'); ax.set_ylabel(r'$\delta c$')

def animate(i):
    global delta
    delta += dt * D * (A @ delta)
    line.set_ydata(delta)
    return line,

ani = FuncAnimation(fig, animate, frames=nsteps, blit=True)
gif_path = os.path.join(out_dir, "diffusion.gif")
ani.save(gif_path, writer=PillowWriter(fps=30))
print(f"GIF gespeichert in: {gif_path}")

# ——— Sechs Snapshots erzeugen ———
times = np.linspace(0, Tfinal, 6)
snapshots = [expm(D*A*t) @ (0.1 * np.sin(np.pi * x / L)) for t in times]

fig2, axes = plt.subplots(2, 3, figsize=(9, 6), constrained_layout=True)
for ax, u, t in zip(axes.flat, snapshots, times):
    ax.plot(x, u)
    ax.set_title(f"t = {t:.3f}")
    ax.set_xlabel('x'); ax.set_ylabel(r'$\delta c$')

snap_path = os.path.join(out_dir, "diffusion_six_snapshots.png")
fig2.savefig(snap_path)
print(f"Snapshots gespeichert in: {snap_path}")
