# Temporal Variation of Caspofungin Concentration in the Zone of Inhibition
import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange

c0 = float(input("What is the initial concenteration? "))
max_radius = float(input("What is the max_radius ?(mm) "))
D = float(input("What is D(m2/s)?  "))
axis_label_font_size = float(input("axis_label_font_size? "))
axis_numbers_font_size = float(input("axis_numbers_font_size? "))
plot_title_font_size = float(input("plot_title_font_size? "))
# File path (update accordingly)
path = "Path"

# Plate size (mm)
w = h = 85.0
# Intervals in x-, y-directions (mm)
dx = dy = 0.1
# Grid size
nx, ny = int(w / dx), int(h / dy)
# Time step calculation using stability condition
dx2, dy2 = dx * dx, dy * dy
# Initial conditions: Circle of radius r centered at (cx, cy) (mm)
r, cx, cy = 3.5, 42.5, 42.5
r2 = r ** 2
# Create meshgrid for coordinates
x = np.linspace(0, w, nx)
y = np.linspace(0, h, ny)
xx, yy = np.meshgrid(x, y, indexing='ij')


@njit(fastmath=True, parallel=True)
def do_timestep(u0, u, D, dt, dx2, dy2):
    # Parallelized loops using prange for faster performance
    for i in prange(1, u0.shape[0] - 1):
        for j in prange(1, u0.shape[1] - 1):
            u[i, j] = (
                u0[i, j]
                + D * dt * (
                    (u0[i+1, j] - 2*u0[i, j] + u0[i-1, j]) / dx2
                    + (u0[i, j+1] - 2*u0[i, j] + u0[i, j-1]) / dy2
                )
            )
    # In-place update
    u0[:, :] = u[:, :]
    return u0, u


max_steps = int(1e5)
D_mm = D * 1e6  # Convert from m²/s to mm²/s
dt = (dx**2 * dy**2) / (2 * D_mm * (dx**2 + dy**2))
time_steps_per_hour = int(3600 / dt)  # Steps per hour

u0 = np.zeros((nx, ny))
u = u0.copy()
mask = ((xx - cx)**2 + (yy - cy)**2) < r2
u0[mask] = c0

nsteps = min(int((48 * 3600) / dt), max_steps)

for step in range(nsteps):
    u0, u = do_timestep(u0, u, D_mm, dt, dx**2, dy**2)
    if step % time_steps_per_hour == 0:
        radius = np.linspace(0,max_radius, int(max_radius/dx))
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(radius, u[int(cx/dx), int(cx/dx):int(cx/dx)+int(max_radius/dx)])
        ax.set_xlabel("Radius (mm)", fontsize = axis_label_font_size, fontweight='bold')
        ax.set_ylabel(r"Concentration ($\frac{\mu g}{ml})$" , fontsize = axis_label_font_size, fontweight='bold')
        ax.set_ylim([0, c0+0.1*c0])
        ax.set_title('{} hour'.format(int(step/time_steps_per_hour)) , fontsize = plot_title_font_size, fontweight='bold' )
        ax.tick_params(axis='both', labelsize=axis_label_font_size)  # Adjust labelsize as needed
        plt.savefig(path + "{}_his.png".format(int(step/time_steps_per_hour)), dpi = 400)
        plt.close(fig)
