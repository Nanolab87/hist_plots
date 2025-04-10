# Temporal Variation of Caspofungin Concentration in the Zone of Inhibition
import numpy as np
from numba import njit, prange

c0 = float(input("What is the initial concenteration? "))
final_radius = float(input("What is the final radius concenteration?(mm) "))
final_c = float(input("What is the final concenteration in the above radius? "))
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


best_D = None
min_diff = float('inf')
max_steps = int(1e5)

for D in np.logspace(-5, -25, 200):
    D_mm = D * 1e6  # Convert from m²/s to mm²/s
    dt = (dx**2 * dy**2) / (2 * D_mm * (dx**2 + dy**2))
    
    u0 = np.zeros((nx, ny))
    u = u0.copy()
    mask = ((xx - cx)**2 + (yy - cy)**2) < r2
    u0[mask] = c0

    nsteps = min(int((48 * 3600) / dt), max_steps)
    
    for _ in range(nsteps):
        u0, u = do_timestep(u0, u, D_mm, dt, dx**2, dy**2)

    tolerance = 1  # Allow a small tolerance due to grid resolution
    distance_mask = np.abs(np.sqrt((xx - 42.5)**2 + (yy - 42.5)**2) - final_radius) <= tolerance
    avg_concentration = np.mean(u[distance_mask])
    diff = abs(avg_concentration - final_c)

    if diff < min_diff:
        min_diff = diff
        best_D = D

    print(D, "   ", round(diff,6))
print(f"Best Diffusion Coefficient: {best_D:.3e} m²/s")
print(f"Difference from Target Concentration: {min_diff:.6f}")
