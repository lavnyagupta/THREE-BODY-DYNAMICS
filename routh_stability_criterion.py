import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# -----------------------------------------------------------------------------
# 1. Routh's Stability Threshold
# -----------------------------------------------------------------------------
mu_routh = 0.5 * (1.0 - np.sqrt(69.0) / 9.0)
print(f"Routh's Critical Mass Ratio: mu_Routh = {mu_routh:.6f}")

# -----------------------------------------------------------------------------
# 2. Full Non-Linear CR3BP Equations of Motion
# -----------------------------------------------------------------------------
def cr3bp_eom(t, state, mu, eps=1e-12):
    """
    Full non-linear equations of motion in the synodic (rotating) frame.
    state = [x, y, vx, vy]
    """
    x, y, vx, vy = state
    r1 = np.sqrt((x + mu)**2 + y**2) + eps
    r2 = np.sqrt((x - 1.0 + mu)**2 + y**2) + eps
    
    # Gravitational and centrifugal accelerations
    ax = 2.0 * vy + x - (1.0 - mu) * (x + mu) / (r1**3) - mu * (x - 1.0 + mu) / (r2**3)
    ay = -2.0 * vx + y - (1.0 - mu) * y / (r1**3) - mu * y / (r2**3)
    
    return [vx, vy, ax, ay]

# -----------------------------------------------------------------------------
# 3. Numerical Integration Setup
# -----------------------------------------------------------------------------
t_span = (0, 80)
t_eval = np.linspace(0, 80, 2500)
delta_0 = 1e-3  # Initial displacement perturbation

def run_simulation(mu_val):
    x_l4 = 0.5 - mu_val
    y_l4 = np.sqrt(3.0) / 2.0
    initial_state = [x_l4 + delta_0, y_l4, 0.0, 0.0]
    
    sol = solve_ivp(cr3bp_eom, t_span, initial_state, args=(mu_val,),
                    t_eval=t_eval, method='DOP853', rtol=1e-10, atol=1e-12)
    
    # Relative coordinates from L4
    delta_x = sol.y[0] - x_l4
    delta_y = sol.y[1] - y_l4
    distance = np.sqrt(delta_x**2 + delta_y**2)
    return sol.t, delta_x, delta_y, distance

# Run Sub-critical and Super-critical scenarios
mu_sub = 0.01215
mu_super = 0.05
t_sub, dx_sub, dy_sub, dist_sub = run_simulation(mu_sub)
t_super, dx_super, dy_super, dist_super = run_simulation(mu_super)

# -----------------------------------------------------------------------------
# 4. Visualization Matching Benchmark Figure
# -----------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle(f"Routh Stability Criterion Test: Triangular Libration (L4) Stability Threshold at mu = {mu_routh:.4f}",
             fontsize=13, fontweight='bold')

# Left Subplot: Distance Deviation vs Time (Log Scale)
ax1.plot(t_sub, dist_sub, color='#1f4e79', linewidth=1.8, label=f'Stable: mu = {mu_sub} (< {mu_routh:.4f})')
ax1.plot(t_super, dist_super, color='#c55a11', linewidth=1.8, label=f'Unstable: mu = {mu_super} (> {mu_routh:.4f})')
ax1.axhline(delta_0, color='gray', linestyle='--', linewidth=1.2, label=f'Initial Perturbation ({delta_0})')
ax1.set_yscale('log')
ax1.set_xlim(0, 80)
ax1.set_title("Deviation from L4 Point vs Time", fontsize=11, fontweight='bold')
ax1.set_xlabel("Nondimensional Time (t)")
ax1.set_ylabel("Distance from L4 Point")
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.legend(loc='upper left', fontsize=9)

# Right Subplot: Phase-Space Trajectories in Rotating Frame
ax2.plot(dx_sub, dy_sub, color='#1f4e79', linewidth=1.5, label=f'Stable Libration (mu = {mu_sub})')
ax2.plot(dx_super, dy_super, color='#c55a11', linewidth=1.2, label=f'Exponential Departure (mu = {mu_super})')
ax2.plot(0, 0, 'k*', markersize=10, label='L4 Equilibrium Point')
ax2.set_title("Phase Space Trajectories Relative to L4", fontsize=11, fontweight='bold')
ax2.set_xlabel("Delta x (Rotating Frame)")
ax2.set_ylabel("Delta y (Rotating Frame)")
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.legend(loc='upper right', fontsize=9)

plt.tight_layout()
plt.savefig('assets/routh_stability_criterion_test.png', dpi=200, bbox_inches='tight')
plt.show()
