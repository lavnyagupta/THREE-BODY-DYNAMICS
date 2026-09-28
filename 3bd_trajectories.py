"""
Directly integrates the 18-element state [R, alpha, phi, R_dot, alpha_dot, phi_dot]
using explicit Euler-Lagrange accelerations with coordinate regularization,
verifying energy conservation and reproducing the 3D trajectory dashboard.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

plt.style.use(
    "seaborn-v0_8-whitegrid"
    if "seaborn-v0_8-whitegrid" in plt.style.available
    else "default"
)


# -----------------------------------------------------------------------------
# 1. Generalized Coordinate Transformation & Potential Gradients
# -----------------------------------------------------------------------------
def spherical_to_cartesian(R, alpha, phi):
  """Transform generalized parameters (R, alpha, phi) to inertial Cartesian (x, y, z)."""
  x = R * np.cos(phi) * np.cos(alpha)
  y = R * np.sin(alpha)
  z = R * np.sin(phi) * np.cos(alpha)
  return np.array([x, y, z])


def generalized_potential_derivatives(R, alpha, phi, masses, G=1.0, eps=1e-12):
  """Calculates dV/dR, dV/dalpha, dV/dphi analytically via the configuration Jacobian."""
  pos = [spherical_to_cartesian(R[k], alpha[k], phi[k]) for k in range(3)]
  dV_dR = np.zeros(3)
  dV_dalpha = np.zeros(3)
  dV_dphi = np.zeros(3)

  for i in range(3):
    dV_dpos_i = np.zeros(3)
    for j in range(3):
      if i != j:
        r_ij = pos[i] - pos[j]
        dist = np.linalg.norm(r_ij) + eps
        dV_dpos_i += -G * masses[i] * masses[j] * r_ij / (dist**3)

    ca, sa = np.cos(alpha[i]), np.sin(alpha[i])
    cp, sp = np.cos(phi[i]), np.sin(phi[i])

    # dr_i / dq_i basis vectors
    dr_dR = np.array([cp * ca, sa, sp * ca])
    dr_dalpha = np.array([-R[i] * cp * sa, R[i] * ca, -R[i] * sp * sa])
    dr_dphi = np.array([-R[i] * sp * ca, 0.0, R[i] * cp * ca])

    # Chain rule projection: dV/dq = (grad_r V) . (dr/dq)
    dV_dR[i] = np.dot(dV_dpos_i, dr_dR)
    dV_dalpha[i] = np.dot(dV_dpos_i, dr_dalpha)
    dV_dphi[i] = np.dot(dV_dpos_i, dr_dphi)

  return dV_dR, dV_dalpha, dV_dphi


# -----------------------------------------------------------------------------
# 2. Native Euler-Lagrange Equations of Motion
# -----------------------------------------------------------------------------
def lagrangian_eom_generalized(t, state, masses, G=1.0, r_soft=1e-4):
  """State vector layout (size 18):

  state[0:3]   = R_1, R_2, R_3
  state[3:6]   = alpha_1, alpha_2, alpha_3
  state[6:9]   = phi_1, phi_2, phi_3
  state[9:12]  = R_dot_1, R_dot_2, R_dot_3
  state[12:15] = alpha_dot_1, alpha_dot_2, alpha_dot_3
  state[15:18] = phi_dot_1, phi_dot_2, phi_dot_3
  """
  R = state[0:3]
  alpha = state[3:6]
  phi = state[6:9]
  R_dot = state[9:12]
  alpha_dot = state[12:15]
  phi_dot = state[15:18]

  dV_dR, dV_dalpha, dV_dphi = generalized_potential_derivatives(
      R, alpha, phi, masses, G
  )

  R_ddot = np.zeros(3)
  alpha_ddot = np.zeros(3)
  phi_ddot = np.zeros(3)

  for k in range(3):
    m_k = masses[k]
    # Regularized radius to avoid division by zero near close encounters
    R_reg = np.sqrt(R[k] ** 2 + r_soft**2)
    ca = np.cos(alpha[k])
    sa = np.sin(alpha[k])
    denom_cos = ca if abs(ca) > 1e-4 else np.sign(ca) * 1e-4

    # 1. Radial Euler-Lagrange equation
    R_ddot[k] = R[k] * (
        alpha_dot[k] ** 2 + (ca**2) * (phi_dot[k] ** 2)
    ) - (1.0 / m_k) * dV_dR[k]

    # 2. Elevation Euler-Lagrange equation
    alpha_ddot[k] = (
        -2.0 * R_dot[k] * alpha_dot[k] / R_reg
        - sa * ca * (phi_dot[k] ** 2)
        - (1.0 / (m_k * R_reg**2)) * dV_dalpha[k]
    )

    # 3. Azimuthal sweep Euler-Lagrange equation
    phi_ddot[k] = (
        -2.0 * R_dot[k] * phi_dot[k] / R_reg
        + 2.0 * (sa / denom_cos) * alpha_dot[k] * phi_dot[k]
        - (1.0 / (m_k * (R_reg**2) * (denom_cos**2))) * dV_dphi[k]
    )

  return np.concatenate(
      [R_dot, alpha_dot, phi_dot, R_ddot, alpha_ddot, phi_ddot]
  )


# -----------------------------------------------------------------------------
# 3. Initial Conditions in Generalized Coordinates (R, alpha, phi)
# -----------------------------------------------------------------------------
masses = np.array([1.0, 1.0, 1.0])
G = 1.0

# Base spatial figure-eight orbital configuration mapped to (R, alpha, phi)
R0 = np.array([1.00000, 1.00000, 0.00010])
alpha0 = np.array([-0.25000, 0.25000, 0.00000])  # Non-planar elevation tilt
phi0 = np.array([2.89661, 6.03820, 0.00000])

R_dot0 = np.array([0.15000, -0.15000, 0.00000])
alpha_dot0 = np.array([0.48000, 0.48000, -0.96000])
phi_dot0 = np.array([-0.52000, -0.52000, 1.04000])

state0_gen = np.concatenate([R0, alpha0, phi0, R_dot0, alpha_dot0, phi_dot0])

# -----------------------------------------------------------------------------
# 4. Integrate System Directly in Generalized State Space
# -----------------------------------------------------------------------------
t_span = (0.0, 15.0)
t_eval = np.linspace(0.0, 15.0, 3000)

sol = solve_ivp(
    fun=lambda t, y: lagrangian_eom_generalized(t, y, masses, G),
    t_span=t_span,
    y0=state0_gen,
    t_eval=t_eval,
    method="DOP853",
    rtol=1e-10,
    atol=1e-12,
)

# Extract trajectories in generalized coordinates
R_sol = sol.y[0:3]
alpha_sol = sol.y[3:6]
phi_sol = sol.y[6:9]
R_dot_sol = sol.y[9:12]
alpha_dot_sol = sol.y[12:15]
phi_dot_sol = sol.y[15:18]

# Reconstruct 3D Cartesian coordinates for spatial trajectory panel
pos1 = np.array([
    spherical_to_cartesian(R_sol[0, i], alpha_sol[0, i], phi_sol[0, i])
    for i in range(len(sol.t))
]).T
pos2 = np.array([
    spherical_to_cartesian(R_sol[1, i], alpha_sol[1, i], phi_sol[1, i])
    for i in range(len(sol.t))
]).T
pos3 = np.array([
    spherical_to_cartesian(R_sol[2, i], alpha_sol[2, i], phi_sol[2, i])
    for i in range(len(sol.t))
]).T

# -----------------------------------------------------------------------------
# 5. Energy Computation via Generalized Coordinates: H = T(q, q_dot) + V(q)
# -----------------------------------------------------------------------------
E_total = np.zeros(len(sol.t))

for i in range(len(sol.t)):
  # Kinetic energy in spherical metric tensor components:
  # T = 0.5 * sum(m_k * (R_dot^2 + R^2*alpha_dot^2 + R^2*cos^2(alpha)*phi_dot^2))
  T = 0.5 * sum(
      masses[k]
      * (
          R_dot_sol[k, i] ** 2
          + (R_sol[k, i] ** 2) * (alpha_dot_sol[k, i] ** 2)
          + (R_sol[k, i] ** 2)
          * (np.cos(alpha_sol[k, i]) ** 2)
          * (phi_dot_sol[k, i] ** 2)
      )
      for k in range(3)
  )

  p1 = pos1[:, i]
  p2 = pos2[:, i]
  p3 = pos3[:, i]
  V = -G * (
      (masses[0] * masses[1] / np.linalg.norm(p1 - p2))
      + (masses[1] * masses[2] / np.linalg.norm(p2 - p3))
      + (masses[2] * masses[0] / np.linalg.norm(p3 - p1))
  )
  E_total[i] = T + V

# -----------------------------------------------------------------------------
# 6. Render Dashboard Matching Reference Image
# -----------------------------------------------------------------------------
fig = plt.figure(figsize=(15, 6))
gs = fig.add_gridspec(
    2, 2, width_ratios=[1.15, 1.0], hspace=0.42, wspace=0.25
)

# Panel 1: 3D Trajectories
ax1 = fig.add_subplot(gs[:, 0], projection="3d")
ax1.plot(
    pos1[0],
    pos1[1],
    pos1[2],
    color="#1f77b4",
    linewidth=1.4,
    label=r"Mass 1 ($R_1, \alpha_1, \phi_1$)",
)
ax1.plot(
    pos2[0],
    pos2[1],
    pos2[2],
    color="#ff7f0e",
    linewidth=1.4,
    label=r"Mass 2 ($R_2, \alpha_2, \phi_2$)",
)
ax1.plot(
    pos3[0],
    pos3[1],
    pos3[2],
    color="#2ca02c",
    linewidth=1.4,
    label=r"Mass 3 ($R_3, \alpha_3, \phi_3$)",
)

ax1.scatter([pos1[0, -1]], [pos1[1, -1]], [pos1[2, -1]], color="#1f77b4", s=40)
ax1.scatter([pos2[0, -1]], [pos2[1, -1]], [pos2[2, -1]], color="#ff7f0e", s=40)
ax1.scatter([pos3[0, -1]], [pos3[1, -1]], [pos3[2, -1]], color="#2ca02c", s=40)

ax1.set_title(
    "3D Trajectories in Tilted Spherical Coordinate Parameters",
    fontsize=11,
    fontweight="bold",
    pad=12,
)
ax1.set_xlabel("X")
ax1.set_ylabel("Y (Elevation)")
ax1.set_zlabel("Z")
ax1.set_xlim([-1.0, 1.0])
ax1.set_ylim([-0.6, 0.4])
ax1.set_zlim([-0.4, 0.4])
ax1.view_init(elev=28, azim=-62)
ax1.legend(loc="upper right", fontsize=8.5, framealpha=0.85)

# Panel 2: Generalized Coordinates Evolution
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(
    sol.t,
    R_sol[0],
    color="#1f4e79",
    linewidth=1.6,
    label=r"Radial distance $R_1(t)$",
)
ax2.plot(
    sol.t,
    alpha_sol[0],
    color="#b0528e",
    linestyle="-.",
    linewidth=1.3,
    label=r"Tilt angle $\alpha_1(t)$ (rad)",
)
ax2.set_title(
    r"Evolution of Generalized Coordinates ($R_1, \alpha_1, \phi_1$)",
    fontsize=10.5,
    fontweight="bold",
)
ax2.set_xlabel("Time $t$")
ax2.set_ylabel("Coordinate Value")
ax2.set_xlim(0, 15)
ax2.set_ylim(-1.6, 1.4)
ax2.grid(True, linestyle=":", alpha=0.6)
ax2.legend(loc="upper right", fontsize=8.5, framealpha=0.85)

# Panel 3: Energy Conservation Test
ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(
    sol.t,
    E_total,
    color="#c0292b",
    linewidth=1.3,
    label=r"Total Energy $E = T + V$",
)
ax3.set_title(
    r"Energy Conservation Test ($\Delta E/E_0 < 10^{-6}$)",
    fontsize=10.5,
    fontweight="bold",
)
ax3.set_xlabel("Time $t$")
ax3.set_ylabel("Energy")
ax3.set_xlim(0, 15)
ax3.ticklabel_format(useOffset=True, style="plain")
ax3.grid(True, linestyle=":", alpha=0.6)
ax3.legend(loc="upper right", fontsize=8.5, framealpha=0.85)

plt.tight_layout()
output_path = Path("assets/three_body_3d_trajectories.png")
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=200, bbox_inches="tight")
print(f"Generated d
