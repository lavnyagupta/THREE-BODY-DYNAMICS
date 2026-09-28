"""
Integrates the 9-DOF non-planar multi-body equations of motion in generalized
coordinates (R_k, alpha_k, phi_k) with nadir-locked attitude-orbit coupling,
and evaluates:
  1. Noether Total Angular Momentum Conservation (L_total = L_orb + L_rot)
  2. Chaotic Lyapunov Divergence (delta = 1e-7 perturbation)
  3. Lagrange L4/L5 Homographic Breathing Symmetries
  4. Coupled Attitude Nutation & Precession Rates
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Configure publication styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

# -----------------------------------------------------------------------------
# 1. Kinematics & Generalized Potential Force Projection
# -----------------------------------------------------------------------------
def spherical_to_cartesian(R, alpha, phi):
    """Map generalized coordinates (R, alpha, phi) to inertial Cartesian (x, y, z)."""
    x = R * np.cos(phi) * np.cos(alpha)
    y = R * np.sin(alpha)
    z = R * np.sin(phi) * np.cos(alpha)
    return np.array([x, y, z])

def generalized_potential_forces(R, alpha, phi, masses, G=1.0, eps=1e-12):
    """
    Project mutual 3D Newtonian gravitational forces onto generalized
    coordinates Q_q = -dV/dq via the kinematic configuration Jacobian.
    """
    pos = [spherical_to_cartesian(R[k], alpha[k], phi[k]) for k in range(3)]
    Q_R = np.zeros(3)
    Q_alpha = np.zeros(3)
    Q_phi = np.zeros(3)

    for i in range(3):
        F_total_i = np.zeros(3)
        for j in range(3):
            if i == j:
                continue
            r_vec = pos[j] - pos[i]
            dist = np.linalg.norm(r_vec) + eps
            F_total_i += G * masses[i] * masses[j] * r_vec / (dist**3)

        # Analytic Jacobian transpose: J_i^T = [dr_i/dR_i, dr_i/dalpha_i, dr_i/dphi_i]^T
        ca, sa = np.cos(alpha[i]), np.sin(alpha[i])
        cp, sp = np.cos(phi[i]), np.sin(phi[i])

        dr_dR = np.array([cp * ca, sa, sp * ca])
        dr_dalpha = np.array([-R[i] * cp * sa, R[i] * ca, -R[i] * sp * sa])
        dr_dphi = np.array([-R[i] * sp * ca, 0.0, R[i] * cp * ca])

        Q_R[i] = np.dot(dr_dR, F_total_i)
        Q_alpha[i] = np.dot(dr_dalpha, F_total_i)
        Q_phi[i] = np.dot(dr_dphi, F_total_i)

    return Q_R, Q_alpha, Q_phi

# -----------------------------------------------------------------------------
# 2. Coupled Lagrangian Equations of Motion (Euler-Lagrange with Attitude)
# -----------------------------------------------------------------------------
def custom_lagrangian_eom(t, state, masses, inertia_tensors, G=1.0):
    """
    Evaluates explicit accelerations [R_ddot, alpha_ddot, phi_ddot] derived
    from the coupled Lagrangian:
      L = T_trans(q, q_dot) + T_rot(omega(q, q_dot)) - V(q)
    """
    R = state[0:3]
    alpha = state[3:6]
    phi = state[6:9]
    R_dot = state[9:12]
    alpha_dot = state[12:15]
    phi_dot = state[15:18]

    Q_R, Q_alpha, Q_phi = generalized_potential_forces(R, alpha, phi, masses, G)

    R_ddot = np.zeros(3)
    alpha_ddot = np.zeros(3)
    phi_ddot = np.zeros(3)

    for k in range(3):
        m = masses[k]
        Ix, Iy, Iz = inertia_tensors[k]
        R_k = max(R[k], 1e-5)
        ca, sa = np.cos(alpha[k]), np.sin(alpha[k])

        # 1. Radial Equation of Motion: d/dt(m R_dot) = dL/dR
        # dL/dR includes centrifugal terms from orbital sweeps
        R_ddot[k] = R_k * (alpha_dot[k]**2 + (ca**2) * (phi_dot[k]**2)) + (1.0 / m) * Q_R[k]

        # 2. Elevation Equation of Motion with Nadir-Locked Inertia:
        # Generalized momentum: p_alpha = (m R^2 + Iy) * alpha_dot
        M_alpha = m * (R_k**2) + Iy
        coriolis_alpha = -2.0 * m * R_k * R_dot[k] * alpha_dot[k]
        centrifugal_alpha = (Ix - Iz - m * (R_k**2)) * sa * ca * (phi_dot[k]**2)
        alpha_ddot[k] = (coriolis_alpha + centrifugal_alpha + Q_alpha[k]) / M_alpha

        # 3. Azimuthal Equation of Motion with Nadir-Locked Inertia:
        # Generalized momentum: p_phi = J_phi * phi_dot
        # where J_phi = (m R^2 + Iz) cos^2(alpha) + Ix sin^2(alpha)
        J_phi = (m * (R_k**2) + Iz) * (ca**2) + Ix * (sa**2)
        dJ_dt = 2.0 * m * R_k * R_dot[k] * (ca**2) + 2.0 * (Ix - Iz - m * (R_k**2)) * sa * ca * alpha_dot[k]
        phi_ddot[k] = (-dJ_dt * phi_dot[k] + Q_phi[k]) / J_phi

    return np.concatenate([R_dot, alpha_dot, phi_dot, R_ddot, alpha_ddot, phi_ddot])

# -----------------------------------------------------------------------------
# 3. Simulation Parameters & Physical Constants
# -----------------------------------------------------------------------------
masses = np.array([1.0, 1.0, 1.0])
# Asymmetrical inertia tensors [Ix, Iy, Iz] per body
inertias = np.array([
    [0.05, 0.10, 0.15],
    [0.05, 0.10, 0.15],
    [0.05, 0.10, 0.15]
])

# Base non-planar initial conditions in generalized coordinates (R, alpha, phi)
R0 = np.array([1.0, 1.05, 0.95])
alpha0 = np.array([0.15, -0.10, 0.05])       # Non-planar elevation tilts
phi0 = np.array([0.0, 2.0944, 4.1888])       # ~120 deg spatial separation

R_dot0 = np.array([0.0, 0.02, -0.01])
alpha_dot0 = np.array([0.05, -0.03, 0.02])
phi_dot0 = np.array([0.85, 0.82, 0.88])

state_base = np.concatenate([R0, alpha0, phi0, R_dot0, alpha_dot0, phi_dot0])

# =============================================================================
# Test 1: Vector Conservation of Total Angular Momentum L (Orbit + Spin)
# =============================================================================
t_eval1 = np.linspace(0, 30, 1800)
sol1 = solve_ivp(
    custom_lagrangian_eom, (0, 30), state_base,
    args=(masses, inertias), t_eval=t_eval1, method='DOP853',
    rtol=1e-10, atol=1e-12
)

L_total_series = np.zeros((len(sol1.t), 3))
for step in range(len(sol1.t)):
    st = sol1.y[:, step]
    R_s, a_s, p_s = st[0:3], st[3:6], st[6:9]
    R_d, a_d, p_d = st[9:12], st[12:15], st[15:18]

    L_t = np.zeros(3)
    for k in range(3):
        pos_k = spherical_to_cartesian(R_s[k], a_s[k], p_s[k])
        # Cartesian velocity transformation via differential chain rule
        ca, sa = np.cos(a_s[k]), np.sin(a_s[k])
        cp, sp = np.cos(p_s[k]), np.sin(p_s[k])
        vx = R_d[k]*cp*ca - R_s[k]*p_d[k]*sp*ca - R_s[k]*a_d[k]*cp*sa
        vy = R_d[k]*sa + R_s[k]*a_d[k]*ca
        vz = R_d[k]*sp*ca + R_s[k]*p_d[k]*cp*ca - R_s[k]*a_d[k]*sp*sa
        vel_k = np.array([vx, vy, vz])

        # Orbital angular momentum
        L_orb = masses[k] * np.cross(pos_k, vel_k)

        # Coupled attitude spin momentum projected into inertial frame
        Ix, Iy, Iz = inertias[k]
        omega_k = np.array([p_d[k] * sa, a_d[k], p_d[k] * ca])
        L_rot = np.array([Ix * omega_k[0], Iy * omega_k[1], Iz * omega_k[2]])

        L_t += (L_orb + L_rot)

    L_total_series[step] = L_t

L0_norm = np.linalg.norm(L_total_series[0])
L_relative_error = np.linalg.norm(L_total_series - L_total_series[0], axis=1) / L0_norm

# =============================================================================
# Test 2: Chaotic Lyapunov Divergence in Generalized Phase Space (delta = 1e-7)
# =============================================================================
delta_pert = 1e-7
state_pert = state_base.copy()
state_pert[0] += delta_pert  # Perturb radial distance R_1

t_eval2 = np.linspace(0, 25, 1500)
sol2_ref = solve_ivp(
    custom_lagrangian_eom, (0, 25), state_base,
    args=(masses, inertias), t_eval=t_eval2, method='DOP853',
    rtol=1e-10, atol=1e-12
)
sol2_pert = solve_ivp(
    custom_lagrangian_eom, (0, 25), state_pert,
    args=(masses, inertias), t_eval=t_eval2, method='DOP853',
    rtol=1e-10, atol=1e-12
)

# Phase-space metric separation in generalized coordinates
diff_generalized = np.linalg.norm(sol2_ref.y[0:9] - sol2_pert.y[0:9], axis=0)

# =============================================================================
# Test 3: Lagrange L4/L5 Homographic Breathing Symmetries
# =============================================================================
# Initialize equilateral breathing mode: equal radii R, planar alpha=0, symmetric sweep rates
r_peri = 1.74
R_eq = r_peri / np.sqrt(3.0)
phi_dot_eq = 0.90 * np.sqrt(1.0 / (R_eq**3))  # Sub-circular rate to excite breathing mode

state_lagrange = np.array([
    R_eq, R_eq, R_eq,                      # R1, R2, R3
    0.0, 0.0, 0.0,                         # alpha1, alpha2, alpha3 (planar)
    0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0,  # phi1, phi2, phi3 (120 deg offsets)
    0.0, 0.0, 0.0,                         # R_dot
    0.0, 0.0, 0.0,                         # alpha_dot
    phi_dot_eq, phi_dot_eq, phi_dot_eq     # phi_dot
])

t_eval3 = np.linspace(0, 20, 1000)
sol3 = solve_ivp(
    custom_lagrangian_eom, (0, 20), state_lagrange,
    args=(masses, inertias), t_eval=t_eval3, method='DOP853',
    rtol=1e-10, atol=1e-12
)

r12 = np.zeros(len(sol3.t))
r23 = np.zeros(len(sol3.t))
r31 = np.zeros(len(sol3.t))

for i in range(len(sol3.t)):
    st = sol3.y[:, i]
    p1 = spherical_to_cartesian(st[0], st[3], st[6])
    p2 = spherical_to_cartesian(st[1], st[4], st[7])
    p3 = spherical_to_cartesian(st[2], st[5], st[8])
    r12[i] = np.linalg.norm(p1 - p2)
    r23[i] = np.linalg.norm(p2 - p3)
    r31[i] = np.linalg.norm(p3 - p1)

# =============================================================================
# Test 4: Asymmetrical Attitude Nutation & Precession Rates
# =============================================================================
# Extract dynamic attitude angular rates for Body 1 from generalized velocities
alpha1_series = sol1.y[3]
alpha_dot1_series = sol1.y[13]
phi_dot1_series = sol1.y[16]

omega_x = phi_dot1_series * np.sin(alpha1_series)       # Roll Rate
omega_y = alpha_dot1_series                            # Yaw / Elevation Rate
omega_z = phi_dot1_series * np.cos(alpha1_series)       # Pitch / Sweep Rate

# =============================================================================
# 4. Generate the 4-Panel Verification Dashboard
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Advanced Physical & Mathematical Verification Tests for 3D Three-Body Solution', fontsize=14, fontweight='bold')

# --- Panel 1: Vector L Conservation ---
axes[0, 0].plot(sol1.t, L_relative_error, color='#1f77b4', linewidth=1.5, label='Relative Vector Error')
axes[0, 0].set_title('Test 1: Vector Conservation of Total Angular Momentum L', fontsize=11, fontweight='bold')
axes[0, 0].set_xlabel('Time t')
axes[0, 0].set_ylabel('Relative Angular Momentum Error')
axes[0, 0].set_yscale('log')
axes[0, 0].set_xlim(0, 30)
axes[0, 0].grid(True, linestyle=':', alpha=0.6)
axes[0, 0].legend(loc='upper left')

# --- Panel 2: Lyapunov Divergence ---
axes[0, 1].plot(sol2_ref.t, diff_generalized, color='#b22222', linewidth=1.8, label=r'Trajectory Distance $\|q_1(t) - q_2(t)\|$')
axes[0, 1].axhline(delta_pert, color='gray', linestyle=':', label=r'Initial Perturbation $\delta = 10^{-7}$')
axes[0, 1].set_title(r'Test 2: Butterfly Effect & Chaotic Divergence ($\delta = 10^{-7}$)', fontsize=11, fontweight='bold')
axes[0, 1].set_xlabel('Time t')
axes[0, 1].set_ylabel('Phase-Space Distance')
axes[0, 1].set_yscale('log')
axes[0, 1].set_xlim(0, 25)
axes[0, 1].grid(True, linestyle=':', alpha=0.6)
axes[0, 1].legend(loc='upper left')

# --- Panel 3: Lagrange Breathing Symmetries ---
axes[1, 0].plot(sol3.t, r12, color='#1f77b4', linewidth=1.8, label=r'Side $r_{12}$')
axes[1, 0].plot(sol3.t, r23, color='#b58900', linestyle='--', linewidth=1.5, label=r'Side $r_{23}$')
axes[1, 0].plot(sol3.t, r31, color='#2aa198', linestyle=':', linewidth=1.5, label=r'Side $r_{31}$')
axes[1, 0].set_title(r'Test 3: Lagrange $L_4/L_5$ Equilateral Configuration Preservation', fontsize=11, fontweight='bold')
axes[1, 0].set_xlabel('Time t')
axes[1, 0].set_ylabel(r'Inter-Mass Distance $r_{ij}$')
axes[1, 0].set_xlim(0, 20)
axes[1, 0].set_ylim(1.7, 2.2)
axes[1, 0].grid(True, linestyle=':', alpha=0.6)
axes[1, 0].legend(loc='upper right')

# --- Panel 4: Attitude Nutation Rates ---
axes[1, 1].plot(sol1.t, omega_x, color='#1f77b4', linewidth=1.6, label=r'$\omega_x(t)$ (Roll Rate)')
axes[1, 1].plot(sol1.t, omega_y, color='#d97706', linewidth=1.6, label=r'$\omega_y(t)$ (Yaw Rate)')
axes[1, 1].plot(sol1.t, omega_z, color='#10b981', linewidth=1.6, label=r'$\omega_z(t)$ (Pitch Rate)')
axes[1, 1].set_title('Test 4: Asymmetrical Rigid-Body Precession & Nutation', fontsize=11, fontweight='bold')
axes[1, 1].set_xlabel('Time t')
axes[1, 1].set_ylabel('Angular Velocity Components')
axes[1, 1].set_xlim(0, 30)
axes[1, 1].grid(True, linestyle=':', alpha=0.6)
axes[1, 1].legend(loc='upper right')

plt.tight_layout()
plt.savefig('assets/verification_tests.png', dpi=200, bbox_inches='tight')
print("Verification dashboard successfully generated: assets/verification_tests.png")
plt.show()
