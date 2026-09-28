import numpy as np


def spherical_to_cartesian(R, alpha, phi):
    """Convert tilted spherical coordinates (R, alpha, phi) to Cartesian (x, y, z)[cite: 1]."""
    x = R * np.cos(phi) * np.cos(alpha)
    y = R * np.sin(alpha)
    z = R * np.sin(phi) * np.cos(alpha)
    return np.array([x, y, z])


def compute_euclidean_distances(R, alpha, phi):
    """Compute pairwise 3D Euclidean distances between the 3 bodies[cite: 1].

    Parameters:
        R, alpha, phi : array-like of length 3 (one value per body)[cite: 1]

    Returns:
        r12, r23, r31 : Mutual separation distances[cite: 1]
    """
    pos = [spherical_to_cartesian(R[k], alpha[k], phi[k]) for k in range(3)]  #[cite: 1]
    r12 = np.linalg.norm(pos[0] - pos[1])
    r23 = np.linalg.norm(pos[1] - pos[2])
    r31 = np.linalg.norm(pos[2] - pos[0])
    return r12, r23, r31


def potential_gradients(R, alpha, phi, masses, G=1.0, eps=1e-12):
    """Numerical or analytical gradients of gravitational potential V w.r.t (R_k, alpha_k, phi_k)[cite: 1]."""
    dV_dR = np.zeros(3)
    dV_dalpha = np.zeros(3)
    dV_dphi = np.zeros(3)

    pos = [spherical_to_cartesian(R[k], alpha[k], phi[k]) for k in range(3)]  #[cite: 1]

    # Pairwise Newtonian gravitational forces projected onto generalized coordinates
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            r_vec = pos[i] - pos[j]
            dist = np.linalg.norm(r_vec) + eps
            dV_dpos_i = G * masses[i] * masses[j] * r_vec / (dist**3)

            # Jacobian of r_i with respect to (R_i, alpha_i, phi_i)[cite: 1]
            dx_dR = np.cos(phi[i]) * np.cos(alpha[i])
            dy_dR = np.sin(alpha[i])
            dz_dR = np.sin(phi[i]) * np.cos(alpha[i])

            dx_dalpha = -R[i] * np.cos(phi[i]) * np.sin(alpha[i])
            dy_dalpha = R[i] * np.cos(alpha[i])
            dz_dalpha = -R[i] * np.sin(phi[i]) * np.sin(alpha[i])

            dx_dphi = -R[i] * np.sin(phi[i]) * np.cos(alpha[i])
            dy_dphi = 0.0
            dz_dphi = R[i] * np.cos(phi[i]) * np.cos(alpha[i])

            J_i = np.array([
                [dx_dR, dx_dalpha, dx_dphi],
                [dy_dR, dy_dalpha, dy_dphi],
                [dz_dR, dz_dalpha, dz_dphi],
            ])

            gen_forces = J_i.T @ dV_dpos_i
            dV_dR[i] += gen_forces[0]
            dV_dalpha[i] += gen_forces[1]
            dV_dphi[i] += gen_forces[2]

    return dV_dR, dV_dalpha, dV_dphi


def eom_tilted_spherical(t, state, masses, G=1.0):
    """First-order ODE system d/dt [q, q_dot] for the 3-body system in generalized coordinates[cite: 1].

    State vector layout (size 18):
        state[0:3]   = R_1, R_2, R_3[cite: 1]
        state[3:6]   = alpha_1, alpha_2, alpha_3[cite: 1]
        state[6:9]   = phi_1, phi_2, phi_3[cite: 1]
        state[9:12]  = R_dot_1, R_dot_2, R_dot_3[cite: 1]
        state[12:15] = alpha_dot_1, alpha_dot_2, alpha_dot_3[cite: 1]
        state[15:18] = phi_dot_1, phi_dot_2, phi_dot_3[cite: 1]
    """
    R = state[0:3]  #[cite: 1]
    alpha = state[3:6]  #[cite: 1]
    phi = state[6:9]  #[cite: 1]
    R_dot = state[9:12]  #[cite: 1]
    alpha_dot = state[12:15]  #[cite: 1]
    phi_dot = state[15:18]  #[cite: 1]

    dV_dR, dV_dalpha, dV_dphi = potential_gradients(R, alpha, phi, masses, G)

    R_ddot = np.zeros(3)
    alpha_ddot = np.zeros(3)
    phi_ddot = np.zeros(3)

    for k in range(3):
        m_k = masses[k]
        cos_a = np.cos(alpha[k])
        sin_a = np.sin(alpha[k])
        R_val = max(R[k], 1e-6)  # Avoid coordinate singularity at origin

        # Radial acceleration from Euler-Lagrange equations[cite: 1]
        R_ddot[k] = R_val * (alpha_dot[k] ** 2 + (cos_a**2) * (phi_dot[k] ** 2)) - (1.0 / m_k) * dV_dR[k]  #[cite: 1]

        # Elevation angular acceleration[cite: 1]
        alpha_ddot[k] = (
            -2.0 * R_dot[k] * alpha_dot[k] / R_val
            - sin_a * cos_a * (phi_dot[k] ** 2)
            - (1.0 / (m_k * R_val**2)) * dV_dalpha[k]
        )  #[cite: 1]

        # In-plane azimuthal angular acceleration[cite: 1]
        denom_cos = cos_a if abs(cos_a) > 1e-5 else np.sign(cos_a) * 1e-5
        phi_ddot[k] = (
            -2.0 * R_dot[k] * phi_dot[k] / R_val
            + 2.0 * np.tan(alpha[k]) * alpha_dot[k] * phi_dot[k]
            - (1.0 / (m_k * (R_val**2) * (denom_cos**2))) * dV_dphi[k]
        )  #[cite: 1]

    return np.concatenate([R_dot, alpha_dot, phi_dot, R_ddot, alpha_ddot, phi_ddot])  #[cite: 1]
