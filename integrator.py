import numpy as np
from scipy.integrate import solve_ivp
from .dynamics import eom_tilted_spherical, spherical_to_cartesian


def compute_invariants(t_eval, solution_states, masses, G=1.0):
    """Compute relative energy and angular momentum error time series[cite: 1, 2, 4].

    Returns:
        energy_error : array of |E(t) - E0| / |E0|[cite: 2]
        angular_momentum_error : array of ||L(t) - L0|| / ||L0||[cite: 4]
    """
    n_steps = solution_states.shape[1]
    E = np.zeros(n_steps)
    L_vec = np.zeros((3, n_steps))

    for idx in range(n_steps):
        state = solution_states[:, idx]
        R = state[0:3]  #[cite: 1]
        alpha = state[3:6]  #[cite: 1]
        phi = state[6:9]  #[cite: 1]
        R_dot = state[9:12]  #[cite: 1]
        alpha_dot = state[12:15]  #[cite: 1]
        phi_dot = state[15:18]  #[cite: 1]

        # Translational Kinetic Energy T_trans[cite: 1]
        T_trans = 0.5 * sum(
            masses[k]
            * (
                R_dot[k] ** 2
                + (R[k] ** 2) * (alpha_dot[k] ** 2)
                + (R[k] ** 2) * (np.cos(alpha[k]) ** 2) * (phi_dot[k] ** 2)
            )
            for k in range(3)
        )  #[cite: 1]

        # Mutual Gravitational Potential V[cite: 1]
        pos = [spherical_to_cartesian(R[k], alpha[k], phi[k]) for k in range(3)]  #[cite: 1]
        V = 0.0
        for i in range(3):
            for j in range(i + 1, 3):
                dist = np.linalg.norm(pos[i] - pos[j])
                V -= G * masses[i] * masses[j] / dist  #[cite: 1]

        E[idx] = T_trans + V  #[cite: 1, 2]

        # Total Angular Momentum Vector L = sum(m_k * (r_k x v_k))
        L_total = np.zeros(3)
        for k in range(3):
            # Cartesian velocity components via chain rule[cite: 1]
            cos_a, sin_a = np.cos(alpha[k]), np.sin(alpha[k])
            cos_p, sin_p = np.cos(phi[k]), np.sin(phi[k])
            vx = (
                R_dot[k] * cos_p * cos_a
                - R[k] * phi_dot[k] * sin_p * cos_a
                - R[k] * alpha_dot[k] * cos_p * sin_a
            )  #[cite: 1]
            vy = R_dot[k] * sin_a + R[k] * alpha_dot[k] * cos_a  #[cite: 1]
            vz = (
                R_dot[k] * sin_p * cos_a
                + R[k] * phi_dot[k] * cos_p * cos_a
                - R[k] * alpha_dot[k] * sin_p * sin_a
            )  #[cite: 1]

            vk = np.array([vx, vy, vz])
            L_total += masses[k] * np.cross(pos[k], vk)

        L_vec[:, idx] = L_total  #[cite: 4]

    # Relative errors
    energy_error = np.abs((E - E[0]) / E[0])  #[cite: 2]
    L0_norm = np.linalg.norm(L_vec[:, 0])
    angular_momentum_error = np.linalg.norm(L_vec - L_vec[:, [0]], axis=0) / L0_norm  #[cite: 4]

    return energy_error, angular_momentum_error


def integrate_system(
    initial_state,
    t_span,
    masses,
    method="DOP853",
    rtol=1e-10,
    atol=1e-12,
    t_eval=None,
):
    """Run adaptive numerical integration for the multi-body system."""
    sol = solve_ivp(
        fun=lambda t, y: eom_tilted_spherical(t, y, masses),
        t_span=t_span,
        y0=initial_state,
        method=method,
        rtol=rtol,
        atol=atol,
        t_eval=t_eval,
    )
    return sol
