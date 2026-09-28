import numpy as np
from scipy.optimize import root_scalar


def pseudo_potential(x, y, z=0.0, mu=0.0121505856, eps=1e-12):
    """Effective potential U(x, y, z) in the synodic rotating frame[cite: 9]."""
    m1, m2 = 1.0 - mu, mu
    r1 = np.sqrt((x + mu) ** 2 + y**2 + z**2) + eps
    r2 = np.sqrt((x - 1.0 + mu) ** 2 + y**2 + z**2) + eps
    return 0.5 * (x**2 + y**2) + (m1 / r1) + (m2 / r2)


def jacobi_constant(state, mu=0.0121505856):
    """Compute the conserved Jacobi energy constant C = 2*U - v^2[cite: 9]."""
    x, y, z, vx, vy, vz = state
    U = pseudo_potential(x, y, z, mu)
    v2 = vx**2 + vy**2 + vz**2
    return 2.0 * U - v2


def compute_lagrange_points(mu=0.0121505856):
    """Find the coordinates of all five libration points L1-L5[cite: 9]."""
    x1, x2 = -mu, 1.0 - mu

    def dudx_line(x):
        r1 = abs(x - x1)
        r2 = abs(x - x2)
        return x - (1.0 - mu) * (x - x1) / (r1**3) - mu * (x - x2) / (r2**3)

    x_l1 = root_scalar(dudx_line, bracket=[x1 + 1e-4, x2 - 1e-4]).root
    x_l2 = root_scalar(dudx_line, bracket=[x2 + 1e-4, 2.5]).root
    x_l3 = root_scalar(dudx_line, bracket=[-2.5, x1 - 1e-4]).root

    return {
        "L1": np.array([x_l1, 0.0, 0.0]),
        "L2": np.array([x_l2, 0.0, 0.0]),
        "L3": np.array([x_l3, 0.0, 0.0]),
        "L4": np.array([0.5 - mu, np.sqrt(3) / 2.0, 0.0]),
        "L5": np.array([0.5 - mu, -np.sqrt(3) / 2.0, 0.0]),
    }


def cr3bp_eom(t, state, mu=0.0121505856, eps=1e-12):
    """Equations of motion for a third body in the CR3BP rotating frame[cite: 8, 9]."""
    x, y, z, vx, vy, vz = state
    r1 = np.sqrt((x + mu) ** 2 + y**2 + z**2) + eps
    r2 = np.sqrt((x - 1.0 + mu) ** 2 + y**2 + z**2) + eps

    # Gradients of the effective potential
    dUdx = x - (1.0 - mu) * (x + mu) / (r1**3) - mu * (x - 1.0 + mu) / (r2**3)
    dUdy = y - (1.0 - mu) * y / (r1**3) - mu * y / (r2**3)
    dUdz = -(1.0 - mu) * z / (r1**3) - mu * z / (r2**3)

    ax = 2.0 * vy + dUdx
    ay = -2.0 * vx + dUdy
    az = dUdz

    return [vx, vy, vz, ax, ay, az]


def l4_variational_matrix(mu):
    """Return the 4x4 Jacobian matrix evaluated at L4 for linear Routh stability analysis[cite: 8]."""
    # Second partial derivatives of U at L4 (x = 0.5 - mu, y = sqrt(3)/2)
    Uxx = 0.75
    Uyy = 2.25
    Uxy = (3.0 * np.sqrt(3) / 4.0) * (1.0 - 2.0 * mu)

    A = np.array([
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [Uxx, Uxy, 0.0, 2.0],
        [Uxy, Uyy, -2.0, 0.0],
    ])
    return A
