from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import root_scalar


def pseudo_potential(x, y, z=0.0, mu=0.0121505856, eps=1e-12):
    """Compute the non-dimensional CR3BP pseudo-potential U(x, y, z)."""
    m1, m2 = 1.0 - mu, mu
    x1, y1 = -mu, 0.0
    x2, y2 = 1.0 - mu, 0.0

    r1 = np.sqrt((x - x1) ** 2 + (y - y1) ** 2 + z**2) + eps
    r2 = np.sqrt((x - x2) ** 2 + (y - y2) ** 2 + z**2) + eps
    return 0.5 * (x**2 + y**2) + (m1 / r1) + (m2 / r2)


def compute_lagrange_points(mu=0.0121505856):
    """Find the coordinates and critical Jacobi constants for L1 through L5."""
    m1, m2 = 1.0 - mu, mu
    x1, x2 = -mu, 1.0 - mu

    def dudx_line(x):
        r1 = abs(x - x1)
        r2 = abs(x - x2)
        return x - m1 * (x - x1) / (r1**3) - m2 * (x - x2) / (r2**3)

    # Collinear roots via 1D bracketed scalar search
    x_l1 = root_scalar(dudx_line, bracket=[x1 + 1e-4, x2 - 1e-4]).root
    x_l2 = root_scalar(dudx_line, bracket=[x2 + 1e-4, 2.5]).root
    x_l3 = root_scalar(dudx_line, bracket=[-2.5, x1 - 1e-4]).root

    points = {
        "L1": (x_l1, 0.0),
        "L2": (x_l2, 0.0),
        "L3": (x_l3, 0.0),
        "L4": (0.5 - mu, np.sqrt(3) / 2.0),
        "L5": (0.5 - mu, -np.sqrt(3) / 2.0),
    }

    constants = {name: 2.0 * pseudo_potential(x, y, mu=mu) for name, (x, y) in points.items()}
    return points, constants


def plot_hill_surfaces(
    mu=0.0121505856,
    grid_res=600,
    save_path="assets/hill_surfaces.png",
):
    """Plot the four topological regimes of the Jacobi constant."""
    points, c_crit = compute_lagrange_points(mu)

    # Coordinate domain
    x_grid = np.linspace(-1.6, 1.6, grid_res)
    y_grid = np.linspace(-1.4, 1.4, grid_res)
    X, Y = np.meshgrid(x_grid, y_grid)
    U2 = 2.0 * pseudo_potential(X, Y, mu=mu)

    fig, axes = plt.subplots(2, 2, figsize=(13, 11), sharex=True, sharey=True)
    fig.suptitle(
        r"Zero-Velocity Curves (Hill Surfaces) & Gateway Dynamics in CR3BP ($\mu = 0.01215$)",
        fontsize=15,
        fontweight="bold",
    )

    cases = [
        (
            c_crit["L1"] + 0.04,
            f"Case 1: $C > C_{{L1}}$ ({c_crit['L1'] + 0.04:.2f})\nClosed Gateways (Isolated Primaries)",
        ),
        (
            0.5 * (c_crit["L1"] + c_crit["L2"]),
            f"Case 2: $C_{{L2}} < C < C_{{L1}}$ ({0.5 * (c_crit['L1'] + c_crit['L2']):.2f})\n$L_1$ Neck Opens (Earth-Moon Transfer Allowed)",
        ),
        (
            0.5 * (c_crit["L2"] + c_crit["L3"]),
            f"Case 3: $C_{{L3}} < C < C_{{L2}}$ ({0.5 * (c_crit['L2'] + c_crit['L3']):.2f})\n$L_2$ Neck Opens (Escape to Heliocentric Space)",
        ),
        (
            c_crit["L4"] - 0.03,
            f"Case 4: $C < C_{{L4,5}}$ ({c_crit['L4'] - 0.03:.2f})\nAll Forbidden Regions Disappear (Unbounded Motion)",
        ),
    ]

    for ax, (c_val, title) in zip(axes.flat, cases):
        # Forbidden realm where kinetic energy v^2 = 2U - C < 0
        ax.contourf(
            X,
            Y,
            U2 < c_val,
            levels=[0.5, 1.5],
            colors=["#2b5c8f"],
            alpha=0.45,
        )
        ax.contour(X, Y, U2, levels=[c_val], colors=["#002b49"], linewidths=1.6)

        # Primary masses
        ax.plot(-mu, 0.0, "ko", markersize=9, label=r"Earth ($m_1$)")
        ax.plot(1.0 - mu, 0.0, "mo", markersize=5.5, label=r"Moon ($m_2$)")

        # Libration equilibrium points
        ax.plot(points["L1"][0], 0.0, "r*", markersize=8, label="$L_1$")
        ax.plot(points["L2"][0], 0.0, "g*", markersize=8, label="$L_2$")
        ax.plot(points["L3"][0], 0.0, "b*", markersize=7, label="$L_3$")
        ax.plot(
            [points["L4"][0], points["L5"][0]],
            [points["L4"][1], points["L5"][1]],
            "c^",
            markersize=6,
            label="$L_4, L_5$",
        )

        ax.set_title(title, fontsize=10.5, fontweight="semibold")
        ax.set_aspect("equal")
        ax.grid(True, linestyle=":", alpha=0.55)

    # Clean axes labels and unified legend
    for ax in axes[:, 0]:
        ax.set_ylabel(r"Rotating Frame $y$ [-]")
    for ax in axes[1, :]:
        ax.set_xlabel(r"Rotating Frame $x$ [-]")

    axes[0, 0].legend(loc="upper right", fontsize=8.5, framealpha=0.85)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Figure successfully saved to: {save_path}")

    plt.show()


if __name__ == "__main__":
    plot_hill_surfaces()
