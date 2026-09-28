# Non-Planar Three-Body Dynamics: Constrained Attitude Coupling & Verification Suite

A computational dynamics package and analytical Lagrangian framework for simulating non-planar multi-body systems in generalized curvilinear coordinates, validated against classical benchmarks in celestial mechanics.

<p align="center">
  <img src="assets/tilted_frames.svg" alt="Tilted Coordinate Frames" width="520">
</p>

## Overview

Textbook multi-body formulations frequently enforce coplanar simplifications. This project develops an analytical Lagrangian framework in generalized spherical coordinates $(R_k, \alpha_k, \phi_k)$ across independently tilted orbital planes:
- **Kinematic Formulation:** Derivation of 3D line elements and mutual Euclidean distances across arbitrary spatial inclinations.
- **Attitude Reduction:** A 9-DOF holonomically constrained model capturing nadir-locked spin-orbit kinetic cross-terms.
- **Canonical Verification:** A test suite validating numerical trajectories against known analytical limits (Lagrange homographic configurations, Routh stability thresholds, and Jacobi zero-velocity surfaces).

---

## Kinematic & Dynamic Model

The Cartesian position $\vec{r}_k = (x_k, y_k, z_k)^T$ of each mass $m_k$ is parameterized via radial distance $R_k(t)$, out-of-plane elevation $\alpha_k(t)$, and in-plane sweep $\phi_k(t)$:

$$\begin{aligned}
x_k &= R_k \cos\phi_k \cos\alpha_k \\
y_k &= R_k \sin\alpha_k \\
z_k &= R_k \sin\phi_k \cos\alpha_k
\end{aligned}$$

The system Lagrangian $L = T_{\text{trans}} + T_{\text{rot}} - V$ incorporates:
1. **Translational Velocity:** $v_k^2 = \dot{R}_k^2 + R_k^2\dot{\alpha}_k^2 + R_k^2\cos^2\alpha_k\dot{\phi}_k^2$.
2. **Constrained Attitude Rates:** Dynamic rotational kinetic energy projected through local orbital frame rates $\vec{\omega}_k$.
3. **Mutual Gravitational Tensors:** Exact 3D Euclidean distances $r_{ij} = \sqrt{R_i^2 + R_j^2 - 2\vec{r}_i \cdot \vec{r}_j}$ without planar projections.

---

## Numerical Verification Suite

The formulation is evaluated across six rigorous physical and analytical benchmarks:

### 1. Noether Invariant Preservation
- **Energy Conservation:** Relative total energy error maintained at $\Delta E/E_0 < 10^{-6}$ over integration cycles.
- **Angular Momentum Conservation:** Vector error of the total spatial angular momentum $\|\Delta \mathbf{L}\| / \|\mathbf{L}_0\| < 10^{-8}$.

<p align="center">
  <img src="ASSETS/advanced_3body_tests.png" alt="Verification Tests" width="800">
</p>

### 2. Chaotic Divergence & Lyapunov Sensitivity
- Quantified exponential divergence of phase-space separation $\|q_1(t) - q_2(t)\|$ from an initial perturbation of $\delta = 10^{-7}$, demonstrating deterministic chaos during close three-body encounters.

### 3. Lagrange $L_4/L_5$ Equilateral Solution
- Reproduction of Joseph-Louis Lagrange's 1772 homographic equilateral solution, confirming identical periodic breathing of inter-mass distances ($r_{12} = r_{23} = r_{31}$).

### 4. Routh Linear Stability Criterion ($\mu_{\text{crit}} \approx 0.0385$)
Validation of triangular libration point stability in the Circular Restricted Three-Body Problem (CR3BP):
- **Stable Regime ($\mu = 0.01215 < \mu_{\text{crit}}$):** Bounded epicyclic libration around the $L_4$ equilibrium point.
- **Unstable Regime ($\mu = 0.05 > \mu_{\text{crit}}$):** Rapid exponential departure driven by positive real eigenvalue components.

<p align="center">
  <img src="ASSETS/routh_stability_criterion_test.png" alt="Routh Stability Criterion" width="750">
</p>

### 5. Zero-Velocity Curves & Hill Surfaces
Topological mapping of the Jacobi energy integral $C$ across the four fundamental gateway transition regimes:
1. $C > C_{L1}$: Closed gateways isolating the primary bodies.
2. $C_{L2} < C < C_{L1}$: Opening of the $L_1$ interior bottleneck permitting Earth-Moon transfer.
3. $C_{L3} < C < C_{L2}$: Opening of the $L_2$ exterior neck allowing heliocentric escape.
4. $C < C_{L4,5}$: Complete disappearance of forbidden regions (unbounded motion).

<p align="center">
  <img src="ASSETS/hill_surfaces_zero_velocity.png" alt="Hill Surfaces and Gateways" width="750">
</p>

---

## Numerical Limitations & Boundary Behaviors

In the interest of rigorous computational evaluation, the following boundary behaviors are documented:
- **Coordinate Singularities:** Formulating motion in spherical coordinates introduces gimbal-lock singularities at polar crossings ($\alpha \to \pm \pi/2$) and coordinate divergence near periapsis close encounters ($R \to 0$). Future iterations will transition translational integration into regularized Kustaanheimo-Stiefel (KS) or Cartesian states.
- **Attitude Sub-manifold:** The current model enforces zero axial spin roll along the line-of-sight vector ($\vec{\omega}_k \cdot \hat{r}_k = 0$). Extending this to an unconstrained 18-DOF system requires decoupling Euler angles/quaternions and integrating MacCullagh quadrupole gravitational potentials.

python benchmarks/test_routh_stability.py
python benchmarks/plot_hill_surfaces.py
python benchmarks/test_lagrange_l4l5.py
