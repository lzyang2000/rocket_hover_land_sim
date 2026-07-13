# Methods: 6-DOF plant, successive-convexification MPC, and paper mapping

This project draws from two main references:

> B. Açıkmeşe, J. M. Carson III, and L. Blackmore, “Lossless Convexification of Nonconvex Control Bound and Pointing Constraints of the Soft Landing Optimal Control Problem,” *IEEE Transactions on Control Systems Technology*, 21(6), 2013.

> M. Szmuk, T. P. Reynolds, and B. Açıkmeşe, “Successive Convexification for Real-Time 6-DoF Powered Descent Guidance with State-Triggered Constraints,” *Journal of Guidance, Control, and Dynamics*, 43(8), 2020. DOI: [10.2514/1.G004549](https://doi.org/10.2514/1.G004549).

The 2013 paper supplies the translational soft-landing equations, mass depletion, nonzero thrust interval, and pointing-cone motivation. The 2020 paper supplies the full quaternion rigid-body model and the successive-convexification structure used by the new MPC.

The implementation is an engineering adaptation rather than a reproduction of either numerical experiment. The exact controller specification and explicit non-goals are in [MPC_DESIGN.md](MPC_DESIGN.md).

## 1. Frames and state

The world frame is a nonrotating local Earth frame:

- `+Z`: local vertical;
- `+X`, `+Y`: horizontal directions;
- `g = [0, 0, -9.81] m/s^2`.

The body frame is fixed to the rocket with `+Z` along its longitudinal axis. The paper uses a different body-axis naming convention, but the equations are equivalent after permuting axes.

MuJoCo represents the free body with

\[
q_{MJ}=(r_x,r_y,r_z,q_w,q_x,q_y,q_z)
\]

and generalized velocity

\[
v_{MJ}=(v_x,v_y,v_z,\omega_x,\omega_y,\omega_z).
\]

Thus the plant has `nq = 7` generalized coordinates and `nv = 6` mechanical degrees of freedom.

The MPC prediction state follows the 2020 paper:

\[
x=(m,r,v,q,\omega)\in\mathbb{R}^{14}.
\]

Here `q` is a scalar-first body-to-world unit quaternion and `omega` is body-frame angular velocity.

## 2. Coupled rigid-body dynamics

The nonlinear prediction model is

\[
\dot m=-\alpha T,
\qquad
\dot r=v,
\]

\[
\dot v=g+\frac{1}{m}R(q)T\hat t_B,
\]

\[
\dot q=\frac{1}{2}q\otimes(0,\omega),
\]

\[
J\dot\omega=r_{T,B}\times T\hat t_B+	au_r e_3
-\omega\times J\omega.
\]

`R(q)` maps body-frame vectors into world coordinates. The engine pivot is approximately 20.1 m below the center of mass:

\[
r_{T,B}=(0,0,-20.1)\text{ m}.
\]

This moment arm is now present in both the optimizer and MuJoCo. The main-engine force is applied to the `thrust_origin` site through `mj_applyFT`; it is no longer applied at the center of mass. Gimbal commands therefore create physical pitch and yaw moments.

A single axial engine cannot create roll torque. The bounded scalar `tau_r` models separate roll-control authority. It is deliberately explicit rather than hidden inside an unconstrained attitude-assist torque.

MuJoCo remains the authoritative plant. The MPC model is propagated with RK4 over each prediction interval, and quaternions are normalized after propagation.

## 3. Falcon 9-like scale and mass depletion

The exterior geometry follows public Falcon 9 first-stage dimensions:

- height: approximately 41.2 m;
- diameter: 3.66 m;
- deployed leg span: approximately 17–18 m.

The landing-condition dynamics use:

- initial mass: 30,000 kg;
- dry mass: 21,000 kg;
- initial landing propellant: 9,000 kg;
- initial pitch/yaw inertia: approximately `4.3e6 kg m^2`;
- initial roll inertia: approximately `6.0e4 kg m^2`.

The mass equation is

\[
\dot m=-\alpha\lVert T\rVert,
\qquad
\alpha=5\times10^{-4}.
\]

MuJoCo mass and inertia are updated periodically as propellant burns. The MPC uses the same linear inertia-to-mass scaling as the plant.

`mj_setConst` evaluates the model at its reference pose, so the implementation preserves and restores the live free-joint position, quaternion, velocity, and simulation time around every mass-property recomputation. A regression test prevents the earlier teleport behavior from returning.

## 4. Thrust and actuator constraints

The paper-style nonzero thrust interval is

\[
0<\rho_1\leq T\leq\rho_2,
\]

with

\[
T_{nominal}=720\text{ kN},
\qquad
\rho_1=144\text{ kN},
\qquad
\rho_2=576\text{ kN}.
\]

Before ignition, thrust is exactly zero. While lit, it cannot enter the forbidden interval between zero and 144 kN. The kill command is a deliberate hybrid transition directly from valid positive thrust to zero.

The MPC control vector is

\[
u=(T,\delta_x,\delta_y,\tau_r).
\]

For `delta = [delta_x, delta_y]`, the unit body-frame thrust direction is

\[
\hat t_B=
\begin{bmatrix}
\sin\lVert\delta\rVert\,\delta/\lVert\delta\rVert\\
\cos\lVert\delta\rVert
\end{bmatrix}.
\]

The mechanical gimbal constraint is the second-order cone

\[
\lVert\delta\rVert_2\leq20^\circ.
\]

Roll torque is bounded by

\[
|\tau_r|\leq1.5\times10^6\text{ N m}.
\]

The GUI direction pad represents the current world-direction demand. Telemetry separately reports the actual mechanical gimbal angle and body tilt.

## 5. Successive convexification

The nonlinear discrete dynamics are denoted

\[
x_{k+1}=F_d(x_k,u_k).
\]

At each SCvx iteration they are linearized numerically around a nominal trajectory:

\[
x_{k+1}\approx A_kx_k+B_ku_k+c_k+\nu_k.
\]

Central finite differences produce `A_k` and `B_k`. The affine offset is

\[
c_k=F_d(\bar x_k,\bar u_k)-A_k\bar x_k-B_k\bar u_k.
\]

The virtual control `nu_k` is the artificial-infeasibility safeguard used in the SCvx literature. It receives a large scaled L1 penalty, allowing a convex subproblem to remain feasible even if a linearization is temporarily inconsistent. Valid converged solutions drive virtual control toward zero.

Scaled trust regions bound deviations from the nominal state and control trajectories. They address artificial unboundedness and keep first-order dynamics approximations locally meaningful.

## 6. Convex subproblem

Each CVXPY subproblem includes:

- affine linearized dynamics with virtual control;
- lower and upper thrust bounds;
- second-order-cone gimbal bounds;
- bounded roll torque;
- dry-mass and ground-height constraints;
- maximum tilt and angular-rate constraints;
- quaternion upper norm and linearized lower-norm safeguards;
- scaled state and control trust regions.

The cost contains:

- running state-tracking error;
- a strong terminal state penalty to prevent receding-horizon procrastination;
- control-slew regularization;
- small fuel-use and non-thrust actuator penalties;
- virtual-control and trust-region penalties.

Clarabel solves the conic problem. CVXPY parameters allow the compiled problem structure to be reused as dynamics matrices and target states change.

The default horizon has eight intervals of 0.35 s, for a 2.8 s prediction window. Three successive-convexification iterations are performed per update. The previous solution is shifted forward as the next warm start.

## 7. Receding-horizon execution and fallback

The paper primarily presents trajectory optimization. This simulator turns that machinery into MPC:

1. measure the MuJoCo state;
2. solve the finite-horizon SCvx problem;
3. apply only the first thrust, gimbal, and roll command;
4. advance the plant;
5. shift the previous solution and solve again.

The GUI submits solves on a background worker so rendering and input remain responsive. The most recent valid command is held between optimizer updates.

Every result is checked for solver status, finite values, actuator bounds, and nonlinear rollout defect. Until the first valid solution arrives—or whenever a solve fails—the simulator uses a deterministic 6-DOF fallback controller. Failure cannot leave unconstrained or stale actuator commands active.

Telemetry distinguishes:

- `SCVX MPC: OPTIMAL`;
- `SCVX MPC: WARMING`;
- `6-DOF FALLBACK`;
- manual `6-DOF TVC` control.

## 8. Manual 6-DOF control

Manual WASD input defines a desired world-frame body-up direction. A full SO(3) attitude controller constructs a desired rotation with fixed heading and computes

\[
e_R=\frac{1}{2}\left(R_d^TR-R^TR_d\right)^\vee,
\]

\[
\tau_d=-K_Re_R-K_\omega\omega.
\]

Pitch and yaw torque demands are allocated to lateral engine force using the 20.1 m lever arm. Roll torque is sent to the bounded roll actuator. Axis-specific gains account for the large difference between pitch/yaw and axial inertia.

This full heading constraint fixes the former underdetermined-yaw behavior. The old controller aligned only the body vertical axis, so rotation about that axis had no restoring attitude error; the full SO(3) controller regulates all three rotational coordinates.

## 9. Hover and auto-land references

Hover captures a target position with zero target velocity, identity attitude, and zero angular velocity. WASD moves the horizontal target, and Up/Down moves its altitude.

Auto-land supplies references through two phases:

### Align

The controller moves above the pad and stabilizes position, velocity, attitude, and angular rate. Descent begins after horizontal error, speed, altitude, and vertical-speed tolerances are satisfied.

### Descend

The target altitude moves downward at:

- 2.0 m/s above 10 m;
- 1.0 m/s from 3 to 10 m;
- 0.35 m/s below 3 m.

Engine cutoff occurs only near the landed center-of-mass height with small horizontal error and velocity and a bounded descent rate. MuJoCo then resolves landing-leg contact and settling.

The phase logic is intentionally separate from the optimizer. It provides interpretable reference generation while the MPC handles coupled six-degree-of-freedom tracking and actuator constraints.

## 10. What is and is not reproduced from the papers

Implemented:

- mass, translation, quaternion, and angular-rate states;
- rigid-body torque from thrust applied at an engine moment arm;
- nonzero thrust and gimbal constraints;
- repeated first-order dynamics linearization;
- convex conic subproblems;
- virtual control and trust regions;
- warm starts and real-time replanning.

Not implemented:

- free ignition time;
- free final time;
- atmospheric lift and drag;
- velocity-triggered angle-of-attack constraints;
- exact first-order-hold discretization matrices;
- a proof of convergence or global optimality.

The controller is tracking-oriented rather than a pure minimum-fuel planner. It is accurately described as an SCvx-inspired 6-DOF MPC, not as the full 2020 algorithm.

## 11. Is the project 6-DOF?

Yes, with an important terminology distinction:

- The MuJoCo plant is mechanically 6-DOF.
- The manual and fallback controllers regulate all three translation and all three rotation coordinates.
- The MPC predicts and controls the complete 14-state mass/6-DOF rigid-body model.
- Main-engine translation and pitch/yaw rotation are physically coupled through the engine moment arm.
- Roll is controlled by an explicit bounded actuator because a single axial engine cannot generate roll moment.

The project does not yet reproduce the paper's entire mission-level optimal-control problem, but it no longer uses a 3-DOF guidance law with an idealized pitch/yaw assist.

## 12. Verification

The test suite covers:

- MJCF compilation and free-joint dimensions;
- Falcon 9-like vehicle proportions;
- thrust, gimbal, tilt, ground, mass, and angular-rate constraints;
- quaternion and rigid-body prediction dynamics;
- physical engine-pivot pitch/yaw coupling;
- full heading and axial-spin recovery;
- virtual-control/nonlinear-defect diagnostics;
- forced solver failure and fallback selection;
- hover braking and position recovery;
- fallback and SCvx-MPC landing/cutoff rollouts;
- ground settling and mass-update teleport regression;
- GUI controls and indicators.

Run the complete suite with:

```bash
.venv/bin/pytest -q
```
