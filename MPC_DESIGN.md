# 6-DOF successive-convexification MPC design

## 1. Purpose

This document specifies the real-time controller used by the MuJoCo rocket landing simulator. The controller is an engineering adaptation of:

> M. Szmuk, T. P. Reynolds, and B. Açıkmeşe, “Successive Convexification for Real-Time 6-DoF Powered Descent Guidance with State-Triggered Constraints,” *Journal of Guidance, Control, and Dynamics*, 43(8), 2020. DOI: [10.2514/1.G004549](https://doi.org/10.2514/1.G004549).

The paper solves a free-final-time trajectory-optimization problem. This project repeatedly solves a shorter fixed-horizon problem and applies only the first command, producing a receding-horizon model predictive controller.

## 2. Scope

The prediction state is

\[
x = (m,r,v,q,\omega) \in \mathbb{R}^{14},
\]

where `m` is mass, `r` and `v` are world-frame position and velocity, `q` is a scalar-first body-to-world quaternion, and `omega` is body-frame angular velocity.

The control is

\[
u = (T,\delta_x,\delta_y,\tau_r),
\]

where `T` is thrust magnitude, `(delta_x, delta_y)` is the two-axis engine-gimbal command, and `tau_r` is the equivalent moment requested from the roll RCS. A single engine can create pitch and yaw torque through its moment arm but cannot create torque about its own thrust axis.

## 3. Nonlinear prediction model

The model uses

\[
\dot m=-\alpha T,
\qquad
\dot r=v,
\qquad
\dot v=g+\frac{1}{m}R(q)T\hat t_B,
\]

\[
\dot q=\frac{1}{2}q\otimes(0,\omega),
\qquad
J(m)\dot\omega=r_{T,B}(m)\times T\hat t_B+\tau_r e_3
-\omega\times J(m)\omega.
\]

The position and velocity states are those of the current center of mass. Both
`J(m)` and the engine-to-COM vector `r_{T,B}(m)` come from the shared draining-
tank mass model.

The gimbal direction is constructed from the two-component angle vector `delta`:

\[
\hat t_B=
\begin{bmatrix}
\sin\lVert\delta\rVert\,\delta/\lVert\delta\rVert\\
\cos\lVert\delta\rVert
\end{bmatrix}.
\]

The nonlinear model is propagated with RK4. Quaternion normalization is applied after each prediction step.

## 4. Successive convexification

At every MPC update, the nonlinear discrete dynamics are linearized numerically around a warm-start trajectory. Three successive convexification iterations are used by default:

\[
x_{k+1}\approx A_kx_k+B_ku_k+c_k+\nu_k.
\]

The virtual control `nu_k` follows the paper’s artificial-infeasibility safeguard and receives a large L1 penalty. Scaled trust regions keep each convex solution near the trajectory about which the dynamics were linearized.

Each convex subproblem includes:

- lower and upper thrust bounds;
- a second-order-cone gimbal-angle bound;
- a bounded roll-control torque;
- dry-mass and ground-height bounds;
- quaternion tilt and angular-rate bounds;
- terminal and running state-tracking costs;
- control-slew and fuel-use costs;
- virtual-control and trust-region penalties.

CVXPY constructs the SOCP and Clarabel solves it. The previous prediction is shifted forward to warm-start the next solve.

## 5. Receding-horizon operation

Hover and auto-land provide a moving target state to the MPC. A target with nonzero velocity is expanded into a time-varying reference trajectory, with reference position advanced by `k * prediction_dt * target_velocity` at each prediction node. The optimizer runs at a lower rate than MuJoCo physics, and the most recent first control is held between solves. The controller exposes solve status, solve time, dynamics defect, and virtual-control magnitude in telemetry.

If the solver is unavailable, infeasible, numerically invalid, or exceeds the accepted defect threshold, the simulator immediately uses a deterministic six-degree-of-freedom fallback controller. Solver failure therefore cannot remove attitude stabilization or leave stale unbounded commands active.

The simulator uses scheduled MPC gimbal bounds of 5° during hover and 6° during high-altitude landing, both inside the 20° mechanical limit, and accepts solutions up to a 0.20 normalized nonlinear rollout defect. The deterministic automatic controller retains a 6° recovery envelope. Hover position commands are limited to a 2.5 m horizontal and 2 m vertical lead relative to the plant. During landing, MPC owns alignment and descent above 7 m; a deterministic terminal controller then performs the final approach under the tighter low-altitude gimbal schedule. This intentional handoff is reported separately from solver fallback.

## 6. Physical control application

MuJoCo receives the engine force at the `thrust_origin` site rather than at the center of mass. The resulting fuel-dependent moment arm creates physical pitch/yaw torque. Roll is not injected as a pure torque: two equal and opposite tangential forces act at sites 1.75 m to either side of the body axis. Their net force is zero and their moments add about the body axis. Each modeled pod is limited to 5 kN, giving 17.5 kN m maximum roll moment, and the applied moment follows the command through a 0.10 s first-order response.

The main-engine gimbal is also a dynamic actuator rather than an instantaneous control. Its normal response time is 0.08 s. Below 5 m, a slower 0.20 s response, progressively smaller angle limits, and a small-command deadband suppress terminal control reversals. The MPC still optimizes the bounded commanded angle, while MuJoCo applies the lagged mechanical state.

The dry structure and effective LOX/RP-1 liquid columns determine mass, center of mass, and principal inertia. The same mass-property function is used by MuJoCo and the nonlinear MPC rollout. As the COM moves, the MPC recomputes the engine position relative to it.

Manual mode uses a full SO(3) attitude controller and the same physical actuator allocation. It holds heading as well as body tilt, fixing the previous underdetermined-yaw behavior.

## 7. Differences from the paper

This implementation deliberately omits:

- free ignition and free final time;
- atmospheric lift and drag;
- state-triggered angle-of-attack constraints;
- exact first-order-hold discretization matrices;
- claims of global optimality or formal convergence guarantees.

It retains the paper’s central practical structure: full quaternion rigid-body dynamics, thrust-at-pivot torque, repeated linearization, convex conic subproblems, virtual control, trust regions, and real-time replanning.

## 8. Acceptance criteria

- A perturbed rocket must remove yaw, pitch, and roll rates during powered flight.
- The roll force pair must have zero net force and produce the requested bounded axial moment.
- MuJoCo and MPC must agree on fuel-dependent COM and inertia.
- Main-engine gimbal must remain inside its mechanical cone.
- Thrust must remain inside the nonzero paper-style interval while lit.
- Hover must recover position and attitude disturbances.
- Auto-land must align, descend, cut off, and settle on the pad.
- A forced solver failure must select the fallback controller without destabilizing the vehicle.
- The GUI must remain responsive and report whether MPC or fallback control owns the vehicle.
