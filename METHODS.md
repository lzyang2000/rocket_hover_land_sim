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
J(m)\dot\omega=r_{T,B}(m)\times T\hat t_B+\tau_r e_3
-\omega\times J(m)\omega.
\]

`R(q)` maps body-frame vectors into world coordinates. The engine site is fixed approximately 20.1 m below the vehicle reference, while the center of mass `c_B(m)` moves with propellant depletion:

\[
r_{T,B}(m)=(0,0,-20.1)-c_B(m).
\]

MuJoCo's free-joint translation is the vehicle reference position `r_O`, not the moving COM. Guidance therefore converts it using

\[
r_C=r_O+R(q)c_B,
\qquad
v_C=v_O+R(q)\left(\omega\times c_B+\dot c_B\right).
\]

The `dot c_B` term is evaluated from the mass-model derivative and current propellant flow. Hover and MPC states would otherwise report a false residual velocity while the liquid columns migrate.

This moment arm is now present in both the optimizer and MuJoCo. The main-engine force is applied to the `thrust_origin` site through `mj_applyFT`; it is no longer applied at the center of mass. Gimbal commands therefore create physical pitch and yaw moments.

A single axial engine cannot create roll torque. The bounded scalar `tau_r` is the equivalent command for two physical RCS forces. Sites at `(+1.75, 0, 15.25)` and `(-1.75, 0, 15.25)` m receive equal and opposite tangential forces:

\[
F_+=(0,F,0),\qquad F_-=(0,-F,0).
\]

Their translational forces cancel while their axial moments add:

\[
F_++F_-=0,\qquad \tau_r=2(1.75)F.
\]

The modeled force limit is 5 kN per pod, producing at most 17.5 kN m. A 0.10 s first-order actuator response prevents instantaneous torque steps. The MPC optimizes the equivalent bounded moment; MuJoCo applies the actual lagged force pair.

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

Mass properties are calculated from a calibrated dry stage and two effective liquid columns:

- LOX: 72% of landing-reserve propellant, effective tank interval from `z=-2` to `z=14` m;
- RP-1: 28%, effective tank interval from `z=-14` to `z=-2` m;
- both columns use an effective 1.65 m radius and shorten toward their lower boundaries as propellant drains.

The dry-stage COM and intrinsic inertia are solved so that the 30,000 kg initial condition has `c_B=0` and exactly matches the original `4.3e6, 4.3e6, 6.0e4 kg m^2` principal inertia. At half landing reserve the model gives approximately `c_z=-1.02 m` and `J_x=J_y=3.89e6 kg m^2`. At dry mass it gives approximately `c_z=-0.89 m` and `J=[3.71e6, 3.71e6, 4.77e4] kg m^2`.

MuJoCo installs this moving inertial frame and principal inertia periodically as propellant burns. Hover guidance uses the corresponding COM position and velocity, while touchdown thresholds remain attached to the physical landing-leg/body reference. The MPC calls the same mass-property model and recomputes both `J(m)` and the engine-to-COM moment arm.

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

MuJoCo does not apply this command instantaneously. The mechanical gimbal state follows it through a 0.08 s first-order actuator. During terminal descent the response time becomes 0.20 s and available angle is scheduled to 3 degrees below 5 m, 1.5 degrees below 2.5 m, and 0.75 degrees below 1 m. Commands smaller than 0.15 degrees are suppressed in this terminal regime. This removes rapid MPC direction reversals that previously appeared as low-altitude wobble.

Equivalent roll-RCS moment is bounded by

\[
|\tau_r|\leq1.75\times10^4\text{ N m}.
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

Pitch and yaw torque demands are allocated to lateral engine force using the current engine-to-COM lever arm. Roll demand is sent to the bounded, lagged RCS force pair. Axis-specific gains account for the large difference between pitch/yaw and axial inertia.

This full heading constraint fixes the former underdetermined-yaw behavior. The old controller aligned only the body vertical axis, so rotation about that axis had no restoring attitude error; the full SO(3) controller regulates all three rotational coordinates.

## 9. Hover and auto-land references

Hover captures a target position with zero target velocity, identity attitude, and zero angular velocity. WASD moves the horizontal target, and Up/Down moves its altitude. Auto-land additionally supplies a nonzero vertical target velocity so the controller tracks a deliberate descent profile rather than chasing a moving position target whose nominal velocity is incorrectly zero.

Auto-land supplies references through two phases:

### Align

The controller moves above the pad and stabilizes position, velocity, attitude, and angular rate. Descent begins once horizontal error is below 1.5 m, horizontal speed is below 0.75 m/s, staging-altitude error is below 1 m, and vertical speed is below 0.75 m/s. These deliberately coarse gates avoid spending excessive time seeking a perfect high-altitude alignment.

### Descend

The target altitude and vertical-velocity reference descend at:

- 12 m/s above 30 m;
- 8 m/s from 18 to 30 m;
- 5 m/s from 10 to 18 m;
- 3 m/s from 5 to 10 m;
- 1.5 m/s from 2.5 to 5 m;
- 0.6 m/s from 1 to 2.5 m;
- 0.25 m/s below 1 m.

The high-altitude bands create a visibly forceful approach, while the final bands reserve enough altitude for powered braking. The altitude reference is integrated continuously, and the velocity reference switches directly from one band speed to the next. No zero-speed waypoint is inserted at a band boundary, and the target altitude is not reset against the measured vehicle altitude.

Engine cutoff occurs within 0.15 m of the landed body/leg reference height when horizontal error is below 0.50 m, horizontal speed is below 0.30 m/s, and vertical speed is between -0.50 and +0.15 m/s. MuJoCo then resolves the deliberately small residual motion through landing-leg contact and pad friction rather than holding the engine on for repeated terminal corrections.

### Fuel-reserve takeover

Outside auto-land, the simulator periodically estimates landing propellant from a transparent impulse approximation. For height $h$, the nominal profile time is

\[
t_d(h)=\int_0^h \frac{d\eta}{v_d(\eta)},
\]

where $v_d(h)$ is the nonzero piecewise descent speed above. The alignment-time estimate is

\[
t_a=\min\left(8,\max\left(\frac{\lVert r_{xy}\rVert}{3},
\frac{\lVert v_{xy}\rVert}{1.5}\right)\right).
\]

With current wet mass $m$, gravity magnitude $g$, horizontal speed $v_{xy}$, and excess downward speed $v_e=\max(-v_z-v_d(h),0)$, estimated fuel is

\[
m_{f,\mathrm{land}}=\alpha\left[m g(t_a+t_d)
+m(\lVert v_{xy}\rVert+v_e)\right]+100\ \mathrm{kg}.
\]

When the engine is lit, the rocket is above the end-burn cutoff height, and remaining fuel satisfies

\[
m_f\leq 1.05\,m_{f,\mathrm{land}},
\]

auto-land takes over and the trigger latches for the flight. Reserve takeover uses the current altitude as its alignment altitude, avoiding an unnecessary climb to the normal staging height. This is a conservative heuristic guard, not an MPC-derived certified propellant-to-go bound.

### Controller ownership indicator

The GUI displays the controller that currently owns the actuators:

- `MPC ACTIVE` after a valid SCvx result is accepted;
- `FALLBACK ACTIVE` while automatic control is using the deterministic backup, including MPC warm-up or solver failure;
- `MANUAL TVC` when hover and auto-land are inactive.

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
- Roll is controlled by an explicit opposed-force RCS couple because a single axial engine cannot generate roll moment.

The project does not yet reproduce the paper's entire mission-level optimal-control problem, but it no longer uses a 3-DOF guidance law with an idealized pitch/yaw assist.

## 12. Verification

The test suite covers:

- MJCF compilation and free-joint dimensions;
- Falcon 9-like vehicle proportions;
- thrust, gimbal, tilt, ground, mass, and angular-rate constraints;
- quaternion and rigid-body prediction dynamics;
- physical engine-pivot pitch/yaw coupling;
- shared fuel-dependent COM/inertia and engine-lever-arm calculations;
- zero-net-force RCS roll-couple allocation and actuator lag;
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
