# Methods: paper model, MuJoCo implementation, and guidance laws

This document explains how the simulator relates to:

> B. Açıkmeşe, J. M. Carson III, and L. Blackmore, “Lossless Convexification of Nonconvex Control Bound and Pointing Constraints of the Soft Landing Optimal Control Problem,” *IEEE Transactions on Control Systems Technology*, 21(6), 2013.

The short version is:

- The simulator directly implements the paper's translational dynamics, mass depletion, thrust-magnitude bounds, and thrust-pointing cone.
- MuJoCo provides a free rigid body with contact, quaternion attitude, angular velocity, and landing-leg collisions.
- Hover and auto-land currently use feedback guidance, not the paper's fuel-optimal lossless-convexification solver.
- Attitude is controlled by a separate high-bandwidth assist torque, preserving the paper's assumption that translation and attitude can be treated as approximately decoupled.

## 1. Frames and state

The world frame is a nonrotating local Earth frame:

- `+Z`: local vertical
- `+X`, `+Y`: horizontal directions
- gravity: `g = [0, 0, -9.81] m/s^2`

The paper's translational state is

\[
x(t) = \begin{bmatrix}r(t) \\ v(t)\end{bmatrix}\in\mathbb{R}^6.
\]

The MuJoCo body uses a free joint:

\[
q = (r_x,r_y,r_z,q_w,q_x,q_y,q_z),
\]

with generalized velocity

\[
\dot q_{minimal}=(v_x,v_y,v_z,\omega_x,\omega_y,\omega_z).
\]

Consequently, the mechanical plant has `nq = 7` coordinates and `nv = 6` velocity degrees of freedom.

## 2. Translational dynamics

With planetary rotation set to zero, the paper's equation reduces to

\[
\dot r=v,
\qquad
\dot v=g+\frac{T}{m}.
\]

MuJoCo numerically integrates these dynamics. The commanded thrust is applied as a Cartesian body force, and gravity is configured in `rocket.xml`.

The current model applies thrust at the center of mass. This intentionally follows the paper's translational abstraction. A higher-fidelity model would apply thrust at the engine pivot so gimbal angle also generates a physical moment.

## 3. Propellant depletion and changing mass

The paper uses

\[
\dot m=-\alpha\lVert T\rVert.
\]

The simulator uses

\[
\alpha=5\times10^{-4},
\]

with an approximate depleted-booster landing mass of 21,000 kg dry mass and 9,000 kg initial propellant. Fuel is integrated every physics step. MuJoCo body mass and inertia are updated periodically as propellant burns.

The exterior geometry follows public Falcon 9 first-stage dimensions: approximately 41.2 m high and 3.66 m in diameter. Its initial pitch/yaw inertia is approximately `4.3e6 kg m^2`, consistent with a tall 30,000 kg rigid body. These are educational approximations rather than authoritative SpaceX mass properties.

`mj_setConst` evaluates the model at its reference pose, so the implementation preserves and restores position, quaternion, velocity, and simulation time around each mass recomputation. A regression test ensures mass updates cannot teleport the free body.

## 4. Main-engine thrust constraints

The paper's nonconvex lower bound and convex upper bound are

\[
0<\rho_1\leq\lVert T\rVert\leq\rho_2.
\]

The throttle fractions are based on the paper-style example, while the force scale is multiplied by 30 to match the larger landing-condition mass:

\[
T_{nominal}=720\text{ kN},
\qquad
\rho_1=0.20T_{nominal}=144\text{ kN},
\qquad
\rho_2=0.80T_{nominal}=576\text{ kN}.
\]

Before ignition, thrust is exactly zero. After ignition, throttle is clipped to `[20%, 80%]`; no command exists in `(0, 20%)`. Scaling both the original 1,000 kg mass and 24 kN nominal force by 30 preserves the original translational thrust-to-weight behavior and its approximately 40.9% hover throttle.

These bounds are not presented as measured Merlin throttle limits. They deliberately preserve the paper experiment's nonzero lower-thrust constraint while using a Falcon 9-like force scale.

The red kill control adds one deliberate hybrid transition from valid positive thrust directly to zero. That is useful for manual touchdown but relaxes the paper's stricter assumption that the descent engine cannot shut down after ignition.

## 5. Gimbal and pointing constraint

The pointing constraint is

\[
\hat n^TT\geq\lVert T\rVert\cos\theta,
\]

where `n_hat = [0, 0, 1]` and `theta = 20 degrees` in the demo. This constrains thrust to a cone around local vertical.

The two-component lateral command `c = [c_x,c_y]` is limited to `||c|| <= 1`. Its magnitude selects a gimbal angle:

\[
\phi=\lVert c\rVert\theta,
\]

and its direction selects the horizontal direction of the thrust vector:

\[
\hat T=
\begin{bmatrix}
\frac{c_x}{\lVert c\rVert}\sin\phi \\
\frac{c_y}{\lVert c\rVert}\sin\phi \\
\cos\phi
\end{bmatrix}.
\]

This construction satisfies the pointing cone by definition. Manual WASD commands are slew-limited before being converted into `T_hat`.

## 6. Attitude-assist controller

The paper explicitly decouples translational and rotational dynamics, assuming attitude control is much faster than translation. The simulator represents that assumption using

\[
\tau=k_R(z_{body}\times\hat T)-k_\omega\omega.
\]

This torque rotates the rendered rigid body toward the desired thrust direction and damps angular velocity. It is disabled while the engine is off so it does not fight landing-leg contacts.

This is not a detailed engine-actuator or reaction-control model. It is an idealized high-bandwidth attitude loop.

## 7. Manual guidance

In manual mode:

- Up/Down changes throttle at a bounded rate.
- WASD or the GUI directional pad commands the gimbal direction.
- Releasing directional control slews the engine back toward vertical.
- Horizontal momentum remains until opposite thrust or an automatic hold mode brakes it.

The GUI thrust slider writes the same throttle state as the keyboard controls. Direction buttons and the slider are also indicators: they are drawn from the actual controller state rather than from mouse state.

## 8. Hover/position hold

Hover captures a target position and uses a PD acceleration command:

\[
a_{cmd}=K_p(r_{target}-r)-K_dv.
\]

The unconstrained thrust request is

\[
T_{request}=m(a_{cmd}-g).
\]

The controller then:

1. projects the requested direction into the 20-degree pointing cone;
2. compensates thrust magnitude for the constrained direction;
3. clips magnitude to `[rho1, rho2]`;
4. recomputes the request as mass decreases.

The velocity term provides braking. For example, during an upward climb, `-K_d v_z` reduces thrust below `mg`; gravity removes the upward velocity. Near zero velocity, thrust returns toward `mg`.

In hover mode, WASD moves the horizontal target and Up/Down moves the altitude target. The GUI direction pad and thrust slider display the resulting automatic commands.

## 9. Auto-land state machine

Auto-land uses the same constrained PD controller with two guidance stages.

### Align

The controller holds at least 12 m above the landed body-center height, brakes horizontal velocity, and moves above the center of the landing pad. Descent starts once horizontal position, horizontal speed, altitude, and vertical speed are within their alignment tolerances.

### Descend

The target altitude moves downward at:

- 2.0 m/s above 10 m;
- 1.0 m/s from 3 to 10 m;
- 0.35 m/s below 3 m.

Engine cutoff occurs only when the rocket is:

- within 10 cm of the landing body height;
- descending no faster than 0.35 m/s;
- within 20 cm horizontally;
- nearly stationary horizontally.

MuJoCo then resolves landing-leg contact and settling.

## 10. Relationship to lossless convexification

The paper does more than simulate these equations. It transforms mass and thrust variables and relaxes the nonconvex lower-thrust and pointing constraints into a convex program, then proves that the relaxation is lossless under stated conditions. That produces a fuel-optimal finite-horizon trajectory with terminal and state constraints.

This project does **not** currently construct or solve that second-order cone program. It uses the same plant and control constraints but obtains commands from manual input or PD feedback.

A paper-faithful optimization mode would add:

- a finite prediction horizon and free or selected final time;
- glide-slope and velocity constraints;
- terminal position and zero-velocity constraints;
- transformed mass/thrust variables;
- an SOCP solver;
- receding-horizon replanning or trajectory tracking.

## 11. Is the project 6-DOF?

The answer depends on which layer is being discussed.

### Mechanical simulation: yes

The MuJoCo rocket is a free rigid body with:

- three translational degrees of freedom;
- three rotational degrees of freedom;
- quaternion attitude;
- angular velocity, inertia, torque, collision, and landing-leg contact.

### Guidance model: not fully coupled 6-DOF

The guidance law computes a translational thrust vector and uses an idealized attitude-assist loop to align the body. Thrust is applied at the center of mass, so gimbal-induced moments and engine-pivot dynamics are not physically coupled into translation and rotation.

Therefore, the most accurate description is:

> a 6-DOF MuJoCo rigid-body plant controlled by a paper-style 3-DOF translational guidance law plus idealized attitude assist.

The later successive-convexification papers listed in the main README are appropriate references for upgrading the guidance layer to a fully coupled 6-DOF formulation.

## 12. Verification

The test suite covers:

- MJCF compilation and free-joint dimensions;
- Falcon 9-like height, diameter, and deployed-leg proportions;
- thrust lower and upper bounds;
- pointing-cone enforcement;
- fuel depletion;
- ground settling;
- liftoff without position teleportation;
- hover feedforward and velocity capture;
- end-to-end pad alignment, descent, cutoff, and settling;
- GUI hitboxes and thrust-slider behavior.

Run it with:

```bash
.venv/bin/pytest -q
```
