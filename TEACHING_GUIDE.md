# Teaching guide: 6-DOF rocket control experiments

This guide turns the simulator into a sequence of reproducible control and robotics labs. It assumes the installation steps in [README.md](README.md) are complete.

Use [METHODS.md](METHODS.md) as the mathematical reference and [MPC_DESIGN.md](MPC_DESIGN.md) as the optimizer specification. The labs below emphasize observation, prediction, and explanation before code modification.

## Learning objectives

By the end of the sequence, a student should be able to:

- distinguish the plant, guidance, controller, and actuator layers;
- explain why a minimum-thrust engine creates discrete and continuous control modes;
- explain translation–attitude coupling for a gimbaled rocket;
- compare reactive PD control with finite-horizon MPC;
- explain successive convexification, virtual control, warm starts, and nonlinear defect;
- derive a stopping-distance ignition condition for ballistic landing;
- identify which simulator assumptions prevent direct comparison with Falcon 9;
- propose the state, force, estimation, and control additions required for wind and drag.

## Before each lab

Start from the project root and close any simulator window left open from a previous run. The process does not hot-reload code.

Run the simulator with synchronous MPC:

```bash
uv run rocket-landing
```

or with the existing virtual environment:

```bash
.venv/bin/rocket-landing
```

Run individual regressions with:

```bash
.venv/bin/pytest -q path/to/test.py::test_name
```

For every experiment, record:

- initial mode and state;
- controller-owner indicator;
- maximum observed tilt and vertical speed;
- fuel before and after the maneuver;
- whether the result matched your prediction;
- which assumption most limits the realism of the result.

## Lab 1: Hybrid thrust and fuel flow

### Question

Why can the engine be at zero thrust before ignition but not smoothly pass from 20% thrust to zero while lit?

### Procedure

1. Reset the rocket.
2. Ignite with `I`, Up, or the GUI button.
3. Hold Down or drag the slider fully left.
4. Observe that throttle stops at 20%.
5. Press `K` and observe the discontinuous transition to zero.
6. Reset, ignite again, and let auto-land enter `COAST`. Compare coast with permanent kill.

### Expected observations

- `OFF → LIT` is a discrete transition.
- While `LIT`, thrust remains in the bounded interval 144–576 kN.
- `COAST` has zero thrust but preserves permission to relight.
- `SHUTDOWN` has zero thrust and cannot relight until reset.
- Fuel flow follows thrust magnitude and becomes zero during coast.

### Code reading

- `src/rocket_landing/controller.py`: `EngineState`, `ignite()`, `begin_coast()`, `relight()`, and `consume_fuel()`.
- `tests/test_controller.py`: minimum-thrust, fuel-flow, coast, and kill regressions.

### Check your understanding

1. Why is the valid thrust set nonconvex if zero and positive bounded thrust are represented in one continuous variable?
2. Why does a state machine make the on/off physics easier to explain?
3. Which transition is reversible during one flight: coast or kill?

### Regression

```bash
.venv/bin/pytest -q tests/test_controller.py
```

## Lab 2: Translation–attitude coupling and roll control

### Question

Why does the engine sometimes gimbal opposite the desired lateral travel direction before the rocket begins moving that way?

### Procedure

1. Lift off and enable hover.
2. Tap `W` briefly while watching the direction indicator, gimbal angle, and body tilt.
3. Release `W` and observe the braking transient.
4. Repeat with a longer command and compare the response.
5. Observe roll behavior separately; the main engine cannot create roll torque about its own axis.

### Expected observations

- Gimbal force acts at a moment arm below the center of mass.
- The controller first creates attitude change, then horizontal thrust direction.
- A tall rocket is non-minimum-phase for lateral commands: the initial actuator direction can oppose the later translation.
- Roll comes from the opposed RCS force pair, not the single main engine.

### Code reading

- `src/rocket_landing/sim.py`: `_attitude_control_torque_body()`, `_allocate_attitude_control()`, and `_apply_roll_rcs()`.
- `src/rocket_landing/mpc.py`: `continuous_dynamics()` and `gimbal_direction_body()`.
- `src/rocket_landing/assets/rocket.xml`: engine and RCS sites.

### Check your understanding

1. What torque is produced by applying thrust through the center of mass?
2. Why do equal and opposite RCS forces create roll without net translation?
3. Why would a point-mass 3-DOF model miss the counter-gimbal transient?

### Regressions

```bash
.venv/bin/pytest -q tests/test_mpc.py::test_nonlinear_model_couples_engine_gimbal_to_attitude
.venv/bin/pytest -q tests/test_model.py -k "roll or heading"
```

## Lab 3: PD control versus MPC

### Question

What does MPC add if a 200 Hz PD controller can already hover and land the rocket?

### Procedure

1. Enable hover and move the target with WASD.
2. Watch the controller indicator switch between `MPC ACTIVE` and `PD ACTIVE`.
3. Run the same maneuver with `--async-mpc`.
4. Compare responsiveness, rendering smoothness, and rejection messages.
5. During a landing, distinguish `PD ACTIVE` from `TERMINAL ACTIVE`.

### Expected observations

- PD reacts to current position and velocity error every 5 ms.
- MPC predicts a bounded trajectory over a 2.8 s horizon.
- Synchronous MPC commands come from the current state but may pause rendering.
- Async MPC preserves rendering responsiveness but can reject stale or mismatched trajectories.
- Terminal PD is a scheduled low-altitude mode, not an optimizer failure.

### Mathematical comparison

The translational PD law is

\[
a_d=a_{ff}+K_p(r_d-r)+K_v(v_d-v).
\]

It is reactive: it does not explicitly represent future constraints. MPC instead chooses a sequence of bounded controls and applies only the first command before replanning.

### Code reading

- `src/rocket_landing/sim.py`: `_pd_hover_guidance()` and `_update_hover_controller()`.
- `src/rocket_landing/mpc.py`: `SixDofMPC.solve()`.
- `METHODS.md`: Sections 7–9.

### Check your understanding

1. Why can PD reject a modest disturbance without predicting it?
2. Why can steady wind leave a PD-controlled position offset without integral action?
3. Why is a short-horizon optimizer not automatically superior in every flight phase?

### Regressions

```bash
.venv/bin/pytest -q tests/test_hover.py
.venv/bin/pytest -q tests/test_landing.py::test_mpc_offset_alignment_is_bounded_and_rarely_uses_pd
```

## Lab 4: Successive convexification and solution acceptance

### Question

Why is an `optimal` convex-solver status insufficient evidence that a command is safe for the nonlinear rocket?

### Procedure

1. Run a high-energy fuel-auto landing.
2. Observe periods of `MPC ACTIVE` and `PD ACTIVE`.
3. Read the virtual-control and nonlinear-defect discussion in the methods document.
4. Inspect the high-energy MPC regression.

### Expected observations

- SCvx linearizes dynamics around a nominal trajectory.
- Virtual control keeps a convex subproblem feasible but is not a real actuator.
- Optimized controls are independently rolled through the nonlinear prediction model.
- Results with excessive nonlinear defect are rejected.
- The current high-energy regression accepts more than 60% of above-terminal attempts while retaining the PD safety path.

### Thought experiment

Consider lowering the virtual-control penalty. The convex optimizer can reduce tracking cost by inserting artificial state motion. The convex problem may look better while the nonlinear rollout becomes worse. Increasing the defect threshold would hide the symptom rather than fix the model inconsistency.

### Code reading

- `src/rocket_landing/mpc.py`: `_build_problem()` and the final acceptance calculation in `solve()`.
- `src/rocket_landing/sim.py`: `_apply_mpc_result()` and async rejection checks.
- `MPC_DESIGN.md`: Sections 4–5.

### Check your understanding

1. What physical device corresponds to virtual control? None.
2. What is the difference between virtual-control magnitude and nonlinear defect?
3. Why is “accept more MPC solves” unsafe if achieved only by raising the defect threshold?

### Regressions

```bash
.venv/bin/pytest -q tests/test_mpc.py
.venv/bin/pytest -q tests/test_landing.py::test_high_energy_landing_mpc_rejects_virtual_state_teleportation
```

## Lab 5: Ballistic coast and landing energy

### Question

When should the engine relight if the rocket is falling rapidly from high altitude?

### Procedure

1. Climb vertically above 40 m and start auto-land.
2. Observe engine cutoff, constant fuel during coast, and automatic relight.
3. Repeat with a much higher centered state or run the 1,000 m landing-only regression. Use `LAUNCH + RETURN` separately to observe the 130 km full-stack mission.
4. Compare the ordinary descent-speed bands with the post-coast energy corridor.

### Expected observations

- Relight height grows approximately with downward speed squared.
- Ignition delay and a fixed height margin move ignition earlier than the ideal stopping-distance result.
- A long coast reaches a much higher speed than the ordinary 12 m/s descent band.
- The energy corridor preserves aggressive descent and progressively reduces allowable speed near the ground.

### Key equation

\[
h_i(v)=1.25\frac{\max(v^2-v_t^2,0)}{2(T_{max}/m-g)}
+v\tau_i+\frac{1}{2}g\tau_i^2+6\text{ m}.
\]

Identify the ideal stopping-distance term, ignition-delay distance, and fixed margin separately.

### Check your understanding

1. Why does lower vehicle mass permit later ignition for the same speed?
2. Why is coast permitted only when the vehicle is upright and dynamically quiet?
3. Why does the current roll RCS not provide pitch/yaw stabilization during coast?

### Regressions

```bash
.venv/bin/pytest -q tests/test_landing.py::test_high_altitude_auto_land_coasts_without_fuel_then_relights
.venv/bin/pytest -q tests/test_landing.py::test_1000m_ballistic_coast_uses_energy_corridor_and_lands
```

## Lab 6: Fuel-auto takeover

### Question

Why does a centered vertical emergency use a different fuel estimate from an offset or tilted rocket?

### Procedure

1. Reset, command full throttle, and do not intervene.
2. Record fuel, altitude, and vertical velocity when fuel auto takes over.
3. Observe that a centered, quiet state enters coast immediately instead of burning fuel in ALIGN.
4. Compare with an offset landing regression that must retain powered lateral alignment.

### Expected observations

- Centered vertical flight uses a direct ballistic-coast estimate.
- Lateral error increases the required controllability reserve.
- The full-throttle regression lands with less than 100 kg in the current deterministic model.
- This narrow reserve is a simulator regression target, not a real-flight safety recommendation.

### Check your understanding

1. Why can altitude be cheap in fuel while lateral alignment is expensive?
2. Why did a long minimum-thrust ALIGN waste fuel during upward flight?
3. Why must an empirical fuel estimator be validated with complete closed-loop rollouts?

### Regressions

```bash
.venv/bin/pytest -q tests/test_landing.py::test_full_throttle_launch_fuel_auto_lands_below_100kg_reserve
.venv/bin/pytest -q tests/test_landing.py::test_real_fuel_reserve_takeover_keeps_enough_fuel_to_land
```

## Lab 7: Design an atmospheric extension

This is a design exercise; the current simulator is a vacuum model.

### Minimum physics additions

1. Define relative air velocity:

   \[
   v_{air}=v-w.
   \]

2. Add a density model $\rho(h)$.
3. Add drag and, if desired, lift:

   \[
   F_D=-\frac{1}{2}\rho C_D A\lVert v_{air}\rVert v_{air}.
   \]

4. Apply aerodynamic force at a center of pressure so it produces attitude moment.
5. Add grid-fin force and moment authority scheduled by dynamic pressure.

### Minimum estimation and control additions

- estimate wind or net disturbance;
- add integral action or a disturbance observer to PD;
- include aerodynamic forces and moments in the MPC prediction model;
- impose angle-of-attack, dynamic-pressure, and actuator constraints;
- consider robust or tube MPC for bounded wind uncertainty;
- validate with gusts, parameter error, sensor noise, and actuator delay.

### Design questions

1. Which aerodynamic parameters are observable from position and attitude alone?
2. When does more grid-fin authority help, and when does low dynamic pressure make it ineffective?
3. How should controller gains change between thin air, max-Q, and terminal descent?
4. What failure criterion should replace a single nominal landing test?

## Lab 8: Pitch-over, separation, and boost-back

### Question

Why does a launch vehicle pitch away from vertical, and how can the booster still return to the launch site?

### Procedure

1. Press `J` from reset and watch the first kilometre remain vertical.
2. Observe pitch increase smoothly toward 18° and note the growing downrange position and horizontal velocity.
3. At separation, watch the upper stack become an independent rigid body and continue upward with a small separation push.
4. Observe `BOOSTBACK ACTIVE`: the booster slews retrograde and moves the predicted ballistic impact point toward the pad.
5. Count engine ignitions in telemetry. The full mission permits exactly two: launch and the combined reentry/landing burn.

### Expected observations

- Pitch creates horizontal velocity, so ballistic flight becomes an arc rather than a vertical line.
- The separated upper body preserves position, attitude, velocity, and angular rate; it does not disappear.
- Boost-back changes the predicted impact point before the long engine-off coast.
- High-altitude `COAST`/`DESCEND` chatter is impossible after the second ignition because no third relight is available.
- This remains a suborbital vacuum demonstration, not an orbital gravity turn or powered upper-stage simulation.

### Check your understanding

1. Why does the booster apogee differ from the 130 km cutoff prediction after boost-back thrust is applied?
2. Why must horizontal pad interception account for the longer flight time created by powered vertical braking?
3. What additional state and constraints would a true ascent-and-entry trajectory optimizer require?

## Assessment prompts

Use these prompts for a report, oral examination, or project review:

1. Draw the information flow from auto-land phase logic to MuJoCo state feedback.
2. Derive the ideal one-dimensional stopping distance and identify every safety term added by the implementation.
3. Explain why the vehicle is mechanically 6-DOF even though MuJoCo stores seven position coordinates.
4. Explain why the main engine controls pitch and yaw but not roll.
5. Compare solver status, virtual control, and nonlinear defect.
6. Defend one case where PD should own the vehicle instead of MPC.
7. Identify three simulator conclusions that should not be generalized to Falcon 9.
8. Propose a test matrix for wind, drag-coefficient error, and sensor noise.

## Extension projects

- Add telemetry logging and plot fuel, thrust, gimbal, tilt, and controller ownership.
- Add a simple exponential atmosphere and axial drag.
- Add a configurable horizontal wind and PD integral term.
- Add center-of-pressure torque and compare passive stability across COM locations.
- Add grid-fin actuators with dynamic-pressure scheduling.
- Compare fixed-horizon MPC with a longer offline landing optimization.
- Replace the empirical fuel estimator with a reachable-set or optimal-control calculation.
- Add randomized Monte Carlo landing tests and report success probability rather than one trajectory.

Keep extensions explicit about assumptions. A teaching result is strongest when the model boundary is as clear as the successful demonstration.
