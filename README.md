# MuJoCo 3-D rocket landing lab

An interactive powered-descent rocket simulator designed to run natively on macOS. It includes a full-scale Falcon 9 first-stage silhouette, manual gimbaled flight, position hold, automatic landing, fuel depletion, landing-leg contact, and live GUI controls.

The MuJoCo vehicle and controller are six-degree-of-freedom. Thrust limits originate with Açıkmeşe, Carson, and Blackmore (2013); hover and landing use a receding-horizon adaptation of the successive-convexification method from Szmuk, Reynolds, and Açıkmeşe (2020).

For equations and paper-to-code mapping, see [METHODS.md](METHODS.md). For the optimizer specification and explicit scope boundaries, see [MPC_DESIGN.md](MPC_DESIGN.md).

## What is included

- MuJoCo free rigid body with 3-D position, quaternion attitude, linear/angular velocity, moving center of mass, inertia, and contact.
- Falcon 9 first-stage proportions: approximately 41.2 m tall, 3.66 m diameter, and 18 m deployed leg span.
- Four horizontally deployed grid fins, four folding landing legs, and a nine-engine base with the center engine shown firing during landing.
- Main-engine force applied at the physical engine pivot, producing coupled pitch/yaw torque.
- Full quaternion attitude and heading control with a physical opposed-thruster RCS roll couple.
- Successive-linearization 6-DOF MPC solved as conic subproblems by CVXPY and Clarabel.
- Optional asynchronous MPC trajectory guidance with a deterministic 200 Hz
  thrust, gimbal, and roll inner loop.
- Gimbaled main-engine thrust constrained to a 20-degree mechanical cone.
- Paper-inspired 20–80% throttle interval with a nonzero minimum after ignition.
- Fuel consumption with shared dry-stage/LOX/RP-1 center-of-mass and inertia modeling in MuJoCo and MPC.
- Keyboard and clickable GUI flight controls.
- DPI-aware responsive GUI sizing for Retina, standard-density, and smaller displays.
- Live directional indicators and a thrust slider that follow automatic guidance commands.
- Live 3-D engine arrow whose direction follows gimbal and whose length follows thrust magnitude.
- Three-dimensional hover/position hold.
- Automatic pad alignment, descent, touchdown cutoff, and settling.
- Relightable high-altitude ballistic coast with a dynamic landing-burn gate.
- Automatic landing takeover at 1.05 times the estimated landing-fuel reserve.
- Deterministic 6-DOF fallback control if an MPC solve fails.
- A controller-owner badge that reports `MPC ACTIVE`, `TERMINAL ACTIVE`,
  `FALLBACK ACTIVE`, or `MANUAL TVC`.

## Requirements

- macOS, Linux, or Windows with desktop OpenGL support.
- Python 3.10 or newer.
- An Intel or Apple Silicon Mac is supported by current MuJoCo wheels.

The project has no CUDA or NVIDIA requirement. CVXPY installs native CPU conic solvers; no external commercial solver is required.

## Setup with `uv`—recommended

Install `uv` on macOS:

```bash
brew install uv
```

From the project directory:

```bash
cd /Users/yangl/Documents/rocket
uv sync --extra dev
uv run rocket-landing
```

The `dev` extra installs pytest. If you only want to run the simulator, `uv sync` is sufficient.

The default launcher uses synchronous MPC so every optimized command is based on the current MuJoCo state and current GUI target. It precompiles the conic problem before opening the window, moving the one-time cold-start cost out of the first hover command.

For responsive rendering and input while the optimizer runs in the background, use:

```bash
uv run rocket-landing --async-mpc
```

In asynchronous mode the optimizer supplies a timestamped nonlinear reference trajectory. A deterministic controller runs at every 5 ms MuJoCo step, compensates for solver latency, rejects stale or inconsistent predictions, shifts guidance toward the latest GUI target, and owns the actual bounded thrust, gimbal, and roll commands. Old raw MPC actuator commands are never applied after a delayed solve.

## Setup with a standard virtual environment

```bash
cd /Users/yangl/Documents/rocket
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
rocket-landing
```

This workspace already has a local environment, so it can also be launched directly:

```bash
cd /Users/yangl/Documents/rocket
.venv/bin/rocket-landing
```

The custom GLFW viewer renders on the main thread, so macOS does not require MuJoCo's special `mjpython` launcher.

## Important when updating

The simulator process does not hot-reload Python or MJCF changes. Close every existing simulator window before relaunching. The current window title should contain `v0.9.21`.

The initial window is limited to the monitor's usable work area. Control widths, font resolution, and telemetry wrapping are derived from the actual GLFW window and framebuffer sizes, so the right-side labels should remain visible on both Retina and standard-density displays.

## Controls

All keyboard flight controls have clickable equivalents in the right-side GUI.

| Keyboard / GUI | Manual mode | Hover mode | Auto-land |
| --- | --- | --- | --- |
| `I`, Up, or `IGNITE ENGINE` | Ignite | Already controlled | Already controlled |
| Up / Down | Increase/decrease throttle | Move altitude target | Ignored |
| `W` / GUI `W` | Gimbal toward world `+Y` | Move target toward `+Y` | Indicator only |
| `S` / GUI `S` | Gimbal toward world `-Y` | Move target toward `-Y` | Indicator only |
| `A` / GUI `A` | Gimbal toward world `-X` | Move target toward `-X` | Indicator only |
| `D` / GUI `D` | Gimbal toward world `+X` | Move target toward `+X` | Indicator only |
| `H` / `HOVER HOLD` | Capture current position | Disable hold | Cancel guidance to hold |
| `L` / `AUTO LAND` | Begin automatic landing | Begin automatic landing | Cancel to hover |
| `K` / `KILL ENGINE` | Cut directly to zero | Abort and cut | Abort and cut |
| `R` | Reset vehicle, fuel, and engine | Reset | Reset |
| `Esc` | Close window | Close window | Close window |

### Directional pad

Click and hold a direction button exactly like holding its keyboard key. The buttons illuminate from the current world-direction guidance demand:

- manual input produces the indicator directly;
- hover guidance illuminates the directions used to brake and hold position;
- auto-land illuminates the directions used to align over the pad.

The telemetry line separately reports the actual mechanical gimbal angle and body tilt. During a transient, the engine may briefly counter-steer to create the torque required to rotate the long stage.

The mapping is in fixed world axes, not camera-relative axes. Rotating the camera can therefore make `W` appear sideways on screen.

### Thrust slider

In manual mode, click or drag the slider to command any throttle from 20% to 80%. Moving the slider automatically ignites an engine that is still off.

During hover and auto-land, the slider becomes read-only and changes color to indicate automatic ownership. Its knob follows the throttle actually commanded by guidance, including mass compensation and braking. During ballistic coast it moves to zero and reads `OFF`; the engine remains armed for an automatic relight.

When the engine is off, killed, fuel-depleted, or shut down after landing, the slider returns to its zero position and reads `THRUST 0.0% OFF` rather than retaining the last powered throttle command.

### 3-D thrust arrow

A colored arrow is anchored at the center engine and follows the actual lagged gimbal state in manual, hover, and auto-land modes. Arrow length and thickness scale with applied thrust, while its color moves from orange toward cyan as thrust increases. It disappears immediately when the engine is off or killed.

The arrow is deliberately drawn outward through the visible plume, so it shows the nozzle/exhaust direction. The reaction force applied to the rocket points in the opposite direction.

## Operating modes

### Manual flight

Ignition jumps directly from zero to the required 20% minimum thrust. Hold Up for roughly two seconds to pass the initial Earth hover point of approximately 40.9%. WASD commands a desired world-frame flight direction. A full SO(3) attitude controller allocates that command to the physical engine gimbal and an opposed RCS force pair for roll.

Releasing WASD returns thrust toward vertical but does not remove existing horizontal momentum. Counter-gimbal or enable hover to brake.

### Hover hold

Press `H` or click `HOVER HOLD`. The controller captures the current 3-D position and starts synchronous 6-DOF successive-convexification MPC by default. It predicts mass, position, velocity, quaternion attitude, and angular velocity over a finite horizon, then applies the first bounded thrust/gimbal/roll command before physics advances again.

With `--async-mpc`, SCvx becomes the slower trajectory-guidance layer and the 200 Hz deterministic 6-DOF controller becomes the actuator layer. Telemetry reports this as `SCVX MPC ASYNC+INNER`; rejected or unavailable predictions switch immediately to `6-DOF FALLBACK` with the rejection reason when available.

In hover mode:

- WASD moves the horizontal target at 3 m/s;
- Up/Down moves the altitude target at 2 m/s;
- releasing the controls leaves the target fixed and the controller settles there.

WASD remains position-target driven: the position error and measured rocket velocity provide the PD/MPC feedback needed to track the moving waypoint. The controller does not falsely label the waypoint itself as travelling at 3 m/s. The horizontal target is limited to a 3.5 m lead and the altitude target to a 2 m lead relative to the rocket; holding a key keeps advancing this bounded “carrot” as the vehicle moves instead of allowing a large error to accumulate. Manual WASD steering also reaches its requested gimbal direction 50% faster. Because the gimbaled engine is below the center of mass, the rocket still briefly counter-gimbals to create the required tilt, but hover MPC commands are limited to 5° of gimbal to suppress swinging. The automatic actuator/fallback envelope and high-altitude landing limit remain 6°.

Telemetry reports `SCVX MPC SYNC: OPTIMAL` and the latest solve time when the default optimizer owns the vehicle. `SCVX MPC ASYNC` appears when the optional background-worker mode is selected. `6-DOF FALLBACK` means a solve was unavailable or rejected. `6-DOF TERMINAL` is the intentional low-altitude controller handoff. The right-side ownership badge makes the current state explicit: green `MPC ACTIVE`, blue `TERMINAL ACTIVE`, orange `FALLBACK ACTIVE`, or gray `MANUAL TVC`.

Synchronous mode can pause rendering and input briefly during a solve—typically around 70–150 ms after warm-up on the development Mac—but it avoids applying a command computed from an older rocket state and an older WASD target. Async mode keeps the window smoother at the cost of that command latency.

### Automatic landing

Press `L` or click `AUTO LAND`.

The state machine supplies moving position and velocity references to the 6-DOF MPC. Normal auto-land first climbs or holds at a staging height at least 25 m above the pad. If LAND is clicked during a rapid manual ascent, staging is raised to the predicted minimum-thrust braking apex; this captures the upward trajectory instead of requiring the rocket to overshoot and later return to the obsolete click altitude. ALIGN uses a four-metre horizontal lead through 18 m altitude; above that, the lead grows by 0.15 m per additional metre and is capped at 8 m. The rocket must complete the lateral capture within 2 m, reduce horizontal speed below 1.0 m/s, and settle within 2 m and 1.5 m/s of the staging altitude before descent begins. Alignment is therefore completed high rather than deferred into the final approach. Fuel-reserve takeover keeps the current altitude unless the current upward stopping trajectory is necessarily higher.

If alignment finishes above 32 m while the stage is nearly upright and dynamically quiet, auto-land enters `COAST`: main-engine thrust and fuel flow go to zero, the GUI reports `LAND: COAST`, the thrust slider moves to zero, and the roll RCS remains available. The engine automatically relights at maximum commanded throttle when the current stopping-distance gate is reached. After a long coast, descent follows an energy-based speed corridor rather than immediately braking to the fixed 12 m/s band; this lets a 1,000 m vacuum drop peak near 90 m/s, relight around 590 m, and progressively brake to landing without wasting fuel in a several-hundred-metre powered hover-like descent. Tilt above 5°, angular rate above 0.25 rad/s, lateral error above 2.75 m, or horizontal speed above 1.5 m/s commands an early relight. Low-altitude landings skip coast entirely. `KILL ENGINE` remains a permanent shutdown and is deliberately separate from this armed coast state.

This high-altitude result is a vacuum rigid-body demonstration. Aerodynamic drag, wind, grid-fin authority, and atmospheric attitude stabilization are not yet modeled, so it should not be read as a realistic Falcon 9 entry simulation.

After relight, guidance uses an aggressive approach with terminal braking:

- 12 m/s above 30 m;
- 8 m/s from 18 to 30 m;
- 5 m/s from 10 to 18 m;
- 3 m/s from 5 to 10 m;
- 1.5 m/s from 2.5 to 5 m;
- 0.6 m/s from 1 to 2.5 m;
- 0.25 m/s inside the final meter.

Crossing a band boundary changes the reference directly to the next nonzero descent speed; guidance does not insert a zero-velocity hold between bands. The altitude reference remains a continuously integrated trajectory, so residual lateral alignment does not reset or jump the vertical target.

Inside the optimizer, a nonzero reference velocity now advances the reference position at every prediction node. This removes the former contradiction that asked the MPC to remain at one fixed altitude while simultaneously tracking a downward velocity. MPC controls hover, alignment, and descent down to 7 m. The deterministic coupled 6-DOF controller then owns the final approach, where its direct feedback is more robust than the short-horizon optimizer under the progressively tight 3°, 1.5°, and 0.75° terminal gimbal limits.

The four landing legs remain folded upward against the fuselage during reset, manual flight, hover, ALIGN, and descent above the 7 m terminal handoff. Entering terminal descent commands a latched 1.25 s deployment down and outward; it continues even if landing guidance is cancelled after deployment begins. The same moving MuJoCo geoms provide the visible legs and their collision contacts, and telemetry reports `LEGS STOWED`, deployment percentage, or `LEGS DEPLOYED`. Reset is the only command that folds them again. An invisible fixed support under the engine section holds the stowed vehicle at startup, then disables permanently at actual liftoff or when auto-land begins.

While the engine is lit outside auto-land, the simulator estimates the propellant needed to align, perform only the powered part of descent below the predicted coast ignition point, and brake excess velocity. If fuel remaining falls to `1.05 ×` that estimate while the rocket is above the end-burn cutoff height, auto-land takes over once and latches. A reserve takeover aligns at the current altitude rather than climbing to the normal staging height. The retuned estimate uses 2.5 seconds of capture/settling allowance, 10% descent-time and control-impulse margins, and a 250 kg terminal reserve. Its controllability floor now charges relatively little for coastable altitude but more strongly for lateral offset, which is the condition that actually destabilizes low-fuel capture in rollout tests. In the standard `(4, -3, 15 m)` regression state, takeover is about 7.48 tonnes rather than 7.76 tonnes; an aligned 1,000 m vacuum descent triggers around 6.5 tonnes. This remains a controller-specific heuristic—not a certified propellant-to-go result from the MPC.

To prevent low-altitude hunting, the physical gimbal follows commands through an actuator lag and is limited to 3° below 5 m, 1.5° below 2.5 m, and 0.75° in the final meter. Very small terminal commands enter a 0.15° deadband.

The engine normally cuts directly to zero once horizontal error is below 0.50 m, horizontal speed is below 0.30 m/s, vertical speed is between -0.50 and +0.15 m/s, and the rocket is within 15 cm of its landing body height. Fuel-reserve takeover uses an emergency envelope of 1.0 m, 0.60 m/s, and 30 cm so it does not spend its remaining reserve hovering for cosmetic pad precision. The legs and pad friction settle the residual motion.

## Falcon 9-like dimensions and dynamics

The exterior uses public Falcon 9 first-stage dimensions and recognizable proportions. The dynamics use an approximate depleted landing configuration, not a claim of exact SpaceX mass properties or engine control limits.

| Parameter | Value |
| --- | ---: |
| Stage height | approximately 41.2 m |
| Tank diameter | 3.66 m |
| Height/diameter ratio | approximately 11.26 |
| Deployed leg span | approximately 17–18 m |
| Nominal engine scale | 720,000 N |
| Allowed throttle | 20–80% |
| Allowed thrust | 144,000–576,000 N |
| Pointing half-angle | 20 degrees |
| Initial wet mass | 30,000 kg |
| Approximate dry mass | 21,000 kg |
| Initial landing propellant | 9,000 kg |
| Dry-stage COM offset | approximately 0.89 m below the initial reference |
| Roll RCS lever arm | 1.75 m per side |
| Maximum force per modeled RCS pod | 5,000 N |
| Maximum roll moment | 17,500 N m |
| RCS response time constant | 0.10 s |
| Gravity | 9.81 m/s² |
| Initial hover throttle | approximately 40.9% |
| Fuel coefficient `alpha` | `5e-4` |

Mass and thrust are both 30 times the original paper-example scale, preserving the same thrust-to-weight ratio while making the vehicle mass more representative of a depleted booster. Initial inertia is matched to the tall stage. As fuel drains, effective LOX and RP-1 liquid columns shorten toward their tank bottoms; MuJoCo and the MPC use the resulting shared COM, inertia, and engine lever arm rather than a uniform inertia-to-mass scale.

The tank intervals and RCS force level are transparent engineering assumptions, not published Block 5 specifications. A real Falcon 9 combines phase-dependent differential engine gimballing, aerodynamic grid-fin authority, and cold-gas attitude control; this single-engine landing model uses the opposed force pair as the explicit low-authority axial actuator.

The 20–80% throttle interval still comes from the paper-inspired educational model. A real Falcon 9 landing burn has different engine limits and generally cannot settle into a sustained hover because minimum Merlin thrust can exceed the nearly empty stage's weight. This simulator deliberately retains hover mode for control experiments.

## Is it 6-DOF?

Yes at both the mechanical and feedback-control layers:

- the MuJoCo model has a free joint with three translation and three rotation degrees of freedom;
- attitude is represented by a quaternion;
- angular velocity, fuel-dependent inertia/COM, torque, collision, and landing-leg contact are simulated;
- engine force is applied at the gimbal pivot rather than at the center of mass;
- gimbal force creates physical pitch/yaw moments;
- two opposed forces at physical RCS sites create a bounded zero-net-force roll couple with actuator lag;
- MPC predicts `mass + position + velocity + quaternion + angular velocity`, a 14-state 6-DOF model.

The optimizer is a practical receding-horizon adaptation, not a verbatim reproduction of the paper's full free-final-time problem. It uses numerical dynamics linearization, trust regions, virtual control, conic thrust/gimbal/tilt/rate constraints, warm starts, and repeated replanning.

The precise description is: **a coupled 6-DOF MuJoCo plant controlled by SCvx-inspired 6-DOF MPC, with fuel-dependent mass properties, a physical gimbaled-engine moment arm, and a lagged RCS roll couple.**

## Relationship to the paper

Implemented from the 2013 and 2020 papers:

- translational powered-descent dynamics;
- mass depletion proportional to thrust magnitude;
- nonzero lower and upper thrust bounds;
- thrust/gimbal cone;
- quaternion rigid-body dynamics;
- engine-pivot torque coupling;
- repeated dynamics linearization and conic subproblems;
- virtual control for artificial infeasibility;
- scaled trust regions and warm starts;
- receding-horizon application of the first optimized command.

Not yet implemented:

- optimizer-selected free ignition and free final time (auto-land uses an
  explicit stopping-distance ignition gate);
- atmospheric lift/drag and angle-of-attack state-triggered constraints;
- exact first-order-hold discretization matrices;
- formal convergence or global-optimality guarantees.

Read [METHODS.md](METHODS.md) for the derivation and code correspondence.

## Run the tests

With the existing environment:

```bash
.venv/bin/pytest -q
```

Or with `uv`:

```bash
uv run pytest -q
```

## Project layout

```text
.
├── README.md                         setup, controls, and operation
├── METHODS.md                        equations and paper-to-code mapping
├── MPC_DESIGN.md                     MPC specification and acceptance criteria
├── pyproject.toml                    dependencies and command entry point
├── src/rocket_landing/
│   ├── controller.py                 thrust bounds, gimbal cone, and fuel
│   ├── mass_properties.py            dry stage and draining LOX/RP-1 model
│   ├── mpc.py                        14-state SCvx MPC and nonlinear model
│   ├── sim.py                        actuators, fallback, GUI, and rendering
│   └── assets/rocket.xml             vehicle, pad, contacts, and visuals
└── tests/                             physics, guidance, landing, and GUI tests
```

## Troubleshooting

### The window shows an older version

Close all Python/MuJoCo simulator windows and relaunch. Existing processes retain the old code.

### Startup takes a moment before the window appears

The launcher performs one throwaway MPC solve to compile and cache CVXPY's conic problem before creating the window. This takes a few tenths of a second but prevents the one-time canonicalization cost from occurring when hover is first enabled.

### The rocket does not lift

Initial hover is about 40.9%. Hold Up or drag the thrust slider above that point.

### The engine will not restart

After `KILL ENGINE`, restart is intentionally latched out. Press `R` for a new flight. The automatic `COAST` phase is different: it keeps the engine armed and relights without user input.

### Manual WASD appears to move in an unexpected screen direction

Controls use world X/Y axes. The camera can rotate independently.

## Follow-up fully coupled 6-DOF literature

- M. Szmuk, U. Eren, and B. Açıkmeşe, [*Successive Convexification for Mars 6-DoF Powered Descent Landing Guidance*](https://doi.org/10.2514/6.2017-1500) (2017).
- M. Szmuk and B. Açıkmeşe, [*Successive Convexification for 6-DoF Mars Rocket Powered Landing with Free-Final-Time*](https://doi.org/10.2514/6.2018-0617) (2018).
- M. Szmuk, T. Reynolds, B. Açıkmeşe, M. Mesbahi, and J. M. Carson, [*Successive Convexification for 6-DoF Powered Descent Guidance with Compound State-Triggered Constraints*](https://doi.org/10.2514/6.2019-0926) (2019).
- M. Szmuk, T. P. Reynolds, and B. Açıkmeşe, [*Successive Convexification for Real-Time Six-Degree-of-Freedom Powered Descent Guidance with State-Triggered Constraints*](https://doi.org/10.2514/1.G004549) (2020).
