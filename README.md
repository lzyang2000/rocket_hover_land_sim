# MuJoCo 3-D rocket landing lab

An interactive powered-descent rocket simulator designed to run natively on macOS. It includes a full-scale Falcon 9 first-stage silhouette, manual gimbaled flight, position hold, automatic landing, fuel depletion, landing-leg contact, and live GUI controls.

The MuJoCo vehicle and controller are six-degree-of-freedom. Thrust limits originate with Açıkmeşe, Carson, and Blackmore (2013); hover and landing use a receding-horizon adaptation of the successive-convexification method from Szmuk, Reynolds, and Açıkmeşe (2020).

For equations and paper-to-code mapping, see [METHODS.md](METHODS.md). For the optimizer specification and explicit scope boundaries, see [MPC_DESIGN.md](MPC_DESIGN.md).

## What is included

- MuJoCo free rigid body with 3-D position, quaternion attitude, linear/angular velocity, moving center of mass, inertia, and contact.
- Falcon 9 first-stage proportions: approximately 41.2 m tall, 3.66 m diameter, and 18 m deployed leg span.
- Four grid fins, four landing legs, and a nine-engine base with the center engine shown firing during landing.
- Main-engine force applied at the physical engine pivot, producing coupled pitch/yaw torque.
- Full quaternion attitude and heading control with a physical opposed-thruster RCS roll couple.
- Successive-linearization 6-DOF MPC solved as conic subproblems by CVXPY and Clarabel.
- Gimbaled main-engine thrust constrained to a 20-degree mechanical cone.
- Paper-inspired 20–80% throttle interval with a nonzero minimum after ignition.
- Fuel consumption with shared dry-stage/LOX/RP-1 center-of-mass and inertia modeling in MuJoCo and MPC.
- Keyboard and clickable GUI flight controls.
- Live directional indicators and a thrust slider that follow automatic guidance commands.
- Live 3-D engine arrow whose direction follows gimbal and whose length follows thrust magnitude.
- Three-dimensional hover/position hold.
- Automatic pad alignment, descent, touchdown cutoff, and settling.
- Deterministic 6-DOF fallback control if an MPC solve fails.

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

The simulator process does not hot-reload Python or MJCF changes. Close every existing simulator window before relaunching. The current window title should contain `v0.9.1`.

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

During hover and auto-land, the slider becomes read-only and changes color to indicate automatic ownership. Its knob follows the throttle actually commanded by guidance, including mass compensation and braking.

### 3-D thrust arrow

A colored arrow is anchored at the center engine and follows the actual gimbal command in manual, hover, and auto-land modes. Arrow length and thickness scale with applied thrust, while its color moves from orange toward cyan as thrust increases. It disappears immediately when the engine is off or killed.

The arrow is deliberately drawn outward through the visible plume, so it shows the nozzle/exhaust direction. The reaction force applied to the rocket points in the opposite direction.

## Operating modes

### Manual flight

Ignition jumps directly from zero to the required 20% minimum thrust. Hold Up for roughly two seconds to pass the initial Earth hover point of approximately 40.9%. WASD commands a desired world-frame flight direction. A full SO(3) attitude controller allocates that command to the physical engine gimbal and an opposed RCS force pair for roll.

Releasing WASD returns thrust toward vertical but does not remove existing horizontal momentum. Counter-gimbal or enable hover to brake.

### Hover hold

Press `H` or click `HOVER HOLD`. The controller captures the current 3-D position and starts asynchronous 6-DOF successive-convexification MPC. It predicts mass, position, velocity, quaternion attitude, and angular velocity over a finite horizon, then applies the first bounded thrust/gimbal/roll command.

In hover mode:

- WASD moves the horizontal target at 2 m/s;
- Up/Down moves the altitude target at 2 m/s;
- releasing the controls leaves the target fixed and the controller settles there.

Telemetry reports `SCVX MPC: OPTIMAL` and the latest solve time when the optimizer owns the vehicle. `6-DOF FALLBACK` means the deterministic backup controller is active.

### Automatic landing

Press `L` or click `AUTO LAND`.

The state machine supplies moving position and velocity references to the same 6-DOF MPC. It first holds altitude while braking and aligning over the pad center, then uses an aggressive approach with terminal braking:

- 12 m/s above 30 m;
- 8 m/s from 18 to 30 m;
- 5 m/s from 10 to 18 m;
- 3 m/s from 5 to 10 m;
- 1.5 m/s from 2.5 to 5 m;
- 0.6 m/s from 1 to 2.5 m;
- 0.25 m/s inside the final meter.

The engine cuts directly to zero only when horizontal error is small, horizontal speed is below 0.10 m/s, vertical speed is inside the touchdown window, and the rocket is within 10 cm of its landing body height.

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

- free ignition and free final time;
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

### Hover initially says `WARMING` or `FALLBACK`

The first conic solve includes one-time solver canonicalization and can take a few tenths of a second. The fallback controller remains active until a valid MPC command arrives. Warm solves are normally much faster.

### The rocket does not lift

Initial hover is about 40.9%. Hold Up or drag the thrust slider above that point.

### The engine will not restart

After `KILL ENGINE`, restart is intentionally latched out. Press `R` for a new flight.

### Manual WASD appears to move in an unexpected screen direction

Controls use world X/Y axes. The camera can rotate independently.

## Follow-up fully coupled 6-DOF literature

- M. Szmuk, U. Eren, and B. Açıkmeşe, [*Successive Convexification for Mars 6-DoF Powered Descent Landing Guidance*](https://doi.org/10.2514/6.2017-1500) (2017).
- M. Szmuk and B. Açıkmeşe, [*Successive Convexification for 6-DoF Mars Rocket Powered Landing with Free-Final-Time*](https://doi.org/10.2514/6.2018-0617) (2018).
- M. Szmuk, T. Reynolds, B. Açıkmeşe, M. Mesbahi, and J. M. Carson, [*Successive Convexification for 6-DoF Powered Descent Guidance with Compound State-Triggered Constraints*](https://doi.org/10.2514/6.2019-0926) (2019).
- M. Szmuk, T. P. Reynolds, and B. Açıkmeşe, [*Successive Convexification for Real-Time Six-Degree-of-Freedom Powered Descent Guidance with State-Triggered Constraints*](https://doi.org/10.2514/1.G004549) (2020).
