"""Successive-linearization 6-DOF model predictive controller.

The formulation follows the practical structure of Szmuk, Reynolds, and
Açıkmeşe (2020): quaternion rigid-body dynamics, thrust applied at an engine
moment arm, convex conic subproblems, virtual control, and trust regions.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
import warnings

import cvxpy as cp
import numpy as np

from rocket_landing.mass_properties import MassProperties, RocketMassModel


MASS = 0
POSITION = slice(1, 4)
VELOCITY = slice(4, 7)
QUATERNION = slice(7, 11)
ANGULAR_VELOCITY = slice(11, 14)
NX = 14

THRUST = 0
GIMBAL = slice(1, 3)
ROLL_TORQUE = 3
NU = 4


def normalize_quaternion(quaternion: np.ndarray) -> np.ndarray:
  """Return a normalized scalar-first quaternion with a stable sign."""

  result = np.asarray(quaternion, dtype=float).copy()
  norm = float(np.linalg.norm(result))
  if norm < 1e-12:
    result[:] = (1.0, 0.0, 0.0, 0.0)
  else:
    result /= norm
  if result[0] < 0.0:
    result *= -1.0
  return result


def quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
  """Hamilton product of scalar-first quaternions."""

  lw, lx, ly, lz = np.asarray(left, dtype=float)
  rw, rx, ry, rz = np.asarray(right, dtype=float)
  return np.array(
    [
      lw * rw - lx * rx - ly * ry - lz * rz,
      lw * rx + lx * rw + ly * rz - lz * ry,
      lw * ry - lx * rz + ly * rw + lz * rx,
      lw * rz + lx * ry - ly * rx + lz * rw,
    ],
    dtype=float,
  )


def quaternion_to_matrix(quaternion: np.ndarray) -> np.ndarray:
  """Body-to-world rotation matrix for a scalar-first quaternion."""

  w, x, y, z = normalize_quaternion(quaternion)
  return np.array(
    [
      [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
      [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
      [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ],
    dtype=float,
  )


def quaternion_slerp(
  start: np.ndarray, end: np.ndarray, fraction: float
) -> np.ndarray:
  """Shortest-path spherical interpolation between two quaternions."""

  first = normalize_quaternion(start)
  second = normalize_quaternion(end)
  dot = float(np.dot(first, second))
  if dot < 0.0:
    second *= -1.0
    dot *= -1.0
  dot = float(np.clip(dot, -1.0, 1.0))
  if dot > 0.9995:
    return normalize_quaternion(first + fraction * (second - first))
  angle = math.acos(dot)
  denominator = math.sin(angle)
  return normalize_quaternion(
    math.sin((1.0 - fraction) * angle) / denominator * first
    + math.sin(fraction * angle) / denominator * second
  )


def gimbal_direction_body(gimbal_radians: np.ndarray) -> np.ndarray:
  """Convert a two-axis angle vector into a body-frame unit thrust vector."""

  gimbal = np.asarray(gimbal_radians, dtype=float)
  angle = float(np.linalg.norm(gimbal))
  if angle < 1e-12:
    return np.array([0.0, 0.0, 1.0], dtype=float)
  lateral = math.sin(angle) * gimbal / angle
  return np.array([lateral[0], lateral[1], math.cos(angle)], dtype=float)


@dataclass(frozen=True)
class MPCConfig:
  """Numerical and physical configuration for the 6-DOF MPC."""

  horizon_steps: int = 8
  prediction_dt: float = 0.35
  successive_iterations: int = 3
  gravity_mps2: tuple[float, float, float] = (0.0, 0.0, -9.81)
  initial_mass_kg: float = 30_000.0
  dry_mass_kg: float = 21_000.0
  initial_inertia_kgm2: tuple[float, float, float] = (
    4_300_000.0,
    4_300_000.0,
    60_000.0,
  )
  mass_model: RocketMassModel | None = None
  engine_position_body_m: tuple[float, float, float] = (0.0, 0.0, -20.10)
  alpha_kg_per_newton_second: float = 5.0e-4
  min_thrust_newtons: float = 144_000.0
  max_thrust_newtons: float = 576_000.0
  max_gimbal_radians: float = math.radians(20.0)
  max_roll_torque_nm: float = 17_500.0
  max_tilt_radians: float = math.radians(30.0)
  max_angular_rate_rps: float = math.radians(35.0)
  minimum_com_height_m: float = 20.66
  state_trust_radius: float = 1.75
  control_trust_radius: float = 1.25
  virtual_control_weight: float = 300_000.0
  trust_region_weight: float = 1.0
  maximum_scaled_defect: float = 0.10
  solver_max_iterations: int = 100
  allow_refinement_recovery: bool = False


@dataclass(frozen=True)
class MPCResult:
  """First command, nonlinear predicted trajectory, and solve diagnostics."""

  success: bool
  control: np.ndarray
  predicted_states: np.ndarray
  status: str
  solve_time_seconds: float
  iterations: int
  scaled_dynamics_defect: float
  scaled_virtual_control: float


class SixDofMPC:
  """Small SOCP-based 6-DOF MPC with successive dynamics linearization."""

  def __init__(self, config: MPCConfig | None = None) -> None:
    self.config = config or MPCConfig()
    self.mass_model = self.config.mass_model or RocketMassModel(
      dry_mass_kg=self.config.dry_mass_kg,
      initial_propellant_mass_kg=(
        self.config.initial_mass_kg - self.config.dry_mass_kg
      ),
      initial_inertia_kgm2=self.config.initial_inertia_kgm2,
    )
    self._state_scale = np.array(
      [
        self.config.initial_mass_kg,
        10.0,
        10.0,
        10.0,
        5.0,
        5.0,
        5.0,
        1.0,
        1.0,
        1.0,
        1.0,
        0.5,
        0.5,
        0.5,
      ],
      dtype=float,
    )
    self._control_scale = np.array(
      [
        self.config.max_thrust_newtons,
        self.config.max_gimbal_radians,
        self.config.max_gimbal_radians,
        self.config.max_roll_torque_nm,
      ],
      dtype=float,
    )
    self._state_difference = np.array(
      [
        1.0,
        0.01,
        0.01,
        0.01,
        0.01,
        0.01,
        0.01,
        1e-4,
        1e-4,
        1e-4,
        1e-4,
        1e-4,
        1e-4,
        1e-4,
      ],
      dtype=float,
    )
    self._control_difference = np.array(
      [100.0, 1e-4, 1e-4, 100.0], dtype=float
    )
    self._nominal_states: np.ndarray | None = None
    self._nominal_controls: np.ndarray | None = None
    self._mass_properties_cache: dict[float, MassProperties] = {}
    self._build_problem()

  def reset(self) -> None:
    self._nominal_states = None
    self._nominal_controls = None

  def mass_properties(self, mass_kg: float) -> MassProperties:
    """Return the fuel-dependent mass properties used by prediction."""

    key = float(mass_kg)
    cached = self._mass_properties_cache.get(key)
    if cached is not None:
      return cached
    properties = self.mass_model.properties(key)
    if len(self._mass_properties_cache) >= 4096:
      self._mass_properties_cache.clear()
    self._mass_properties_cache[key] = properties
    return properties

  def continuous_dynamics(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
    """Evaluate the paper-style 6-DOF rigid-body differential equation."""

    state = np.asarray(state, dtype=float)
    control = np.asarray(control, dtype=float)
    mass = max(float(state[MASS]), self.config.dry_mass_kg)
    quaternion = normalize_quaternion(state[QUATERNION])
    angular_velocity = state[ANGULAR_VELOCITY]
    thrust = float(
      np.clip(
        control[THRUST],
        self.config.min_thrust_newtons,
        self.config.max_thrust_newtons,
      )
    )
    gimbal = np.asarray(control[GIMBAL], dtype=float).copy()
    gimbal_norm = float(np.linalg.norm(gimbal))
    if gimbal_norm > self.config.max_gimbal_radians:
      gimbal *= self.config.max_gimbal_radians / gimbal_norm
    thrust_direction = gimbal_direction_body(gimbal)
    thrust_body = thrust * thrust_direction
    rotation = quaternion_to_matrix(quaternion)
    mass_properties = self.mass_properties(mass)
    inertia = mass_properties.inertia_at_com_kgm2
    angular_momentum = inertia * angular_velocity
    engine_position_relative_com = (
      np.asarray(self.config.engine_position_body_m, dtype=float)
      - mass_properties.center_of_mass_body_m
    )
    engine_moment = np.cross(
      engine_position_relative_com, thrust_body
    )
    roll_torque = float(
      np.clip(
        control[ROLL_TORQUE],
        -self.config.max_roll_torque_nm,
        self.config.max_roll_torque_nm,
      )
    )
    total_torque = engine_moment + np.array([0.0, 0.0, roll_torque])

    derivative = np.zeros(NX, dtype=float)
    derivative[MASS] = -self.config.alpha_kg_per_newton_second * thrust
    derivative[POSITION] = state[VELOCITY]
    derivative[VELOCITY] = (
      np.asarray(self.config.gravity_mps2, dtype=float)
      + rotation @ thrust_body / mass
    )
    derivative[QUATERNION] = 0.5 * quaternion_multiply(
      quaternion, np.concatenate(([0.0], angular_velocity))
    )
    derivative[ANGULAR_VELOCITY] = (
      total_torque - np.cross(angular_velocity, angular_momentum)
    ) / inertia
    return derivative

  def discrete_dynamics(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
    """RK4 propagation over one prediction interval."""

    dt = self.config.prediction_dt
    initial = np.asarray(state, dtype=float)
    command = np.asarray(control, dtype=float)
    k1 = self.continuous_dynamics(initial, command)
    k2 = self.continuous_dynamics(initial + 0.5 * dt * k1, command)
    k3 = self.continuous_dynamics(initial + 0.5 * dt * k2, command)
    k4 = self.continuous_dynamics(initial + dt * k3, command)
    result = initial + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    result[QUATERNION] = normalize_quaternion(result[QUATERNION])
    result[MASS] = max(result[MASS], self.config.dry_mass_kg)
    return result

  def _linearize(
    self,
    state: np.ndarray,
    control: np.ndarray,
    *,
    central_differences: bool,
  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base = self.discrete_dynamics(state, control)
    state_matrix = np.zeros((NX, NX), dtype=float)
    control_matrix = np.zeros((NX, NU), dtype=float)
    for index, epsilon in enumerate(self._state_difference):
      plus = np.asarray(state, dtype=float).copy()
      plus[index] += epsilon
      if central_differences:
        minus = np.asarray(state, dtype=float).copy()
        minus[index] -= epsilon
        state_matrix[:, index] = (
          self.discrete_dynamics(plus, control)
          - self.discrete_dynamics(minus, control)
        ) / (2.0 * epsilon)
      else:
        state_matrix[:, index] = (
          self.discrete_dynamics(plus, control)
          - base
        ) / epsilon
    for index, epsilon in enumerate(self._control_difference):
      plus = np.asarray(control, dtype=float).copy()
      plus[index] += epsilon
      if central_differences:
        minus = np.asarray(control, dtype=float).copy()
        minus[index] -= epsilon
        control_matrix[:, index] = (
          self.discrete_dynamics(state, plus)
          - self.discrete_dynamics(state, minus)
        ) / (2.0 * epsilon)
      else:
        control_matrix[:, index] = (
          self.discrete_dynamics(state, plus)
          - base
        ) / epsilon
    offset = base - state_matrix @ state - control_matrix @ control
    return state_matrix, control_matrix, offset

  def _initial_guess(
    self, initial_state: np.ndarray, target_states: np.ndarray
  ) -> tuple[np.ndarray, np.ndarray]:
    steps = self.config.horizon_steps
    states = np.zeros((NX, steps + 1), dtype=float)
    controls = np.zeros((NU, steps), dtype=float)
    hover_thrust = float(
      np.clip(
        initial_state[MASS] * abs(self.config.gravity_mps2[2]),
        self.config.min_thrust_newtons,
        self.config.max_thrust_newtons,
      )
    )
    for index in range(steps + 1):
      fraction = index / steps
      states[:, index] = (
        (1.0 - fraction) * initial_state
        + fraction * target_states[:, index]
      )
      states[MASS, index] = max(
        initial_state[MASS]
        - self.config.alpha_kg_per_newton_second
        * hover_thrust
        * self.config.prediction_dt
        * index,
        self.config.dry_mass_kg,
      )
      states[QUATERNION, index] = quaternion_slerp(
        initial_state[QUATERNION], target_states[QUATERNION, index], fraction
      )
    controls[THRUST, :] = hover_thrust
    return states, controls

  def _shift_warm_start(
    self, initial_state: np.ndarray, target_states: np.ndarray
  ) -> tuple[np.ndarray, np.ndarray]:
    if self._nominal_states is None or self._nominal_controls is None:
      return self._initial_guess(initial_state, target_states)
    controls = np.column_stack(
      (self._nominal_controls[:, 1:], self._nominal_controls[:, -1])
    )
    states = self.rollout(initial_state, controls)
    return states, controls

  def _reference_trajectory(self, target_state: np.ndarray) -> np.ndarray:
    """Advance a moving target consistently across the prediction horizon."""

    target = np.asarray(target_state, dtype=float)
    references = np.repeat(
      target[:, None], self.config.horizon_steps + 1, axis=1
    )
    for index in range(self.config.horizon_steps + 1):
      elapsed = index * self.config.prediction_dt
      references[POSITION, index] = (
        target[POSITION] + elapsed * target[VELOCITY]
      )
      if references[POSITION.stop - 1, index] < self.config.minimum_com_height_m:
        references[POSITION.stop - 1, index] = self.config.minimum_com_height_m
        references[VELOCITY.stop - 1, index] = 0.0
    return references

  def rollout(self, initial_state: np.ndarray, controls: np.ndarray) -> np.ndarray:
    states = np.zeros((NX, controls.shape[1] + 1), dtype=float)
    states[:, 0] = np.asarray(initial_state, dtype=float)
    states[QUATERNION, 0] = normalize_quaternion(states[QUATERNION, 0])
    for index in range(controls.shape[1]):
      states[:, index + 1] = self.discrete_dynamics(
        states[:, index], controls[:, index]
      )
    return states

  def _build_problem(self) -> None:
    steps = self.config.horizon_steps
    self._x = cp.Variable((NX, steps + 1))
    self._u = cp.Variable((NU, steps))
    self._virtual = cp.Variable((NX, steps))
    self._initial = cp.Parameter(NX)
    self._target = cp.Parameter((NX, steps + 1))
    self._previous_control = cp.Parameter(NU)
    self._gimbal_limit = cp.Parameter(nonneg=True)
    self._nominal_x_parameter = cp.Parameter((NX, steps + 1))
    self._nominal_u_parameter = cp.Parameter((NU, steps))
    self._a_parameters = [cp.Parameter((NX, NX)) for _ in range(steps)]
    self._b_parameters = [cp.Parameter((NX, NU)) for _ in range(steps)]
    self._c_parameters = [cp.Parameter(NX) for _ in range(steps)]

    inverse_state_scale = (1.0 / self._state_scale)[:, None]
    inverse_control_scale = (1.0 / self._control_scale)[:, None]
    running_weights = np.array(
      [0.0, 1.5, 1.5, 2.0, 0.8, 0.8, 1.2, 0.5, 5.0, 5.0, 5.0, 2.0, 2.0, 2.0]
    )
    terminal_weights = np.array(
      [
        0.0,
        60.0,
        60.0,
        100.0,
        30.0,
        30.0,
        50.0,
        3.0,
        25.0,
        25.0,
        25.0,
        10.0,
        10.0,
        10.0,
      ]
    )
    constraints: list[cp.Constraint] = [self._x[:, 0] == self._initial]
    objective: cp.Expression = 0.0

    scaled_state_delta = cp.multiply(
      inverse_state_scale, self._x - self._nominal_x_parameter
    )
    scaled_control_delta = cp.multiply(
      inverse_control_scale, self._u - self._nominal_u_parameter
    )
    constraints.extend(
      [
        cp.abs(scaled_state_delta) <= self.config.state_trust_radius,
        cp.abs(scaled_control_delta) <= self.config.control_trust_radius,
      ]
    )
    objective += self.config.trust_region_weight * (
      cp.sum_squares(scaled_state_delta) + cp.sum_squares(scaled_control_delta)
    )

    for index in range(steps):
      constraints.append(
        self._x[:, index + 1]
        == self._a_parameters[index] @ self._x[:, index]
        + self._b_parameters[index] @ self._u[:, index]
        + self._c_parameters[index]
        + self._virtual[:, index]
      )
      constraints.extend(
        [
          self._u[THRUST, index] >= self.config.min_thrust_newtons,
          self._u[THRUST, index] <= self.config.max_thrust_newtons,
          cp.norm(self._u[GIMBAL, index], 2)
          <= self._gimbal_limit,
          cp.abs(self._u[ROLL_TORQUE, index])
          <= self.config.max_roll_torque_nm,
        ]
      )
      scaled_error = cp.multiply(
        running_weights / self._state_scale,
        self._x[:, index] - self._target[:, index],
      )
      objective += cp.sum_squares(scaled_error)
      objective += 0.00001 * self._u[THRUST, index] / self.config.max_thrust_newtons
      objective += 0.05 * cp.sum_squares(
        cp.multiply(
          np.array(
            [
              0.0,
              1.0 / self.config.max_gimbal_radians,
              1.0 / self.config.max_gimbal_radians,
              1.0 / self.config.max_roll_torque_nm,
            ]
          ),
          self._u[:, index],
        )
      )
      previous = (
        self._previous_control if index == 0 else self._u[:, index - 1]
      )
      objective += 0.35 * cp.sum_squares(
        cp.multiply(1.0 / self._control_scale, self._u[:, index] - previous)
      )
      objective += self.config.virtual_control_weight * cp.norm1(
        cp.multiply(1.0 / self._state_scale, self._virtual[:, index])
      )

    for index in range(steps + 1):
      constraints.extend(
        [
          self._x[MASS, index] >= self.config.dry_mass_kg,
          self._x[MASS, index] <= self.config.initial_mass_kg + 1.0,
          self._x[POSITION.stop - 1, index] >= self.config.minimum_com_height_m,
          cp.norm(self._x[ANGULAR_VELOCITY, index], 2)
          <= self.config.max_angular_rate_rps,
          cp.norm(self._x[8:10, index], 2)
          <= math.sin(self.config.max_tilt_radians / 2.0),
          cp.norm(self._x[QUATERNION, index], 2) <= 1.02,
          self._nominal_x_parameter[QUATERNION, index]
          @ self._x[QUATERNION, index]
          >= 0.96,
        ]
      )

    terminal_error = cp.multiply(
      terminal_weights / self._state_scale,
      self._x[:, steps] - self._target[:, steps],
    )
    objective += cp.sum_squares(terminal_error)
    self._problem = cp.Problem(cp.Minimize(objective), constraints)

  def solve(
    self,
    initial_state: np.ndarray,
    target_state: np.ndarray,
    previous_control: np.ndarray | None = None,
    max_gimbal_radians: float | None = None,
    central_differences: bool = True,
  ) -> MPCResult:
    """Solve successive convex subproblems and return the first command."""

    start_time = time.perf_counter()
    initial = np.asarray(initial_state, dtype=float).copy()
    target = np.asarray(target_state, dtype=float).copy()
    initial[QUATERNION] = normalize_quaternion(initial[QUATERNION])
    target[QUATERNION] = normalize_quaternion(target[QUATERNION])
    target_states = self._reference_trajectory(target)
    gimbal_limit = min(
      self.config.max_gimbal_radians,
      (
        self.config.max_gimbal_radians
        if max_gimbal_radians is None
        else max(float(max_gimbal_radians), 0.0)
      ),
    )
    if previous_control is None:
      previous = np.array(
        [
          np.clip(
            initial[MASS] * abs(self.config.gravity_mps2[2]),
            self.config.min_thrust_newtons,
            self.config.max_thrust_newtons,
          ),
          0.0,
          0.0,
          0.0,
        ],
        dtype=float,
      )
    else:
      previous = np.asarray(previous_control, dtype=float).copy()
    previous_gimbal_norm = float(np.linalg.norm(previous[GIMBAL]))
    if previous_gimbal_norm > gimbal_limit > 0.0:
      previous[GIMBAL] *= gimbal_limit / previous_gimbal_norm
    nominal_states, nominal_controls = self._shift_warm_start(
      initial, target_states
    )
    for node in range(self.config.horizon_steps):
      nominal_gimbal_norm = float(np.linalg.norm(nominal_controls[GIMBAL, node]))
      if nominal_gimbal_norm > gimbal_limit > 0.0:
        nominal_controls[GIMBAL, node] *= gimbal_limit / nominal_gimbal_norm
    nominal_states = self.rollout(initial, nominal_controls)
    status = "not_solved"
    solution_states = nominal_states
    solution_controls = nominal_controls
    virtual_scaled = math.inf
    iterations_completed = 0
    refinement_failure_status: str | None = None

    try:
      for iteration in range(self.config.successive_iterations):
        for node in range(self.config.horizon_steps):
          matrix_a, matrix_b, offset = self._linearize(
            nominal_states[:, node],
            nominal_controls[:, node],
            central_differences=central_differences,
          )
          self._a_parameters[node].value = matrix_a
          self._b_parameters[node].value = matrix_b
          self._c_parameters[node].value = offset
        self._initial.value = initial
        self._target.value = target_states
        self._previous_control.value = previous
        self._gimbal_limit.value = gimbal_limit
        self._nominal_x_parameter.value = nominal_states
        self._nominal_u_parameter.value = nominal_controls
        self._x.value = nominal_states
        self._u.value = nominal_controls
        self._virtual.value = np.zeros(
          (NX, self.config.horizon_steps), dtype=float
        )
        with warnings.catch_warnings():
          warnings.filterwarnings(
            "ignore",
            message="Solution may be inaccurate.*",
            category=UserWarning,
          )
          self._problem.solve(
            solver=cp.CLARABEL,
            warm_start=True,
            verbose=False,
            tol_gap_abs=1e-5,
            tol_feas=1e-5,
            max_iter=self.config.solver_max_iterations,
          )
        status = str(self._problem.status)
        if status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
          if (
            self.config.allow_refinement_recovery
            and iterations_completed > 0
          ):
            refinement_failure_status = status
          break
        if self._x.value is None or self._u.value is None:
          status = "empty_solution"
          break
        solution_states = np.asarray(self._x.value, dtype=float)
        solution_controls = np.asarray(self._u.value, dtype=float)
        for node in range(self.config.horizon_steps + 1):
          solution_states[QUATERNION, node] = normalize_quaternion(
            solution_states[QUATERNION, node]
          )
        if self._virtual.value is not None:
          virtual_scaled = float(
            np.max(
              np.abs(
                np.asarray(self._virtual.value, dtype=float)
                / self._state_scale[:, None]
              )
            )
          )
        nominal_states = solution_states
        nominal_controls = solution_controls
        iterations_completed = iteration + 1
    except (cp.error.SolverError, ValueError, FloatingPointError) as error:
      failure_status = f"solver_error:{type(error).__name__}"
      if (
        self.config.allow_refinement_recovery
        and iterations_completed > 0
      ):
        refinement_failure_status = failure_status
      else:
        status = failure_status

    if refinement_failure_status is not None:
      status = f"recovered_after_{refinement_failure_status}"

    nonlinear_states = self.rollout(initial, solution_controls)
    scaled_defect = float(
      np.max(
        np.abs((nonlinear_states - solution_states) / self._state_scale[:, None])
      )
    )
    finite = bool(
      np.all(np.isfinite(solution_controls))
      and np.all(np.isfinite(solution_states))
    )
    success = (
      (
        status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
        or refinement_failure_status is not None
      )
      and finite
      and scaled_defect <= self.config.maximum_scaled_defect
    )
    if success:
      self._nominal_states = solution_states.copy()
      self._nominal_controls = solution_controls.copy()
      control = solution_controls[:, 0].copy()
    else:
      control = previous.copy()
    return MPCResult(
      success=success,
      control=control,
      predicted_states=nonlinear_states,
      status=status,
      solve_time_seconds=time.perf_counter() - start_time,
      iterations=iterations_completed,
      scaled_dynamics_defect=scaled_defect,
      scaled_virtual_control=virtual_scaled,
    )
