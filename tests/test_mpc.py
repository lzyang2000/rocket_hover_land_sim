import math

import numpy as np

from rocket_landing.mpc import (
  ANGULAR_VELOCITY,
  GIMBAL,
  MPCConfig,
  POSITION,
  QUATERNION,
  ROLL_TORQUE,
  SixDofMPC,
  THRUST,
  VELOCITY,
)


def test_nonlinear_model_couples_engine_gimbal_to_attitude() -> None:
  controller = SixDofMPC(MPCConfig(horizon_steps=4, successive_iterations=1))
  state = np.array(
    [30_000.0, 0.0, 0.0, 30.76, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  )
  vertical_control = np.array([300_000.0, 0.0, 0.0, 0.0])
  gimbaled_control = np.array([300_000.0, math.radians(5.0), 0.0, 0.0])

  vertical_derivative = controller.continuous_dynamics(state, vertical_control)
  gimbaled_derivative = controller.continuous_dynamics(state, gimbaled_control)

  assert np.linalg.norm(vertical_derivative[ANGULAR_VELOCITY]) < 1e-12
  assert gimbaled_derivative[ANGULAR_VELOCITY][1] < 0.0
  assert gimbaled_derivative[VELOCITY][0] > 0.0


def test_scvx_mpc_returns_bounded_control_and_reduces_motion() -> None:
  config = MPCConfig(horizon_steps=6, successive_iterations=2)
  controller = SixDofMPC(config)
  yaw = math.radians(8.0)
  state = np.array(
    [
      30_000.0,
      0.0,
      0.0,
      30.76,
      1.0,
      -0.5,
      0.2,
      math.cos(yaw / 2.0),
      0.0,
      0.0,
      math.sin(yaw / 2.0),
      0.04,
      -0.03,
      0.08,
    ]
  )
  target = state.copy()
  target[VELOCITY] = 0.0
  target[QUATERNION] = (1.0, 0.0, 0.0, 0.0)
  target[ANGULAR_VELOCITY] = 0.0

  result = controller.solve(state, target)

  assert result.success, result.status
  assert config.min_thrust_newtons <= result.control[THRUST] <= config.max_thrust_newtons
  assert np.linalg.norm(result.control[GIMBAL]) <= config.max_gimbal_radians + 1e-7
  assert abs(result.control[ROLL_TORQUE]) <= config.max_roll_torque_nm + 1e-7
  assert np.linalg.norm(result.predicted_states[VELOCITY, -1]) < np.linalg.norm(
    state[VELOCITY]
  )
  # Pitch/yaw rates may temporarily rise while the vehicle tilts to arrest
  # translation, but axial spin must still be damped by the roll actuator.
  assert abs(result.predicted_states[ANGULAR_VELOCITY.stop - 1, -1]) < abs(
    state[ANGULAR_VELOCITY.stop - 1]
  )
  assert result.scaled_dynamics_defect < 0.05


def test_mpc_prediction_respects_ground_and_tilt_constraints() -> None:
  config = MPCConfig(horizon_steps=5, successive_iterations=2)
  controller = SixDofMPC(config)
  state = np.array(
    [30_000.0, 0.0, 0.0, 25.0, 0.0, 0.0, -0.5, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
  )
  target = state.copy()
  target[POSITION] = (0.0, 0.0, config.minimum_com_height_m)
  target[VELOCITY] = 0.0

  result = controller.solve(state, target)

  assert result.success, result.status
  assert np.min(result.predicted_states[POSITION.stop - 1]) >= (
    config.minimum_com_height_m - 1e-5
  )
  max_quaternion_lateral = math.sin(config.max_tilt_radians / 2.0)
  assert np.max(np.linalg.norm(result.predicted_states[8:10], axis=0)) <= (
    max_quaternion_lateral + 1e-5
  )
