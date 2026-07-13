import mujoco
import numpy as np
import pytest

from rocket_landing.sim import (
  HOVER_TARGET_HORIZONTAL_LEAD_M,
  HOVER_TARGET_VERTICAL_LEAD_M,
  ROCKET_LANDED_COM_Z_M,
  RocketSimulation,
)


def test_hover_feedforward_matches_weight_with_com_migration_compensation() -> None:
  simulation = RocketSimulation()
  simulation.data.qpos[0:3] = (0.0, 0.0, ROCKET_LANDED_COM_Z_M + 10.0)
  simulation.data.qvel[:] = 0.0
  mujoco.mj_forward(simulation.model, simulation.data)

  assert simulation.enable_hover()
  expected_throttle = (
    simulation.controller.wet_mass_kg * 9.81
    / simulation.controller.limits.nominal_max_newtons
  )
  assert simulation.controller.throttle == pytest.approx(
    expected_throttle, rel=0.01
  )


def test_hover_brakes_velocity_and_returns_to_captured_position() -> None:
  simulation = RocketSimulation()
  simulation.data.qpos[0:3] = (0.0, 0.0, ROCKET_LANDED_COM_Z_M + 10.0)
  simulation.data.qvel[0:3] = (2.0, -1.0, 5.0)
  mujoco.mj_forward(simulation.model, simulation.data)

  assert simulation.enable_hover()
  target = simulation.hover_target_position.copy()
  # Physical gimbal torque must first rotate the full-size stage before it can
  # redirect translation, so recovery is intentionally slower than the former
  # instantaneous center-of-mass force model.
  for _ in range(5000):
    simulation.step()

  assert np.linalg.norm(simulation.center_of_mass_position_world() - target) < 0.07
  assert np.linalg.norm(simulation.center_of_mass_velocity_world()) < 0.02
  assert (
    simulation.controller.pointing_angle_deg()
    <= simulation.controller.limits.pointing_half_angle_deg
  )


def test_hover_target_lead_is_bounded_relative_to_the_rocket() -> None:
  simulation = RocketSimulation()
  simulation.data.qpos[0:3] = (0.0, 0.0, ROCKET_LANDED_COM_Z_M + 10.0)
  mujoco.mj_forward(simulation.model, simulation.data)

  assert simulation.enable_hover()
  simulation.move_hover_target(np.array([100.0, -100.0, 100.0]))
  offset = (
    simulation.hover_target_position - simulation.center_of_mass_position_world()
  )

  assert np.linalg.norm(offset[0:2]) == pytest.approx(
    HOVER_TARGET_HORIZONTAL_LEAD_M
  )
  assert offset[2] == pytest.approx(HOVER_TARGET_VERTICAL_LEAD_M)


def test_mpc_tracks_a_wasd_style_moving_position_target() -> None:
  simulation = RocketSimulation(enable_mpc=True)
  simulation.data.qpos[0:3] = (0.0, 0.0, ROCKET_LANDED_COM_Z_M + 10.0)
  simulation.data.qvel[:] = 0.0
  mujoco.mj_forward(simulation.model, simulation.data)

  assert simulation.enable_hover()
  updates = 0
  fallback_updates = 0
  last_request_time = -1.0
  maximum_target_lead = 0.0
  maximum_tilt_deg = 0.0
  position_after_eight_seconds = 0.0
  for step in range(2400):
    if step < 1600:
      simulation.move_hover_target(
        np.array([2.0 * simulation.model.opt.timestep, 0.0, 0.0])
      )
    simulation.step()
    maximum_target_lead = max(
      maximum_target_lead,
      float(
        np.linalg.norm(
          simulation.hover_target_position[0:2]
          - simulation.center_of_mass_position_world()[0:2]
        )
      ),
    )
    rotation = simulation.data.xmat[simulation.rocket_body_id].reshape(3, 3)
    maximum_tilt_deg = max(
      maximum_tilt_deg,
      float(np.degrees(np.arccos(np.clip(rotation[2, 2], -1.0, 1.0)))),
    )
    if simulation.last_mpc_request_time != last_request_time:
      last_request_time = simulation.last_mpc_request_time
      updates += 1
      fallback_updates += int(simulation.mpc_using_fallback)
    if step == 1599:
      position_after_eight_seconds = float(
        simulation.center_of_mass_position_world()[0]
      )

  assert simulation.hover_target_velocity == pytest.approx(np.zeros(3))
  assert maximum_target_lead <= 2.51
  assert maximum_tilt_deg < 4.0
  assert position_after_eight_seconds > 5.0
  assert simulation.center_of_mass_position_world()[0] == pytest.approx(
    simulation.hover_target_position[0], abs=0.15
  )
  assert abs(float(simulation.center_of_mass_velocity_world()[0])) < 0.10
  assert fallback_updates <= max(1, updates // 10)
