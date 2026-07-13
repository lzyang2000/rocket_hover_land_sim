import mujoco
import numpy as np
import pytest

from rocket_landing.sim import ROCKET_LANDED_COM_Z_M, RocketSimulation


def test_hover_feedforward_matches_weight_on_earth() -> None:
  simulation = RocketSimulation()
  simulation.data.qpos[0:3] = (0.0, 0.0, ROCKET_LANDED_COM_Z_M + 10.0)
  simulation.data.qvel[:] = 0.0
  mujoco.mj_forward(simulation.model, simulation.data)

  assert simulation.enable_hover()
  expected_throttle = (
    simulation.controller.wet_mass_kg * 9.81
    / simulation.controller.limits.nominal_max_newtons
  )
  assert simulation.controller.throttle == pytest.approx(expected_throttle)


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

  assert np.linalg.norm(simulation.data.qpos[0:3] - target) < 0.05
  assert np.linalg.norm(simulation.data.qvel[0:3]) < 0.02
  assert (
    simulation.controller.pointing_angle_deg()
    <= simulation.controller.limits.pointing_half_angle_deg
  )
