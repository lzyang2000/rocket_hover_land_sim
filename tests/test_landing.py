import mujoco
import numpy as np
import pytest

from rocket_landing.controller import EngineState
from rocket_landing.sim import (
  LandingPhase,
  ROCKET_LANDED_COM_Z_M,
  RocketSimulation,
)


def test_auto_land_aligns_descends_and_cuts_off_on_the_pad() -> None:
  simulation = RocketSimulation()
  simulation.data.qpos[0:3] = (
    4.0,
    -3.0,
    ROCKET_LANDED_COM_Z_M + 15.0,
  )
  simulation.data.qvel[0:3] = (1.0, -0.5, 1.5)
  mujoco.mj_forward(simulation.model, simulation.data)

  assert simulation.start_landing()
  phases_seen = {simulation.landing_phase}
  for _ in range(7000):
    simulation.step()
    phases_seen.add(simulation.landing_phase)
    if simulation.controller.engine_state is EngineState.LIT:
      assert (
        simulation.controller.limits.min_thrust_newtons
        <= simulation.controller.thrust_magnitude_newtons()
        <= simulation.controller.limits.max_thrust_newtons
      )
      assert (
        simulation.controller.pointing_angle_deg()
        <= simulation.controller.limits.pointing_half_angle_deg
      )
    if simulation.landing_phase is LandingPhase.COMPLETE:
      break

  assert LandingPhase.ALIGN in phases_seen
  assert LandingPhase.DESCEND in phases_seen
  assert simulation.landing_phase is LandingPhase.COMPLETE
  assert simulation.controller.engine_state is EngineState.SHUTDOWN

  for _ in range(1000):
    simulation.step()

  assert np.linalg.norm(simulation.data.qpos[0:2]) < 0.05
  assert abs(float(simulation.data.qpos[2]) - ROCKET_LANDED_COM_Z_M) < 0.03
  assert np.linalg.norm(simulation.data.qvel[0:3]) < 0.05
  assert simulation.controller.fuel_mass_kg > 0.0


def test_auto_land_uses_aggressive_approach_and_terminal_braking_profile() -> None:
  simulation = RocketSimulation()
  rate_by_height = {
    40.0: 12.0,
    20.0: 8.0,
    15.0: 5.0,
    8.0: 3.0,
    3.0: 1.5,
    1.5: 0.6,
    0.5: 0.25,
  }
  for height, expected_rate in rate_by_height.items():
    simulation.data.qpos[2] = ROCKET_LANDED_COM_Z_M + height
    assert simulation._descent_rate_mps() == expected_rate

  simulation.data.qpos[2] = ROCKET_LANDED_COM_Z_M + 15.0
  simulation.data.qvel[:] = 0.0
  mujoco.mj_forward(simulation.model, simulation.data)
  simulation.controller.ignite()
  simulation.hover_enabled = True
  simulation.landing_phase = LandingPhase.DESCEND
  simulation.hover_target_position = simulation.landing_center_of_mass_position()
  simulation.hover_target_position[2] += 15.0
  simulation._update_landing_guidance()

  assert simulation.hover_target_velocity == pytest.approx(
    np.array([0.0, 0.0, -5.0])
  )

  flight = RocketSimulation()
  flight.data.qpos[2] = ROCKET_LANDED_COM_Z_M + 40.0
  flight.data.qvel[:] = 0.0
  mujoco.mj_forward(flight.model, flight.data)
  assert flight.start_landing()
  peak_descent_speed = 0.0
  for _ in range(4_000):
    flight.step()
    peak_descent_speed = max(
      peak_descent_speed, -float(flight.data.qvel[2])
    )
    if flight.landing_phase is LandingPhase.COMPLETE:
      break

  cutoff_height = float(
    flight.data.qpos[2] - ROCKET_LANDED_COM_Z_M
  )
  assert flight.landing_phase is LandingPhase.COMPLETE
  assert peak_descent_speed > 10.0
  assert cutoff_height > 0.05
  assert -0.35 <= float(flight.data.qvel[2]) <= 0.10


def test_align_phase_accepts_a_coarser_pad_capture() -> None:
  simulation = RocketSimulation()
  simulation.data.qpos[0:3] = (
    1.25,
    0.0,
    ROCKET_LANDED_COM_Z_M + 12.0,
  )
  simulation.data.qvel[0:3] = (0.50, 0.0, 0.50)
  mujoco.mj_forward(simulation.model, simulation.data)
  simulation.controller.ignite()
  simulation.hover_enabled = True
  simulation.landing_phase = LandingPhase.ALIGN
  simulation.landing_staging_altitude = (
    simulation.center_of_mass_position_world()[2]
  )

  simulation._update_landing_guidance()

  assert simulation.landing_phase is LandingPhase.DESCEND


def test_scvx_mpc_flies_terminal_descent_and_cuts_off() -> None:
  simulation = RocketSimulation(enable_mpc=True)
  simulation.data.qpos[0:3] = (
    0.35,
    -0.20,
    ROCKET_LANDED_COM_Z_M + 2.0,
  )
  simulation.data.qvel[0:3] = (0.10, -0.05, -0.10)
  mujoco.mj_forward(simulation.model, simulation.data)
  simulation.controller.ignite()
  simulation.hover_enabled = True
  simulation.landing_phase = LandingPhase.DESCEND
  simulation.hover_target_position = simulation.data.qpos[0:3].copy()

  saw_valid_mpc_command = False
  maximum_terminal_gimbal = 0.0
  terminal_gimbal_reversals = 0
  previous_terminal_gimbal = None
  for _ in range(3000):
    simulation.step()
    saw_valid_mpc_command |= (
      simulation.last_mpc_result is not None
      and simulation.last_mpc_result.success
      and not simulation.mpc_using_fallback
    )
    height = float(simulation.data.qpos[2] - ROCKET_LANDED_COM_Z_M)
    if 0.0 < height <= 2.5:
      gimbal = simulation.engine_gimbal_radians.copy()
      maximum_terminal_gimbal = max(
        maximum_terminal_gimbal, float(np.linalg.norm(gimbal))
      )
      if (
        previous_terminal_gimbal is not None
        and np.linalg.norm(previous_terminal_gimbal) > 1e-3
        and np.linalg.norm(gimbal) > 1e-3
        and np.dot(previous_terminal_gimbal, gimbal) < 0.0
      ):
        terminal_gimbal_reversals += 1
      previous_terminal_gimbal = gimbal
    if simulation.landing_phase is LandingPhase.COMPLETE:
      break

  assert simulation.landing_phase is LandingPhase.COMPLETE
  assert saw_valid_mpc_command
  assert simulation.controller.engine_state is EngineState.SHUTDOWN
  assert maximum_terminal_gimbal <= np.radians(1.5) + 1e-6
  assert terminal_gimbal_reversals == 0
  assert np.linalg.norm(simulation.data.qpos[0:2]) < 0.30
  assert np.linalg.norm(
    simulation.center_of_mass_velocity_world()[0:2]
  ) < 0.21

  for _ in range(1_000):
    simulation.step()

  assert np.linalg.norm(simulation.data.qpos[0:2]) < 0.50
  assert np.linalg.norm(simulation.data.qvel[0:3]) < 0.05
