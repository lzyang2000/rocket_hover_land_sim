import math

import mujoco
import numpy as np
import pytest

from rocket_landing.controller import EngineState
from rocket_landing.sim import (
  AUTO_LAND_FUEL_CHECK_PERIOD_S,
  AUTO_LAND_FUEL_MARGIN,
  LANDING_STAGING_HEIGHT_M,
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
  assert simulation.landing_staging_altitude == pytest.approx(
    simulation.landing_center_of_mass_position()[2]
    + LANDING_STAGING_HEIGHT_M
  )
  phases_seen = {simulation.landing_phase}
  descent_start_position = None
  for _ in range(7000):
    previous_phase = simulation.landing_phase
    simulation.step()
    phases_seen.add(simulation.landing_phase)
    if (
      previous_phase is LandingPhase.ALIGN
      and simulation.landing_phase is LandingPhase.DESCEND
    ):
      descent_start_position = simulation.center_of_mass_position_world().copy()
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
  assert descent_start_position is not None
  assert np.linalg.norm(descent_start_position[0:2]) < 2.0
  assert abs(descent_start_position[2] - simulation.landing_staging_altitude) < 2.0
  assert simulation.landing_phase is LandingPhase.COMPLETE
  assert simulation.controller.engine_state is EngineState.SHUTDOWN

  for _ in range(1000):
    simulation.step()

  assert np.linalg.norm(simulation.data.qpos[0:2]) < 0.12
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

  transition = RocketSimulation()
  transition.controller.ignite()
  transition.hover_enabled = True
  transition.landing_phase = LandingPhase.DESCEND
  transition.data.qpos[2] = ROCKET_LANDED_COM_Z_M + 31.0
  mujoco.mj_forward(transition.model, transition.data)
  transition.hover_target_position = (
    transition.center_of_mass_position_world().copy()
  )
  transition_rates = []
  for height in (31.0, 29.99, 17.99, 9.99, 4.99, 2.49, 0.99):
    transition.data.qpos[2] = ROCKET_LANDED_COM_Z_M + height
    mujoco.mj_forward(transition.model, transition.data)
    transition._update_landing_guidance()
    transition_rates.append(-float(transition.hover_target_velocity[2]))

  assert transition_rates == pytest.approx(
    [12.0, 8.0, 5.0, 3.0, 1.5, 0.6, 0.25]
  )
  assert min(transition_rates) > 0.0

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
  assert cutoff_height < 0.151
  assert -0.50 <= float(flight.data.qvel[2]) <= 0.15


def test_fuel_reserve_takeover_triggers_at_105_percent_and_latches() -> None:
  simulation = RocketSimulation()
  simulation.data.qpos[2] = ROCKET_LANDED_COM_Z_M + 40.0
  mujoco.mj_forward(simulation.model, simulation.data)
  simulation.controller.ignite()
  estimated_fuel_kg = 2_000.0
  simulation.estimated_landing_fuel_kg = lambda: estimated_fuel_kg

  simulation.controller.fuel_mass_kg = (
    AUTO_LAND_FUEL_MARGIN * estimated_fuel_kg + 0.01
  )
  simulation._check_fuel_reserve_takeover()
  assert simulation.landing_phase is LandingPhase.INACTIVE

  simulation.data.time += AUTO_LAND_FUEL_CHECK_PERIOD_S
  simulation.controller.fuel_mass_kg = AUTO_LAND_FUEL_MARGIN * estimated_fuel_kg
  takeover_altitude = float(simulation.center_of_mass_position_world()[2])
  simulation._check_fuel_reserve_takeover()

  assert simulation.landing_phase is LandingPhase.ALIGN
  assert simulation.fuel_takeover_active
  assert simulation.fuel_takeover_triggered
  assert simulation.landing_staging_altitude == pytest.approx(takeover_altitude)

  simulation.cancel_landing()
  simulation.data.time += AUTO_LAND_FUEL_CHECK_PERIOD_S
  simulation._check_fuel_reserve_takeover()
  assert simulation.landing_phase is LandingPhase.INACTIVE
  assert not simulation.fuel_takeover_active


def test_full_fuel_does_not_trigger_reserve_takeover() -> None:
  simulation = RocketSimulation()
  simulation.data.qpos[2] = ROCKET_LANDED_COM_Z_M + 40.0
  mujoco.mj_forward(simulation.model, simulation.data)
  simulation.controller.ignite()

  assert (
    simulation.fuel_takeover_threshold_kg()
    < simulation.controller.fuel_mass_kg
  )
  simulation._check_fuel_reserve_takeover()

  assert simulation.landing_phase is LandingPhase.INACTIVE
  assert not simulation.fuel_takeover_triggered


def test_align_phase_accepts_a_coarser_pad_capture() -> None:
  simulation = RocketSimulation()
  simulation.data.qpos[0:3] = (
    1.80,
    0.0,
    ROCKET_LANDED_COM_Z_M + 12.0,
  )
  simulation.data.qvel[0:3] = (0.90, 0.0, 1.40)
  mujoco.mj_forward(simulation.model, simulation.data)
  simulation.controller.ignite()
  simulation.hover_enabled = True
  simulation.landing_phase = LandingPhase.ALIGN
  simulation.landing_staging_altitude = (
    simulation.center_of_mass_position_world()[2]
  )

  simulation._update_landing_guidance()

  assert simulation.landing_phase is LandingPhase.DESCEND


def test_align_phase_waits_outside_descent_capture_corridor() -> None:
  simulation = RocketSimulation()
  simulation.data.qpos[0:3] = (
    2.20,
    0.0,
    ROCKET_LANDED_COM_Z_M + 12.0,
  )
  simulation.data.qvel[0:3] = (0.0, 0.0, 0.0)
  mujoco.mj_forward(simulation.model, simulation.data)
  simulation.controller.ignite()
  simulation.hover_enabled = True
  simulation.landing_phase = LandingPhase.ALIGN
  simulation.landing_staging_altitude = (
    simulation.center_of_mass_position_world()[2]
  )

  simulation._update_landing_guidance()

  assert simulation.landing_phase is LandingPhase.ALIGN


def test_high_altitude_align_uses_wider_lead_but_requires_capture() -> None:
  simulation = RocketSimulation()
  simulation.data.qpos[0:3] = (
    4.25,
    0.0,
    ROCKET_LANDED_COM_Z_M + 35.0,
  )
  simulation.data.qvel[:] = 0.0
  mujoco.mj_forward(simulation.model, simulation.data)

  assert simulation.start_landing()
  simulation._update_landing_guidance()

  assert simulation.landing_horizontal_lead_for_height_m(18.0) == pytest.approx(
    4.0
  )
  assert simulation.landing_horizontal_lead_for_height_m(35.0) == pytest.approx(
    6.55
  )
  assert simulation.landing_horizontal_lead_for_height_m(80.0) == pytest.approx(
    8.0
  )
  assert simulation.landing_phase is LandingPhase.ALIGN


def test_descent_keeps_horizontal_target_inside_bounded_lead() -> None:
  simulation = RocketSimulation()
  simulation.data.qpos[0:3] = (
    10.0,
    0.0,
    ROCKET_LANDED_COM_Z_M + 35.0,
  )
  simulation.data.qvel[:] = 0.0
  mujoco.mj_forward(simulation.model, simulation.data)
  simulation.controller.ignite()
  simulation.hover_enabled = True
  simulation.landing_phase = LandingPhase.DESCEND
  simulation.hover_target_position = simulation.center_of_mass_position_world()
  position = simulation.center_of_mass_position_world()

  simulation._update_landing_guidance()

  target_offset = simulation.hover_target_position[0:2] - position[0:2]
  assert np.linalg.norm(target_offset) == pytest.approx(6.55)
  assert abs(float(simulation.hover_target_position[0])) < abs(float(position[0]))


def test_mpc_offset_alignment_is_bounded_and_rarely_falls_back() -> None:
  simulation = RocketSimulation(enable_mpc=True)
  simulation.data.qpos[0:3] = (
    4.0,
    -3.0,
    ROCKET_LANDED_COM_Z_M + 15.0,
  )
  simulation.data.qvel[0:3] = (1.0, -0.5, 1.5)
  mujoco.mj_forward(simulation.model, simulation.data)

  assert simulation.start_landing()
  updates = 0
  fallback_updates = 0
  last_request_time = -1.0
  maximum_horizontal_speed = 0.0
  maximum_tilt_deg = 0.0
  for _ in range(3000):
    simulation.step()
    maximum_horizontal_speed = max(
      maximum_horizontal_speed,
      float(np.linalg.norm(simulation.center_of_mass_velocity_world()[0:2])),
    )
    rotation = simulation.data.xmat[simulation.rocket_body_id].reshape(3, 3)
    maximum_tilt_deg = max(
      maximum_tilt_deg,
      math.degrees(math.acos(float(np.clip(rotation[2, 2], -1.0, 1.0)))),
    )
    if simulation.last_mpc_request_time != last_request_time:
      last_request_time = simulation.last_mpc_request_time
      updates += 1
      fallback_updates += int(simulation.mpc_using_fallback)
    if simulation.landing_phase is LandingPhase.DESCEND:
      break

  assert simulation.landing_phase is LandingPhase.DESCEND
  assert maximum_horizontal_speed < 2.5
  assert maximum_tilt_deg < 8.0
  assert updates >= 10
  assert fallback_updates <= max(1, updates // 5)

  saw_terminal_controller = False
  for _ in range(4000):
    simulation.step()
    saw_terminal_controller |= simulation.terminal_controller_active
    if simulation.landing_phase is LandingPhase.COMPLETE:
      break

  assert saw_terminal_controller
  assert simulation.landing_phase is LandingPhase.COMPLETE


def test_terminal_controller_flies_final_descent_and_cuts_off() -> None:
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

  saw_terminal_controller = False
  maximum_terminal_gimbal = 0.0
  terminal_gimbal_reversals = 0
  previous_terminal_gimbal = None
  for _ in range(3000):
    simulation.step()
    saw_terminal_controller |= simulation.terminal_controller_active
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
  assert saw_terminal_controller
  assert simulation.controller.engine_state is EngineState.SHUTDOWN
  assert maximum_terminal_gimbal <= np.radians(1.5) + 1e-6
  assert terminal_gimbal_reversals == 0
  assert np.linalg.norm(simulation.data.qpos[0:2]) < 0.50
  assert np.linalg.norm(
    simulation.center_of_mass_velocity_world()[0:2]
  ) < 0.31

  for _ in range(2_000):
    simulation.step()

  assert np.linalg.norm(simulation.data.qpos[0:2]) < 0.50
  assert np.linalg.norm(simulation.data.qvel[0:3]) < 0.05
