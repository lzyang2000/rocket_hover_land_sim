import math

import mujoco
import numpy as np
import pytest

from rocket_landing.controller import EngineState
from rocket_landing.sim import (
  AUTO_LAND_FUEL_CHECK_PERIOD_S,
  AUTO_LAND_FUEL_MARGIN,
  LANDING_LEG_DEPLOYMENT_TIME_S,
  LANDING_STAGING_HEIGHT_M,
  MPC_TERMINAL_HANDOFF_HEIGHT_M,
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
  fully_deployed_before_cutoff = False
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
    if (
      simulation.landing_leg_deployment >= 1.0 - 1e-9
      and simulation.controller.engine_state is EngineState.LIT
    ):
      fully_deployed_before_cutoff = True
    if simulation.landing_phase is LandingPhase.COMPLETE:
      break

  assert LandingPhase.ALIGN in phases_seen
  assert LandingPhase.DESCEND in phases_seen
  assert descent_start_position is not None
  assert np.linalg.norm(descent_start_position[0:2]) < 2.0
  assert abs(descent_start_position[2] - simulation.landing_staging_altitude) < 2.0
  assert simulation.landing_phase is LandingPhase.COMPLETE
  assert simulation.controller.engine_state is EngineState.SHUTDOWN
  assert fully_deployed_before_cutoff
  assert simulation.landing_leg_deployment == pytest.approx(1.0)

  for _ in range(1000):
    simulation.step()

  assert np.linalg.norm(simulation.data.qpos[0:2]) < 0.12
  assert abs(float(simulation.data.qpos[2]) - ROCKET_LANDED_COM_Z_M) < 0.03
  assert np.linalg.norm(simulation.data.qvel[0:3]) < 0.05
  assert simulation.controller.fuel_mass_kg > 0.0


def test_auto_land_after_manual_takeoff_captures_the_braking_apex() -> None:
  simulation = RocketSimulation()
  simulation.fuel_takeover_triggered = True
  simulation.controller.ignite()
  simulation.controller.throttle = 0.80

  for _ in range(round(3.0 / simulation.model.opt.timestep)):
    simulation.step()

  click_altitude = float(simulation.center_of_mass_position_world()[2])
  click_vertical_speed = float(simulation.center_of_mass_velocity_world()[2])
  assert click_altitude - ROCKET_LANDED_COM_Z_M > 25.0
  assert click_vertical_speed > 20.0

  assert simulation.start_landing()
  braking_acceleration = (
    abs(float(simulation.model.opt.gravity[2]))
    - simulation.controller.limits.min_thrust_newtons
    / simulation.controller.wet_mass_kg
  )
  expected_staging_altitude = click_altitude + (
    click_vertical_speed * click_vertical_speed
    / (2.0 * braking_acceleration)
  )
  assert simulation.landing_staging_altitude == pytest.approx(
    expected_staging_altitude
  )

  align_duration = None
  for step in range(8_000):
    simulation.step()
    if (
      align_duration is None
      and simulation.landing_phase is LandingPhase.DESCEND
    ):
      align_duration = step * simulation.model.opt.timestep
    if simulation.landing_phase is LandingPhase.COMPLETE:
      break

  assert align_duration is not None
  assert align_duration < 10.0
  assert simulation.landing_phase is LandingPhase.COMPLETE


def test_landing_legs_stay_stowed_until_terminal_descent_then_latch() -> None:
  simulation = RocketSimulation()
  foot_id = simulation.landing_foot_geom_ids["xp"]

  assert simulation.landing_leg_deployment == 0.0
  assert not simulation.landing_legs_deploy_commanded
  assert simulation.model.geom_pos[foot_id] == pytest.approx(
    np.array([2.05, 0.0, -3.10])
  )
  assert simulation.launch_mount_enabled
  assert simulation.model.geom_contype[simulation.launch_mount_geom_id] > 0

  simulation.controller.ignite()
  simulation.hover_enabled = True
  simulation.landing_phase = LandingPhase.ALIGN
  simulation._update_landing_legs()
  assert simulation.landing_leg_deployment == 0.0
  assert not simulation.launch_mount_enabled

  simulation.landing_phase = LandingPhase.DESCEND
  simulation.data.qpos[2] = (
    ROCKET_LANDED_COM_Z_M + MPC_TERMINAL_HANDOFF_HEIGHT_M + 0.01
  )
  mujoco.mj_forward(simulation.model, simulation.data)
  simulation._update_landing_legs()
  assert simulation.landing_leg_deployment == 0.0
  assert not simulation.landing_legs_deploy_commanded

  simulation.data.qpos[2] = (
    ROCKET_LANDED_COM_Z_M + MPC_TERMINAL_HANDOFF_HEIGHT_M
  )
  mujoco.mj_forward(simulation.model, simulation.data)
  half_deployment_steps = round(
    LANDING_LEG_DEPLOYMENT_TIME_S / simulation.model.opt.timestep / 2.0
  )
  for _ in range(half_deployment_steps):
    simulation._update_landing_legs()

  assert simulation.landing_leg_deployment == pytest.approx(0.5)
  assert simulation.landing_legs_deploy_commanded
  assert simulation.model.geom_pos[foot_id] == pytest.approx(
    np.array([5.275, 0.0, -11.74])
  )

  simulation.cancel_landing()
  for _ in range(half_deployment_steps):
    simulation._update_landing_legs()

  assert simulation.landing_phase is LandingPhase.INACTIVE
  assert simulation.landing_leg_deployment == pytest.approx(1.0)
  assert simulation.model.geom_pos[foot_id] == pytest.approx(
    np.array([8.50, 0.0, -20.38])
  )

  simulation.reset()
  assert simulation.landing_leg_deployment == 0.0
  assert not simulation.landing_legs_deploy_commanded
  assert simulation.launch_mount_enabled
  assert simulation.model.geom_pos[foot_id] == pytest.approx(
    np.array([2.05, 0.0, -3.10])
  )


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


def test_high_altitude_auto_land_coasts_without_fuel_then_relights() -> None:
  simulation = RocketSimulation()
  simulation.data.qpos[2] = ROCKET_LANDED_COM_Z_M + 40.0
  simulation.data.qvel[:] = 0.0
  mujoco.mj_forward(simulation.model, simulation.data)

  assert simulation.start_landing()
  phases_seen = set()
  coast_fuel = None
  relight_height = None
  for _ in range(7_000):
    simulation.step()
    phases_seen.add(simulation.landing_phase)
    if simulation.landing_phase is LandingPhase.COAST:
      assert simulation.controller.engine_state is EngineState.COAST
      if coast_fuel is None:
        coast_fuel = simulation.controller.fuel_mass_kg
      assert simulation.controller.fuel_mass_kg == pytest.approx(coast_fuel)
      assert simulation.controller.thrust_magnitude_newtons() == 0.0
    elif (
      coast_fuel is not None
      and relight_height is None
      and simulation.landing_phase is LandingPhase.DESCEND
    ):
      relight_height = float(
        simulation.data.qpos[2] - ROCKET_LANDED_COM_Z_M
      )
      assert simulation.controller.engine_state is EngineState.LIT
    if simulation.landing_phase is LandingPhase.COMPLETE:
      break

  assert LandingPhase.COAST in phases_seen
  assert relight_height is not None
  assert 20.0 < relight_height < 35.0
  assert simulation.landing_phase is LandingPhase.COMPLETE
  assert simulation.controller.engine_state is EngineState.SHUTDOWN


def test_1000m_ballistic_coast_uses_energy_corridor_and_lands() -> None:
  simulation = RocketSimulation()
  simulation.data.qpos[2] = ROCKET_LANDED_COM_Z_M + 1_000.0
  simulation.data.qvel[:] = 0.0
  simulation.controller.fuel_mass_kg = 4_000.0
  simulation._update_model_mass(force=True)
  mujoco.mj_forward(simulation.model, simulation.data)

  takeover_threshold = simulation.fuel_takeover_threshold_kg()
  assert 4_000.0 < takeover_threshold < 4_100.0
  simulation.controller.ignite()
  simulation._check_fuel_reserve_takeover()
  assert simulation.fuel_takeover_active
  assert simulation.landing_phase is LandingPhase.COAST
  peak_descent_speed = 0.0
  relight_height = None
  for _ in range(12_000):
    previous_phase = simulation.landing_phase
    simulation.step()
    peak_descent_speed = max(
      peak_descent_speed,
      -float(simulation.data.qvel[2]),
    )
    if (
      previous_phase is LandingPhase.COAST
      and simulation.landing_phase is LandingPhase.DESCEND
    ):
      relight_height = float(
        simulation.data.qpos[2] - ROCKET_LANDED_COM_Z_M
      )
      assert simulation._descent_rate_mps() > 60.0
    if simulation.landing_phase is LandingPhase.COMPLETE:
      break

  assert relight_height is not None
  assert 450.0 < relight_height < 650.0
  assert peak_descent_speed > 80.0
  assert simulation.landing_phase is LandingPhase.COMPLETE
  assert 0.0 < simulation.controller.fuel_mass_kg < 100.0


def test_full_throttle_launch_fuel_auto_lands_below_100kg_reserve() -> None:
  simulation = RocketSimulation()
  simulation.controller.ignite()
  simulation.controller.throttle = simulation.controller.limits.max_throttle

  takeover_fuel = None
  takeover_height = None
  takeover_vertical_speed = None
  for _ in range(30_000):
    previous_phase = simulation.landing_phase
    simulation.step()
    if (
      previous_phase is LandingPhase.INACTIVE
      and simulation.landing_phase is LandingPhase.COAST
    ):
      takeover_fuel = simulation.controller.fuel_mass_kg
      takeover_height = float(
        simulation.data.qpos[2] - ROCKET_LANDED_COM_Z_M
      )
      takeover_vertical_speed = float(simulation.data.qvel[2])
    if simulation.landing_phase in (
      LandingPhase.COMPLETE,
      LandingPhase.ABORTED,
    ):
      break

  assert takeover_fuel is not None
  assert 5_000.0 < takeover_fuel < 5_300.0
  assert takeover_height is not None and takeover_height > 800.0
  assert takeover_vertical_speed is not None and takeover_vertical_speed > 130.0
  assert simulation.landing_phase is LandingPhase.COMPLETE
  assert simulation.controller.engine_state is EngineState.SHUTDOWN
  assert 0.0 < simulation.controller.fuel_mass_kg < 100.0


def test_low_altitude_landing_skips_ballistic_coast() -> None:
  simulation = RocketSimulation()
  simulation.data.qpos[2] = ROCKET_LANDED_COM_Z_M + 15.0
  simulation.data.qvel[:] = 0.0
  mujoco.mj_forward(simulation.model, simulation.data)

  assert simulation.start_landing(fuel_takeover=True)
  phases_seen = set()
  for _ in range(7_000):
    simulation.step()
    phases_seen.add(simulation.landing_phase)
    if simulation.landing_phase is LandingPhase.COMPLETE:
      break

  assert LandingPhase.COAST not in phases_seen
  assert simulation.landing_phase is LandingPhase.COMPLETE


def test_unsafe_coast_attitude_triggers_early_relight() -> None:
  simulation = RocketSimulation()
  simulation.data.qpos[2] = ROCKET_LANDED_COM_Z_M + 60.0
  angle = math.radians(6.0)
  simulation.data.qpos[3:7] = (
    math.cos(angle / 2.0),
    math.sin(angle / 2.0),
    0.0,
    0.0,
  )
  mujoco.mj_forward(simulation.model, simulation.data)
  simulation.controller.ignite()
  assert simulation.controller.begin_coast()
  simulation.hover_enabled = True
  simulation.landing_phase = LandingPhase.COAST

  simulation._update_landing_guidance()

  assert simulation.landing_phase is LandingPhase.DESCEND
  assert simulation.controller.engine_state is EngineState.LIT
  assert simulation.controller.throttle == simulation.controller.limits.max_throttle


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

  assert simulation.landing_phase is LandingPhase.COAST
  assert simulation.controller.engine_state is EngineState.COAST
  assert simulation.fuel_takeover_active
  assert simulation.fuel_takeover_triggered
  assert simulation.landing_staging_altitude == pytest.approx(takeover_altitude)

  simulation.cancel_landing()
  simulation.data.time += AUTO_LAND_FUEL_CHECK_PERIOD_S
  simulation._check_fuel_reserve_takeover()
  assert simulation.landing_phase is LandingPhase.INACTIVE
  assert not simulation.fuel_takeover_active


def test_real_fuel_reserve_takeover_keeps_enough_fuel_to_land() -> None:
  simulation = RocketSimulation()
  simulation.controller.ignite()
  simulation.data.qpos[0:3] = (
    4.0,
    -3.0,
    ROCKET_LANDED_COM_Z_M + 15.0,
  )
  simulation.data.qvel[0:3] = (1.0, -0.5, 1.5)
  mujoco.mj_forward(simulation.model, simulation.data)

  takeover_threshold = simulation.fuel_takeover_threshold_kg()
  assert 7_400.0 < takeover_threshold < 7_550.0
  simulation.controller.fuel_mass_kg = takeover_threshold
  simulation._update_model_mass(force=True)
  mujoco.mj_forward(simulation.model, simulation.data)

  simulation._check_fuel_reserve_takeover()
  assert simulation.landing_phase is LandingPhase.ALIGN
  assert simulation.fuel_takeover_active

  for _ in range(8_000):
    simulation.step()
    if simulation.landing_phase is LandingPhase.COMPLETE:
      break

  assert simulation.landing_phase is LandingPhase.COMPLETE
  assert simulation.controller.engine_state is EngineState.SHUTDOWN
  assert simulation.controller.fuel_mass_kg > 4_500.0


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
