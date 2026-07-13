import math
import time

import glfw
import mujoco
import numpy as np
import pytest

from rocket_landing.mpc import MPCResult, gimbal_direction_body
from rocket_landing.sim import (
  LandingPhase,
  MAX_ROLL_CONTROL_TORQUE_NM,
  ROCKET_DIAMETER_M,
  ROCKET_HEIGHT_M,
  ROCKET_LANDED_COM_Z_M,
  ROLL_RCS_MAX_THRUSTER_FORCE_N,
  THRUST_ARROW_MAX_LENGTH_M,
  RocketSimulation,
  RocketWindow,
  build_argument_parser,
  model_path,
)


def test_mjcf_compiles_and_contains_free_rocket() -> None:
  model = mujoco.MjModel.from_xml_path(str(model_path()))
  rocket_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "rocket")
  assert rocket_id > 0
  assert model.nq == 7
  assert model.nv == 6


def test_launcher_defaults_to_synchronous_mpc() -> None:
  parser = build_argument_parser()

  assert not parser.parse_args([]).async_mpc
  assert parser.parse_args(["--async-mpc"]).async_mpc


def test_mpc_warmup_does_not_mutate_the_vehicle() -> None:
  simulation = RocketSimulation(enable_mpc=True)
  qpos = simulation.data.qpos.copy()
  qvel = simulation.data.qvel.copy()
  fuel_mass_kg = simulation.controller.fuel_mass_kg

  simulation.warm_up_mpc()

  assert simulation.data.qpos == pytest.approx(qpos)
  assert simulation.data.qvel == pytest.approx(qvel)
  assert simulation.controller.fuel_mass_kg == pytest.approx(fuel_mass_kg)
  assert simulation.last_mpc_result is None
  assert simulation.mpc_using_fallback


def test_vehicle_has_falcon_9_first_stage_proportions() -> None:
  simulation = RocketSimulation()
  fuselage_id = mujoco.mj_name2id(
    simulation.model, mujoco.mjtObj.mjOBJ_GEOM, "fuselage"
  )
  foot_id = mujoco.mj_name2id(
    simulation.model, mujoco.mjtObj.mjOBJ_GEOM, "foot_xp"
  )

  assert 2.0 * simulation.model.geom_size[fuselage_id, 0] == pytest.approx(
    ROCKET_DIAMETER_M
  )
  assert ROCKET_HEIGHT_M / ROCKET_DIAMETER_M == pytest.approx(11.26, rel=0.01)
  assert simulation.data.qpos[2] == pytest.approx(ROCKET_LANDED_COM_Z_M)
  assert simulation.model.geom_pos[foot_id, 0] == pytest.approx(8.50)


def test_headless_rollout_lifts_with_vertical_thrust() -> None:
  simulation = RocketSimulation()
  initial_altitude = float(simulation.data.qpos[2])
  simulation.controller.ignite()
  simulation.controller.throttle = 0.60
  altitude_changes = []
  previous_altitude = initial_altitude
  for _ in range(800):
    simulation.step()
    altitude = float(simulation.data.qpos[2])
    altitude_changes.append(altitude - previous_altitude)
    previous_altitude = altitude
  assert simulation.data.qpos[2] > initial_altitude + 20.0
  assert min(altitude_changes) > -0.05


def test_thrust_arrow_tracks_engine_magnitude_and_gimbal_direction() -> None:
  simulation = RocketSimulation()
  assert simulation.thrust_arrow_world() is None

  simulation.controller.ignite()
  minimum_arrow = simulation.thrust_arrow_world()
  assert minimum_arrow is not None
  minimum_origin, minimum_tip, minimum_fraction = minimum_arrow
  minimum_vector = minimum_tip - minimum_origin
  assert np.linalg.norm(minimum_vector) == pytest.approx(
    THRUST_ARROW_MAX_LENGTH_M * minimum_fraction
  )
  assert minimum_vector / np.linalg.norm(minimum_vector) == pytest.approx(
    np.array([0.0, 0.0, -1.0])
  )

  simulation.controller.throttle = simulation.controller.limits.max_throttle
  simulation.engine_gimbal_radians[:] = (math.radians(6.0), math.radians(-3.0))
  maximum_arrow = simulation.thrust_arrow_world()
  assert maximum_arrow is not None
  maximum_origin, maximum_tip, maximum_fraction = maximum_arrow
  maximum_vector = maximum_tip - maximum_origin
  rotation = simulation.data.xmat[simulation.rocket_body_id].reshape(3, 3)
  expected_plume_direction = -rotation @ gimbal_direction_body(
    simulation.engine_gimbal_radians
  )
  assert maximum_fraction == pytest.approx(1.0)
  assert np.linalg.norm(maximum_vector) == pytest.approx(
    THRUST_ARROW_MAX_LENGTH_M
  )
  assert maximum_vector / np.linalg.norm(maximum_vector) == pytest.approx(
    expected_plume_direction
  )


def test_thrust_arrow_is_appended_only_while_engine_is_lit() -> None:
  window = RocketWindow.__new__(RocketWindow)
  window.simulation = RocketSimulation()
  window.scene = mujoco.MjvScene(window.simulation.model, maxgeom=100)
  camera = mujoco.MjvCamera()
  option = mujoco.MjvOption()
  mujoco.mjv_defaultCamera(camera)
  mujoco.mjv_defaultOption(option)

  mujoco.mjv_updateScene(
    window.simulation.model,
    window.simulation.data,
    option,
    None,
    camera,
    mujoco.mjtCatBit.mjCAT_ALL.value,
    window.scene,
  )
  base_geom_count = window.scene.ngeom
  window._append_thrust_arrow()
  assert window.scene.ngeom == base_geom_count

  window.simulation.controller.ignite()
  window._append_thrust_arrow()
  assert window.scene.ngeom == base_geom_count + 1
  assert window.scene.geoms[base_geom_count].type == mujoco.mjtGeom.mjGEOM_ARROW


def test_powered_six_dof_controller_removes_axial_spin_and_tilt_rates() -> None:
  simulation = RocketSimulation()
  yaw = math.radians(12.0)
  simulation.data.qpos[3:7] = (
    math.cos(yaw / 2.0),
    0.0,
    0.0,
    math.sin(yaw / 2.0),
  )
  simulation.data.qvel[3:6] = (0.03, -0.02, 0.12)
  mujoco.mj_forward(simulation.model, simulation.data)
  simulation.controller.ignite()
  simulation.controller.throttle = 0.60

  for _ in range(2000):
    simulation.step()

  assert np.linalg.norm(simulation.data.qvel[3:6]) < 1e-5
  assert abs(float(simulation.data.qpos[3]) - 1.0) < 1e-5
  assert np.linalg.norm(simulation.engine_gimbal_radians) < 1e-5


def test_roll_rcs_is_a_lagged_zero_net_force_couple() -> None:
  simulation = RocketSimulation()
  simulation.controller.ignite()
  simulation.roll_control_torque_command_nm = MAX_ROLL_CONTROL_TORQUE_NM

  simulation._update_roll_actuator()
  assert 0.0 < simulation.roll_control_torque_nm < MAX_ROLL_CONTROL_TORQUE_NM
  for _ in range(200):
    simulation._update_roll_actuator()

  positive_force, negative_force = simulation.roll_rcs_force_pair_body()
  assert positive_force + negative_force == pytest.approx(np.zeros(3))
  assert abs(positive_force[1]) <= ROLL_RCS_MAX_THRUSTER_FORCE_N

  mass_properties = simulation.applied_mass_properties()
  positive_position = simulation.model.site_pos[
    simulation.roll_rcs_xp_site_id
  ]
  negative_position = simulation.model.site_pos[
    simulation.roll_rcs_xm_site_id
  ]
  moment = np.cross(
    positive_position - mass_properties.center_of_mass_body_m,
    positive_force,
  ) + np.cross(
    negative_position - mass_properties.center_of_mass_body_m,
    negative_force,
  )
  assert moment[0:2] == pytest.approx(np.zeros(2), abs=1e-9)
  assert moment[2] == pytest.approx(simulation.roll_control_torque_nm)

  simulation.controller.kill_engine()
  simulation._apply_control()
  assert simulation.data.qfrc_applied[0:5] == pytest.approx(
    np.zeros(5), abs=1e-9
  )
  assert simulation.data.qfrc_applied[5] == pytest.approx(
    simulation.roll_control_torque_nm
  )


def test_solver_failure_selects_safe_six_dof_fallback() -> None:
  simulation = RocketSimulation(enable_mpc=True)

  class FailingMPC:
    def reset(self) -> None:
      pass

    def solve(self, state, target, previous_control):
      del state, target
      return MPCResult(
        success=False,
        control=previous_control,
        predicted_states=np.empty((14, 0)),
        status="forced_failure",
        solve_time_seconds=0.0,
        iterations=0,
        scaled_dynamics_defect=math.inf,
        scaled_virtual_control=math.inf,
      )

  simulation.mpc = FailingMPC()
  assert simulation.enable_hover()
  assert simulation.mpc_using_fallback
  for _ in range(100):
    simulation.step()

  assert simulation.controller.engine_state.name == "LIT"
  assert np.linalg.norm(simulation.engine_gimbal_radians) <= math.radians(20.0)


def test_async_mpc_uses_fallback_while_first_solution_is_pending() -> None:
  simulation = RocketSimulation(enable_mpc=True, asynchronous_mpc=True)

  class SlowMPC:
    def reset(self) -> None:
      pass

    def solve(self, state, target, previous_control):
      del state, target
      time.sleep(0.05)
      return MPCResult(
        success=False,
        control=previous_control,
        predicted_states=np.empty((14, 0)),
        status="delayed_failure",
        solve_time_seconds=0.05,
        iterations=0,
        scaled_dynamics_defect=math.inf,
        scaled_virtual_control=math.inf,
      )

  simulation.mpc = SlowMPC()
  assert simulation.enable_hover()
  assert simulation._mpc_future is not None
  assert simulation.mpc_using_fallback
  assert simulation.controller.throttle > simulation.controller.limits.min_throttle
  simulation.close()


def test_kill_button_hitbox_matches_drawn_rectangle() -> None:
  x, y, width, height = RocketWindow._engine_button_rect_window(1280)
  assert RocketWindow._point_in_engine_button(x + width / 2, y + height / 2, 1280)
  assert not RocketWindow._point_in_engine_button(x - 1, y + height / 2, 1280)
  assert not RocketWindow._point_in_engine_button(
    x + width / 2, y + height + 1, 1280
  )

  hover_x, hover_y, hover_width, hover_height = (
    RocketWindow._hover_button_rect_window(1280)
  )
  assert RocketWindow._point_in_hover_button(
    hover_x + hover_width / 2, hover_y + hover_height / 2, 1280
  )

  land_x, land_y, land_width, land_height = RocketWindow._land_button_rect_window(
    1280
  )
  assert RocketWindow._point_in_land_button(
    land_x + land_width / 2, land_y + land_height / 2, 1280
  )

  direction_rects = RocketWindow._direction_button_rects_window(1280)
  for direction, (button_x, button_y, button_width, button_height) in (
    direction_rects.items()
  ):
    command = RocketWindow._direction_command_for_point(
      button_x + button_width / 2,
      button_y + button_height / 2,
      1280,
    )
    assert command is not None, direction

  slider_x, slider_y, slider_width, slider_height = (
    RocketWindow._thrust_slider_rect_window(1280)
  )
  assert RocketWindow._point_in_thrust_slider(
    slider_x + slider_width / 2,
    slider_y + slider_height / 2,
    1280,
  )


def test_engine_off_rocket_settles_without_attitude_controller_vibration() -> None:
  simulation = RocketSimulation()
  vertical_speeds = []
  for _ in range(1600):
    simulation.step()
    vertical_speeds.append(abs(float(simulation.data.qvel[2])))
  assert max(vertical_speeds[-200:]) < 0.03


def test_gui_actions_ignite_kill_and_reset_without_a_gl_context() -> None:
  window = RocketWindow.__new__(RocketWindow)
  window.simulation = RocketSimulation()
  window.status_message = ""
  window.status_until = 0.0

  window._apply_throttle_input(1.0, 0.1)
  assert window.simulation.controller.engine_state.name == "LIT"
  assert window.simulation.controller.throttle > 0.20

  window._kill_engine()
  assert window.simulation.controller.engine_state.name == "SHUTDOWN"
  assert window.simulation.controller.thrust_magnitude_newtons() == 0.0

  window._reset_flight()
  assert window.simulation.controller.engine_state.name == "OFF"
  assert window.simulation.controller.fuel_mass_kg == (
    window.simulation.controller.initial_fuel_mass_kg
  )


def test_gui_thrust_slider_maps_limits_and_becomes_read_only_in_hover() -> None:
  window = RocketWindow.__new__(RocketWindow)
  window.simulation = RocketSimulation()
  window.status_message = ""
  window.status_until = 0.0
  slider_x, _, slider_width, _ = window._thrust_slider_rect_window(1280)

  assert window._throttle_from_slider_x(slider_x, 1280) == pytest.approx(0.20)
  assert window._throttle_from_slider_x(
    slider_x + slider_width, 1280
  ) == pytest.approx(0.80)
  assert window._set_manual_throttle_from_slider(
    slider_x + slider_width / 2, 1280
  )
  assert window.simulation.controller.engine_state.name == "LIT"
  assert window.simulation.controller.throttle == pytest.approx(0.50)

  assert window.simulation.enable_hover()
  autopilot_throttle = window.simulation.controller.throttle
  assert not window._set_manual_throttle_from_slider(
    slider_x + slider_width, 1280
  )
  assert window.simulation.controller.throttle == pytest.approx(
    autopilot_throttle
  )


def test_direction_indicator_levels_match_gimbal_command() -> None:
  levels = RocketWindow._direction_button_levels(np.array([0.4, -0.7]))
  assert levels == {"W": 0.0, "A": 0.0, "S": 0.7, "D": 0.4}


def test_controller_indicator_reports_manual_fallback_and_mpc_ownership() -> None:
  window = RocketWindow.__new__(RocketWindow)
  window.simulation = RocketSimulation()

  label, _ = window._controller_indicator_style()
  assert label == "MANUAL TVC"

  window.simulation.hover_enabled = True
  label, _ = window._controller_indicator_style()
  assert label == "FALLBACK ACTIVE"

  window.simulation.enable_mpc = True
  window.simulation.mpc_using_fallback = False
  window.simulation.last_mpc_result = MPCResult(
    success=True,
    control=np.zeros(4),
    predicted_states=np.empty((14, 0)),
    status="optimal",
    solve_time_seconds=0.01,
    iterations=1,
    scaled_dynamics_defect=0.0,
    scaled_virtual_control=0.0,
  )
  label, _ = window._controller_indicator_style()
  assert label == "MPC ACTIVE"

  window.simulation.landing_phase = LandingPhase.DESCEND
  window.simulation.data.qpos[2] = ROCKET_LANDED_COM_Z_M + 5.0
  label, _ = window._controller_indicator_style()
  assert label == "TERMINAL ACTIVE"


def test_short_key_events_trigger_mode_commands() -> None:
  window = RocketWindow.__new__(RocketWindow)
  window.simulation = RocketSimulation()
  window.status_message = ""
  window.status_until = 0.0
  window.previous_command_keys = {
    glfw.KEY_H: False,
    glfw.KEY_I: False,
    glfw.KEY_K: False,
    glfw.KEY_L: False,
    glfw.KEY_R: False,
  }

  window._on_key(None, glfw.KEY_H, 0, glfw.PRESS, 0)
  assert window.simulation.hover_enabled
  assert window.simulation.controller.engine_state.name == "LIT"
  window._on_key(None, glfw.KEY_H, 0, glfw.RELEASE, 0)

  window._on_key(None, glfw.KEY_K, 0, glfw.PRESS, 0)
  assert window.simulation.controller.engine_state.name == "SHUTDOWN"
  window._on_key(None, glfw.KEY_K, 0, glfw.RELEASE, 0)

  window._on_key(None, glfw.KEY_R, 0, glfw.PRESS, 0)
  assert window.simulation.controller.engine_state.name == "OFF"
