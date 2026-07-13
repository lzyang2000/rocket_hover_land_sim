"""Interactive MuJoCo rocket simulation with a native clickable control panel."""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
from enum import Enum, auto
import math
from pathlib import Path
import time

import glfw
import mujoco
import numpy as np

from rocket_landing.controller import EngineState, RocketController
from rocket_landing.mass_properties import MassProperties, RocketMassModel
from rocket_landing.mpc import (
  ANGULAR_VELOCITY,
  GIMBAL,
  MASS,
  MPCConfig,
  MPCResult,
  POSITION,
  QUATERNION,
  ROLL_TORQUE,
  SixDofMPC,
  THRUST,
  VELOCITY,
  gimbal_direction_body,
  normalize_quaternion,
  quaternion_slerp,
)


THROTTLE_RATE_PER_SECOND = 0.12
LATERAL_SLEW_RATE_PER_SECOND = 2.4
ATTITUDE_KP_BODY = np.array([10_000_000.0, 10_000_000.0, 1_000_000.0])
ATTITUDE_KD_BODY = np.array([13_000_000.0, 13_000_000.0, 350_000.0])
ROLL_RCS_LEVER_ARM_M = 1.75
ROLL_RCS_MAX_THRUSTER_FORCE_N = 5_000.0
MAX_ROLL_CONTROL_TORQUE_NM = (
  2.0 * ROLL_RCS_LEVER_ARM_M * ROLL_RCS_MAX_THRUSTER_FORCE_N
)
ROLL_RCS_TIME_CONSTANT_S = 0.10
GIMBAL_TIME_CONSTANT_S = 0.08
TERMINAL_GIMBAL_TIME_CONSTANT_S = 0.20
TERMINAL_GIMBAL_DEADBAND_RADIANS = math.radians(0.15)
HOVER_MPC_GIMBAL_LIMIT_RADIANS = math.radians(5.0)
LANDING_GIMBAL_LIMIT_RADIANS = math.radians(6.0)
HOVER_TARGET_HORIZONTAL_LEAD_M = 3.5
HOVER_TARGET_VERTICAL_LEAD_M = 2.0
LANDING_ALIGNMENT_HORIZONTAL_LEAD_M = 4.0
LANDING_ALIGNMENT_MAX_HORIZONTAL_LEAD_M = 8.0
LANDING_ALIGNMENT_LEAD_PER_HEIGHT = 0.15
LANDING_ALIGNMENT_VERTICAL_LEAD_M = 2.0
LANDING_ALTITUDE_SCHEDULE_START_HEIGHT_M = 18.0
LANDING_DESCENT_CAPTURE_RADIUS_M = 2.0
LANDING_DESCENT_CAPTURE_HORIZONTAL_SPEED_MPS = 1.0
LANDING_DESCENT_CAPTURE_ALTITUDE_ERROR_M = 2.0
LANDING_DESCENT_CAPTURE_VERTICAL_SPEED_MPS = 1.5
MPC_TERMINAL_HANDOFF_HEIGHT_M = 7.0
AUTO_LAND_FUEL_MARGIN = 1.05
AUTO_LAND_FUEL_CHECK_PERIOD_S = 0.25
AUTO_LAND_TAKEOVER_MIN_HEIGHT_M = 0.15
AUTO_LAND_FIXED_FUEL_RESERVE_KG = 250.0
AUTO_LAND_ALIGNMENT_SPEED_MPS = 1.5
AUTO_LAND_ALIGNMENT_BRAKING_ACCEL_MPS2 = 0.75
AUTO_LAND_ALIGNMENT_SETTLE_TIME_S = 2.5
AUTO_LAND_DESCENT_TIME_MARGIN = 1.10
AUTO_LAND_CONTROL_IMPULSE_MARGIN = 1.10
AUTO_LAND_MINIMUM_TAKEOVER_FUEL_KG = 5_500.0
AUTO_LAND_HEIGHT_RESERVE_KG_PER_M = 25.0
AUTO_LAND_HORIZONTAL_RESERVE_KG_PER_M = 320.0
COAST_MIN_ENTRY_HEIGHT_M = 32.0
COAST_MIN_ALTITUDE_SAVING_M = 5.0
COAST_ENTRY_MAX_TILT_RADIANS = math.radians(2.5)
COAST_ENTRY_MAX_ANGULAR_RATE_RPS = 0.12
COAST_ABORT_MAX_TILT_RADIANS = math.radians(5.0)
COAST_ABORT_MAX_ANGULAR_RATE_RPS = 0.25
COAST_ABORT_MAX_HORIZONTAL_ERROR_M = 2.75
COAST_ABORT_MAX_HORIZONTAL_SPEED_MPS = 1.50
COAST_TARGET_TOUCHDOWN_SPEED_MPS = 0.50
COAST_IGNITION_DELAY_S = 0.45
COAST_STOPPING_DISTANCE_FACTOR = 1.25
COAST_FIXED_BURN_MARGIN_M = 6.0
COAST_BURN_REFERENCE_DECELERATION_MPS2 = 4.0
COAST_BURN_REFERENCE_TERMINAL_HEIGHT_M = 6.0
MASS_UPDATE_PERIOD_S = 0.10
MPC_UPDATE_PERIOD_S = 0.30
ASYNC_MPC_MAX_ACCEPT_AGE_S = 0.35
ASYNC_MPC_MAX_POSITION_MISMATCH_M = 2.0
ASYNC_MPC_MAX_VELOCITY_MISMATCH_MPS = 2.5
ASYNC_MPC_MAX_ATTITUDE_MISMATCH_RADIANS = math.radians(15.0)
ASYNC_MPC_MAX_ANGULAR_RATE_MISMATCH_RPS = 0.40
ASYNC_MPC_MAX_HOVER_TARGET_SHIFT_M = 1.0
ASYNC_MPC_MAX_LANDING_TARGET_SHIFT_M = 5.0
ASYNC_MPC_MAX_FEEDFORWARD_ACCELERATION_MPS2 = 3.0
ASYNC_MPC_HOVER_TRACKING_LOOKAHEAD_S = 1.05
ASYNC_MPC_LANDING_TRACKING_LOOKAHEAD_S = 0.35
ASYNC_MPC_HOVER_TARGET_BLEND = 0.85
ASYNC_MPC_LANDING_TARGET_BLEND = 0.85
ASYNC_MPC_HOVER_VELOCITY_BLEND = 1.0
ASYNC_MPC_LANDING_VELOCITY_BLEND = 1.0
ASYNC_MPC_HOVER_POSITION_GAIN_SCALE = 1.50
ASYNC_MPC_LANDING_POSITION_GAIN_SCALE = 1.0
HOVER_POSITION_KP = np.array([0.12, 0.12, 0.80])
HOVER_VELOCITY_KD = np.array([0.70, 0.70, 1.80])
HOVER_TARGET_SPEED_MPS = 3.0
ROCKET_HEIGHT_M = 41.2
ROCKET_DIAMETER_M = 3.66
ROCKET_LANDED_COM_Z_M = 20.76
LANDING_PAD_POSITION = np.array([0.0, 0.0, ROCKET_LANDED_COM_Z_M])
LANDING_STAGING_HEIGHT_M = 25.0
LANDING_STAGING_MIN_ALTITUDE_M = (
  ROCKET_LANDED_COM_Z_M + LANDING_STAGING_HEIGHT_M
)
LANDING_LEG_DEPLOYMENT_TIME_S = 1.25
LANDING_LEG_MAIN_PIVOT_RADIUS_M = 1.63
LANDING_LEG_MAIN_PIVOT_Z_M = -13.10
LANDING_LEG_STRUT_PIVOT_RADIUS_M = 1.72
LANDING_LEG_STRUT_PIVOT_Z_M = -17.15
LANDING_LEG_DEPLOYED_FOOT_RADIUS_M = 8.50
LANDING_LEG_DEPLOYED_FOOT_Z_M = -20.38
LANDING_LEG_STOWED_FOOT_RADIUS_M = 2.05
LANDING_LEG_STOWED_FOOT_Z_M = -3.10
LANDING_LEG_STOWED_STRUT_RADIUS_M = 1.92
LANDING_LEG_STOWED_STRUT_Z_M = -11.50
FLAME_ORIGIN_BODY_Z_M = -20.10
ENGINE_POSITION_BODY = np.array([0.0, 0.0, FLAME_ORIGIN_BODY_Z_M])
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 820
GUI_PANEL_MIN_WIDTH = 188.0
GUI_PANEL_MAX_WIDTH = 260.0
GUI_PANEL_WIDTH_FRACTION = 0.20
GUI_PANEL_MARGIN = 22.0
THRUST_ARROW_MAX_LENGTH_M = 36.0
THRUST_ARROW_MIN_WIDTH_M = 0.30
THRUST_ARROW_MAX_WIDTH_M = 0.66
APP_TITLE = "MuJoCo Powered Descent Lab v0.9.21 - 6-DOF SCvx MPC"


class LandingPhase(Enum):
  INACTIVE = auto()
  ALIGN = auto()
  COAST = auto()
  DESCEND = auto()
  COMPLETE = auto()
  ABORTED = auto()


def model_path() -> Path:
  return Path(__file__).with_name("assets") / "rocket.xml"


class RocketSimulation:
  """MuJoCo plant plus the paper-inspired constrained thrust controller."""

  def __init__(
    self,
    *,
    enable_mpc: bool = False,
    asynchronous_mpc: bool = False,
  ) -> None:
    self.model = mujoco.MjModel.from_xml_path(str(model_path()))
    self.data = mujoco.MjData(self.model)
    self.controller = RocketController()
    self.rocket_body_id = mujoco.mj_name2id(
      self.model, mujoco.mjtObj.mjOBJ_BODY, "rocket"
    )
    self.flame_geom_id = mujoco.mj_name2id(
      self.model, mujoco.mjtObj.mjOBJ_GEOM, "engine_flame"
    )
    self.thrust_site_id = mujoco.mj_name2id(
      self.model, mujoco.mjtObj.mjOBJ_SITE, "thrust_origin"
    )
    self.roll_rcs_xp_site_id = mujoco.mj_name2id(
      self.model, mujoco.mjtObj.mjOBJ_SITE, "roll_rcs_xp"
    )
    self.roll_rcs_xm_site_id = mujoco.mj_name2id(
      self.model, mujoco.mjtObj.mjOBJ_SITE, "roll_rcs_xm"
    )
    self.launch_mount_geom_id = mujoco.mj_name2id(
      self.model, mujoco.mjtObj.mjOBJ_GEOM, "launch_mount"
    )
    self._launch_mount_contype = int(
      self.model.geom_contype[self.launch_mount_geom_id]
    )
    self._launch_mount_conaffinity = int(
      self.model.geom_conaffinity[self.launch_mount_geom_id]
    )
    self.landing_leg_axes = {
      "xp": np.array([1.0, 0.0, 0.0]),
      "xm": np.array([-1.0, 0.0, 0.0]),
      "yp": np.array([0.0, 1.0, 0.0]),
      "ym": np.array([0.0, -1.0, 0.0]),
    }
    self.landing_leg_geom_ids = {
      suffix: mujoco.mj_name2id(
        self.model, mujoco.mjtObj.mjOBJ_GEOM, f"leg_{suffix}"
      )
      for suffix in self.landing_leg_axes
    }
    self.landing_strut_geom_ids = {
      suffix: mujoco.mj_name2id(
        self.model, mujoco.mjtObj.mjOBJ_GEOM, f"strut_{suffix}"
      )
      for suffix in self.landing_leg_axes
    }
    self.landing_foot_geom_ids = {
      suffix: mujoco.mj_name2id(
        self.model, mujoco.mjtObj.mjOBJ_GEOM, f"foot_{suffix}"
      )
      for suffix in self.landing_leg_axes
    }
    self._deployed_foot_quaternions = {
      suffix: self.model.geom_quat[geom_id].copy()
      for suffix, geom_id in self.landing_foot_geom_ids.items()
    }
    half_sqrt = math.sqrt(0.5)
    self._stowed_foot_quaternions = {
      "xp": np.array([half_sqrt, 0.0, -half_sqrt, 0.0]),
      "xm": np.array([half_sqrt, 0.0, -half_sqrt, 0.0]),
      "yp": np.array([half_sqrt, half_sqrt, 0.0, 0.0]),
      "ym": np.array([half_sqrt, half_sqrt, 0.0, 0.0]),
    }
    self.initial_body_inertia = self.model.body_inertia[
      self.rocket_body_id
    ].copy()
    self.mass_model = RocketMassModel(
      dry_mass_kg=self.controller.dry_mass_kg,
      initial_propellant_mass_kg=self.controller.initial_fuel_mass_kg,
      initial_inertia_kgm2=tuple(
        float(value) for value in self.initial_body_inertia
      ),
    )
    self.last_mass_update_time = -math.inf
    self.last_applied_mass = math.nan
    self.enable_mpc = enable_mpc
    self.asynchronous_mpc = asynchronous_mpc and enable_mpc
    self._mpc_config = MPCConfig(
      initial_mass_kg=(
        self.controller.dry_mass_kg + self.controller.initial_fuel_mass_kg
      ),
      dry_mass_kg=self.controller.dry_mass_kg,
      initial_inertia_kgm2=tuple(float(value) for value in self.initial_body_inertia),
      mass_model=self.mass_model,
      engine_position_body_m=tuple(float(value) for value in ENGINE_POSITION_BODY),
      alpha_kg_per_newton_second=(
        self.controller.limits.alpha_kg_per_newton_second
      ),
      min_thrust_newtons=self.controller.limits.min_thrust_newtons,
      max_thrust_newtons=self.controller.limits.max_thrust_newtons,
      max_gimbal_radians=LANDING_GIMBAL_LIMIT_RADIANS,
      max_roll_torque_nm=MAX_ROLL_CONTROL_TORQUE_NM,
      minimum_com_height_m=(
        ROCKET_LANDED_COM_Z_M
        + self.mass_model.properties(self.controller.dry_mass_kg)
        .center_of_mass_body_m[2]
        - 0.10
      ),
      maximum_scaled_defect=0.20,
    )
    self.mpc = SixDofMPC(self._mpc_config) if enable_mpc else None
    self._mpc_executor = (
      ThreadPoolExecutor(max_workers=1, thread_name_prefix="rocket-mpc")
      if self.asynchronous_mpc
      else None
    )
    self._mpc_future: Future[
      tuple[int, float, np.ndarray, np.ndarray, MPCResult]
    ] | None = None
    self._mpc_future_metadata: (
      tuple[int, float, np.ndarray, np.ndarray] | None
    ) = None
    self._mpc_generation = 0
    self.last_mpc_result: MPCResult | None = None
    self.last_mpc_request_time = -math.inf
    self.mpc_using_fallback = True
    self._active_async_mpc_result: MPCResult | None = None
    self._active_async_mpc_request_time = -math.inf
    self._active_async_mpc_target_position = np.zeros(3, dtype=float)
    self._active_async_mpc_target_velocity = np.zeros(3, dtype=float)
    self.async_mpc_rejection_reason = ""
    self.engine_gimbal_command_radians = np.zeros(2, dtype=float)
    self.engine_gimbal_radians = np.zeros(2, dtype=float)
    self.roll_control_torque_command_nm = 0.0
    self.roll_control_torque_nm = 0.0
    self.hover_enabled = False
    self.hover_target_position = np.zeros(3, dtype=float)
    self.hover_target_velocity = np.zeros(3, dtype=float)
    self.landing_phase = LandingPhase.INACTIVE
    self.landing_staging_altitude = LANDING_STAGING_MIN_ALTITUDE_M
    self.landing_burn_from_coast = False
    self.landing_leg_deployment = 0.0
    self.landing_legs_deploy_commanded = False
    self.launch_mount_enabled = True
    self.fuel_takeover_triggered = False
    self.fuel_takeover_active = False
    self.last_fuel_takeover_check_time = -math.inf
    self.last_estimated_landing_fuel_kg = 0.0
    self.reset()

  def reset(self) -> None:
    self._invalidate_mpc_solution()
    self.controller.reset()
    mujoco.mj_resetData(self.model, self.data)
    self.hover_enabled = False
    self.hover_target_position = self.data.qpos[0:3].copy()
    self.hover_target_velocity[:] = 0.0
    self.landing_phase = LandingPhase.INACTIVE
    self.landing_staging_altitude = LANDING_STAGING_MIN_ALTITUDE_M
    self.landing_burn_from_coast = False
    self.landing_leg_deployment = 0.0
    self.landing_legs_deploy_commanded = False
    self.fuel_takeover_triggered = False
    self.fuel_takeover_active = False
    self.last_fuel_takeover_check_time = -math.inf
    self.last_estimated_landing_fuel_kg = 0.0
    self.last_mass_update_time = -math.inf
    self.last_applied_mass = math.nan
    self.engine_gimbal_command_radians[:] = 0.0
    self.engine_gimbal_radians[:] = 0.0
    self.roll_control_torque_command_nm = 0.0
    self.roll_control_torque_nm = 0.0
    self._set_landing_leg_deployment(0.0)
    self._set_launch_mount_enabled(True)
    self._update_model_mass(force=True)
    self.hover_target_position = self.center_of_mass_position_world()
    self._update_flame()
    mujoco.mj_forward(self.model, self.data)

  def close(self) -> None:
    if self._mpc_executor is not None:
      self._mpc_executor.shutdown(wait=False, cancel_futures=True)
      self._mpc_executor = None

  def warm_up_mpc(self) -> None:
    """Compile and cache the MPC problem without changing the vehicle state."""

    if not isinstance(self.mpc, SixDofMPC):
      return
    self.mpc.solve(
      self._mpc_state(),
      self._mpc_target_state(),
      self._current_actuator_control(),
      max_gimbal_radians=self._mpc_gimbal_limit_radians(),
      central_differences=False,
    )
    self.mpc.reset()

  def _invalidate_mpc_solution(self) -> None:
    self._mpc_generation += 1
    self.last_mpc_result = None
    self.last_mpc_request_time = -math.inf
    self.mpc_using_fallback = True
    self._active_async_mpc_result = None
    self._active_async_mpc_request_time = -math.inf
    self._active_async_mpc_target_position[:] = 0.0
    self._active_async_mpc_target_velocity[:] = 0.0
    self.async_mpc_rejection_reason = ""
    if self.mpc is not None and self._mpc_future is None:
      self.mpc.reset()

  def enable_hover(self) -> bool:
    if self.controller.engine_state is EngineState.OFF:
      self.controller.ignite()
    elif self.controller.engine_state is EngineState.COAST:
      self.controller.relight()
    if self.controller.engine_state is not EngineState.LIT:
      return False
    self._invalidate_mpc_solution()
    self.hover_target_position = self.center_of_mass_position_world()
    self.hover_target_velocity[:] = 0.0
    self.hover_enabled = True
    self.landing_phase = LandingPhase.INACTIVE
    self._update_hover_controller(force=True)
    return True

  def disable_hover(self) -> None:
    if (
      self.landing_phase is LandingPhase.COAST
      and self.controller.engine_state is EngineState.COAST
    ):
      self.controller.relight()
    self._invalidate_mpc_solution()
    self.hover_enabled = False
    self.hover_target_velocity[:] = 0.0
    self.controller.center_lateral()
    if self.landing_phase not in (LandingPhase.COMPLETE, LandingPhase.ABORTED):
      self.landing_phase = LandingPhase.INACTIVE
      self.landing_burn_from_coast = False

  @property
  def landing_active(self) -> bool:
    return self.landing_phase in (
      LandingPhase.ALIGN,
      LandingPhase.COAST,
      LandingPhase.DESCEND,
    )

  @property
  def terminal_controller_active(self) -> bool:
    return (
      self.landing_phase is LandingPhase.DESCEND
      and float(self.data.qpos[2] - LANDING_PAD_POSITION[2])
      <= MPC_TERMINAL_HANDOFF_HEIGHT_M
    )

  def _set_launch_mount_enabled(self, enabled: bool) -> None:
    """Enable the invisible reset support until the vehicle has lifted off."""

    self.launch_mount_enabled = bool(enabled)
    if self.launch_mount_enabled:
      self.model.geom_contype[self.launch_mount_geom_id] = (
        self._launch_mount_contype
      )
      self.model.geom_conaffinity[self.launch_mount_geom_id] = (
        self._launch_mount_conaffinity
      )
    else:
      self.model.geom_contype[self.launch_mount_geom_id] = 0
      self.model.geom_conaffinity[self.launch_mount_geom_id] = 0

  @staticmethod
  def _capsule_pose_from_endpoints(
    start: np.ndarray, end: np.ndarray
  ) -> tuple[np.ndarray, np.ndarray, float]:
    """Return midpoint, +Z alignment quaternion, and half-length."""

    first = np.asarray(start, dtype=float)
    second = np.asarray(end, dtype=float)
    delta = second - first
    length = float(np.linalg.norm(delta))
    if length < 1e-12:
      return first.copy(), np.array([1.0, 0.0, 0.0, 0.0]), 0.0

    direction = delta / length
    dot = float(np.clip(direction[2], -1.0, 1.0))
    if dot < -1.0 + 1e-10:
      quaternion = np.array([0.0, 1.0, 0.0, 0.0])
    else:
      cross = np.cross(np.array([0.0, 0.0, 1.0]), direction)
      scale = math.sqrt(max(2.0 * (1.0 + dot), 1e-12))
      quaternion = np.array(
        [
          scale / 2.0,
          cross[0] / scale,
          cross[1] / scale,
          cross[2] / scale,
        ]
      )
    return 0.5 * (first + second), quaternion, 0.5 * length

  def _set_landing_leg_deployment(self, fraction: float) -> None:
    """Move all rendered and colliding leg geoms to one deployment state."""

    deployment = float(np.clip(fraction, 0.0, 1.0))
    self.landing_leg_deployment = deployment
    interpolation = deployment * deployment * (3.0 - 2.0 * deployment)

    for suffix, axis in self.landing_leg_axes.items():
      main_pivot = np.array(
        [
          axis[0] * LANDING_LEG_MAIN_PIVOT_RADIUS_M,
          axis[1] * LANDING_LEG_MAIN_PIVOT_RADIUS_M,
          LANDING_LEG_MAIN_PIVOT_Z_M,
        ]
      )
      strut_pivot = np.array(
        [
          axis[0] * LANDING_LEG_STRUT_PIVOT_RADIUS_M,
          axis[1] * LANDING_LEG_STRUT_PIVOT_RADIUS_M,
          LANDING_LEG_STRUT_PIVOT_Z_M,
        ]
      )
      stowed_foot = np.array(
        [
          axis[0] * LANDING_LEG_STOWED_FOOT_RADIUS_M,
          axis[1] * LANDING_LEG_STOWED_FOOT_RADIUS_M,
          LANDING_LEG_STOWED_FOOT_Z_M,
        ]
      )
      deployed_foot = np.array(
        [
          axis[0] * LANDING_LEG_DEPLOYED_FOOT_RADIUS_M,
          axis[1] * LANDING_LEG_DEPLOYED_FOOT_RADIUS_M,
          LANDING_LEG_DEPLOYED_FOOT_Z_M,
        ]
      )
      stowed_strut_end = np.array(
        [
          axis[0] * LANDING_LEG_STOWED_STRUT_RADIUS_M,
          axis[1] * LANDING_LEG_STOWED_STRUT_RADIUS_M,
          LANDING_LEG_STOWED_STRUT_Z_M,
        ]
      )
      foot_position = (
        (1.0 - interpolation) * stowed_foot
        + interpolation * deployed_foot
      )
      strut_end = (
        (1.0 - interpolation) * stowed_strut_end
        + interpolation * deployed_foot
      )

      leg_position, leg_quaternion, leg_half_length = (
        self._capsule_pose_from_endpoints(main_pivot, foot_position)
      )
      leg_geom_id = self.landing_leg_geom_ids[suffix]
      self.model.geom_pos[leg_geom_id] = leg_position
      self.model.geom_quat[leg_geom_id] = leg_quaternion
      self.model.geom_size[leg_geom_id, 1] = leg_half_length

      strut_position, strut_quaternion, strut_half_length = (
        self._capsule_pose_from_endpoints(strut_pivot, strut_end)
      )
      strut_geom_id = self.landing_strut_geom_ids[suffix]
      self.model.geom_pos[strut_geom_id] = strut_position
      self.model.geom_quat[strut_geom_id] = strut_quaternion
      self.model.geom_size[strut_geom_id, 1] = strut_half_length

      foot_geom_id = self.landing_foot_geom_ids[suffix]
      self.model.geom_pos[foot_geom_id] = foot_position
      self.model.geom_quat[foot_geom_id] = quaternion_slerp(
        self._stowed_foot_quaternions[suffix],
        self._deployed_foot_quaternions[suffix],
        interpolation,
      )

  def _update_landing_legs(self) -> None:
    """Latch and animate deployment during the terminal landing phase."""

    height = float(self.data.qpos[2] - ROCKET_LANDED_COM_Z_M)
    if self.launch_mount_enabled and (
      self.landing_active or height > 0.05
    ):
      self._set_launch_mount_enabled(False)

    if (
      self.terminal_controller_active
      or self.landing_phase is LandingPhase.COMPLETE
    ):
      self.landing_legs_deploy_commanded = True

    if (
      self.landing_legs_deploy_commanded
      and self.landing_leg_deployment < 1.0
    ):
      self._set_landing_leg_deployment(
        self.landing_leg_deployment
        + self.model.opt.timestep / LANDING_LEG_DEPLOYMENT_TIME_S
      )

  def applied_mass_properties(self) -> MassProperties:
    """Return the mass properties currently installed in MuJoCo."""

    return MassProperties(
      mass_kg=float(self.model.body_mass[self.rocket_body_id]),
      center_of_mass_body_m=self.model.body_ipos[self.rocket_body_id].copy(),
      inertia_at_com_kgm2=self.model.body_inertia[self.rocket_body_id].copy(),
    )

  def current_mass_properties(self) -> MassProperties:
    """Return continuous mass properties at the controller's current fuel mass."""

    return self.mass_model.properties(self.controller.wet_mass_kg)

  def center_of_mass_offset_rate_body(self) -> np.ndarray:
    """Body-frame COM migration rate caused by current propellant flow."""

    if self.controller.engine_state is not EngineState.LIT:
      return np.zeros(3, dtype=float)
    mass = self.controller.wet_mass_kg
    if mass <= self.controller.dry_mass_kg:
      return np.zeros(3, dtype=float)
    lower_mass = max(mass - 1.0, self.controller.dry_mass_kg)
    upper_mass = min(mass + 1.0, self.mass_model.initial_mass_kg)
    if upper_mass <= lower_mass:
      return np.zeros(3, dtype=float)
    lower_com = self.mass_model.properties(
      lower_mass
    ).center_of_mass_body_m
    upper_com = self.mass_model.properties(
      upper_mass
    ).center_of_mass_body_m
    derivative_per_kg = (upper_com - lower_com) / (
      upper_mass - lower_mass
    )
    mass_rate = -(
      self.controller.limits.alpha_kg_per_newton_second
      * self.controller.thrust_magnitude_newtons()
    )
    return derivative_per_kg * mass_rate

  def center_of_mass_position_world(self) -> np.ndarray:
    """World position of the current fuel-dependent center of mass."""

    rotation = self.data.xmat[self.rocket_body_id].reshape(3, 3)
    center_of_mass_body = (
      self.current_mass_properties().center_of_mass_body_m
    )
    return (
      self.data.qpos[0:3]
      + rotation @ center_of_mass_body
    )

  def center_of_mass_velocity_world(self) -> np.ndarray:
    """World linear velocity of the current center of mass."""

    rotation = self.data.xmat[self.rocket_body_id].reshape(3, 3)
    offset_body = self.current_mass_properties().center_of_mass_body_m
    return (
      self.data.qvel[0:3]
      + rotation @ np.cross(self.data.qvel[3:6], offset_body)
      + rotation @ self.center_of_mass_offset_rate_body()
    )

  def landing_center_of_mass_position(self) -> np.ndarray:
    """Upright COM target when the landing-leg reference is on the pad."""

    target = LANDING_PAD_POSITION.copy()
    target[2] += self.current_mass_properties().center_of_mass_body_m[2]
    return target

  def _landing_staging_altitude_for_current_state(
    self, *, fuel_takeover: bool
  ) -> float:
    """Choose a staging altitude reachable through the current climb."""

    position = self.center_of_mass_position_world()
    velocity = self.center_of_mass_velocity_world()
    current_altitude = float(position[2])
    landed_com_altitude = float(self.landing_center_of_mass_position()[2])
    staging_altitude = (
      current_altitude
      if fuel_takeover
      else max(
        current_altitude,
        landed_com_altitude + LANDING_STAGING_HEIGHT_M,
      )
    )

    upward_speed = max(float(velocity[2]), 0.0)
    if upward_speed <= 0.0:
      return staging_altitude

    gravity = abs(float(self.model.opt.gravity[2]))
    minimum_thrust_acceleration = (
      self.controller.limits.min_thrust_newtons
      / self.controller.wet_mass_kg
    )
    maximum_downward_acceleration = gravity - minimum_thrust_acceleration
    if maximum_downward_acceleration <= 1e-6:
      return staging_altitude

    stopping_distance = upward_speed * upward_speed / (
      2.0 * maximum_downward_acceleration
    )
    return max(staging_altitude, current_altitude + stopping_distance)

  def start_landing(self, *, fuel_takeover: bool = False) -> bool:
    if self.controller.engine_state is EngineState.OFF:
      self.controller.ignite()
    elif self.controller.engine_state is EngineState.COAST:
      self.controller.relight()
    if self.controller.engine_state is not EngineState.LIT:
      return False

    self._invalidate_mpc_solution()
    self._set_launch_mount_enabled(False)

    self.landing_staging_altitude = (
      self._landing_staging_altitude_for_current_state(
        fuel_takeover=fuel_takeover
      )
    )
    self.hover_target_position = self.center_of_mass_position_world()
    self.hover_target_velocity[:] = 0.0
    self.hover_enabled = True
    self.landing_phase = LandingPhase.ALIGN
    self.landing_burn_from_coast = False
    self.fuel_takeover_active = fuel_takeover
    if fuel_takeover:
      self.fuel_takeover_triggered = True
    self._update_hover_controller(force=True)
    return True

  def cancel_landing(self) -> None:
    if self.landing_active:
      if (
        self.landing_phase is LandingPhase.COAST
        and not self.controller.relight()
      ):
        self.landing_phase = LandingPhase.ABORTED
        self.hover_enabled = False
        return
      self._invalidate_mpc_solution()
      self.landing_phase = LandingPhase.INACTIVE
      self.landing_burn_from_coast = False
      self.fuel_takeover_active = False
      self.hover_target_position = self.center_of_mass_position_world()
      self.hover_target_velocity[:] = 0.0
      self.hover_enabled = True

  @staticmethod
  def descent_rate_for_height_mps(height_m: float) -> float:
    """Return the nonzero speed assigned to the current altitude band."""

    height = max(float(height_m), 0.0)
    if height > 30.0:
      return 12.0
    if height > 18.0:
      return 8.0
    if height > 10.0:
      return 5.0
    if height > 5.0:
      return 3.0
    if height > 2.5:
      return 1.5
    if height > 1.0:
      return 0.6
    return 0.25

  def _descent_rate_mps(self) -> float:
    height = float(self.data.qpos[2] - LANDING_PAD_POSITION[2])
    scheduled_rate = self.descent_rate_for_height_mps(height)
    if not self.landing_burn_from_coast:
      return scheduled_rate
    energy_rate = math.sqrt(
      COAST_TARGET_TOUCHDOWN_SPEED_MPS ** 2
      + 2.0
      * COAST_BURN_REFERENCE_DECELERATION_MPS2
      * max(height - COAST_BURN_REFERENCE_TERMINAL_HEIGHT_M, 0.0)
    )
    return max(scheduled_rate, energy_rate)

  @staticmethod
  def landing_horizontal_lead_for_height_m(height_m: float) -> float:
    """Increase lateral target lead only where altitude leaves braking time."""

    expanded_lead = (
      LANDING_ALIGNMENT_HORIZONTAL_LEAD_M
      + LANDING_ALIGNMENT_LEAD_PER_HEIGHT
      * max(
        float(height_m) - LANDING_ALTITUDE_SCHEDULE_START_HEIGHT_M,
        0.0,
      )
    )
    return min(expanded_lead, LANDING_ALIGNMENT_MAX_HORIZONTAL_LEAD_M)

  @staticmethod
  def _bounded_landing_horizontal_target(
    position: np.ndarray,
    landed_position: np.ndarray,
    maximum_lead_m: float,
  ) -> np.ndarray:
    horizontal_offset = landed_position[0:2] - position[0:2]
    horizontal_distance = float(np.linalg.norm(horizontal_offset))
    if horizontal_distance > maximum_lead_m:
      horizontal_offset *= (
        maximum_lead_m / horizontal_distance
      )
    return position[0:2] + horizontal_offset

  @classmethod
  def estimated_descent_time_seconds(cls, height_m: float) -> float:
    """Numerically integrate dh/v(h) through the nonzero speed bands."""

    height = max(float(height_m), 0.0)
    if height <= 0.0:
      return 0.0
    sample_count = max(32, int(math.ceil(height * 2.0)) + 1)
    samples = np.linspace(0.0, height, sample_count)
    inverse_rates = np.array(
      [1.0 / cls.descent_rate_for_height_mps(value) for value in samples]
    )
    intervals = np.diff(samples)
    return float(
      np.sum(0.5 * intervals * (inverse_rates[:-1] + inverse_rates[1:]))
    )

  def estimated_landing_fuel_kg(self) -> float:
    """Estimate remaining propellant required to align, descend, and brake."""

    position = self.center_of_mass_position_world()
    velocity = self.center_of_mass_velocity_world()
    landed_com_altitude = float(self.landing_center_of_mass_position()[2])
    staging_altitude = self._landing_staging_altitude_for_current_state(
      fuel_takeover=True
    )
    staging_height = max(staging_altitude - landed_com_altitude, 0.0)
    horizontal_error = float(np.linalg.norm(position[0:2]))
    horizontal_speed = float(np.linalg.norm(velocity[0:2]))
    mass = self.controller.wet_mass_kg
    gravity = abs(float(self.model.opt.gravity[2]))
    minimum_thrust_acceleration = (
      self.controller.limits.min_thrust_newtons / mass
    )
    upward_braking_acceleration = max(
      gravity - minimum_thrust_acceleration, 1e-6
    )
    upward_speed = max(float(velocity[2]), 0.0)
    upward_braking_time = upward_speed / upward_braking_acceleration
    align_time = max(
      horizontal_error / AUTO_LAND_ALIGNMENT_SPEED_MPS,
      horizontal_speed / AUTO_LAND_ALIGNMENT_BRAKING_ACCEL_MPS2,
      upward_braking_time,
    ) + AUTO_LAND_ALIGNMENT_SETTLE_TIME_S
    powered_descent_height = self.estimated_powered_descent_height_m(
      staging_height
    )
    coast_planned = powered_descent_height < staging_height
    descent_time = AUTO_LAND_DESCENT_TIME_MARGIN * (
      self.estimated_coast_burn_time_seconds(powered_descent_height)
      if coast_planned
      else self.estimated_descent_time_seconds(powered_descent_height)
    )
    desired_descent_speed = self.descent_rate_for_height_mps(staging_height)
    excess_descent_speed = max(-float(velocity[2]) - desired_descent_speed, 0.0)
    coast_braking_speed = (
      math.sqrt(
        max(
          2.0 * gravity * (staging_height - powered_descent_height),
          0.0,
        )
      )
      if coast_planned
      else 0.0
    )
    estimated_impulse = AUTO_LAND_CONTROL_IMPULSE_MARGIN * (
      mass * gravity * (align_time + descent_time)
      + mass
      * (horizontal_speed + excess_descent_speed + coast_braking_speed)
    )
    impulse_estimate = float(
      self.controller.limits.alpha_kg_per_newton_second * estimated_impulse
      + AUTO_LAND_FIXED_FUEL_RESERVE_KG
    )
    minimum_takeover_fuel = min(
      self.controller.initial_fuel_mass_kg,
      AUTO_LAND_MINIMUM_TAKEOVER_FUEL_KG
      + AUTO_LAND_HEIGHT_RESERVE_KG_PER_M
      * min(powered_descent_height, 40.0)
      + AUTO_LAND_HORIZONTAL_RESERVE_KG_PER_M * horizontal_error,
    )
    controllability_floor = minimum_takeover_fuel / AUTO_LAND_FUEL_MARGIN
    return max(impulse_estimate, controllability_floor)

  def landing_burn_altitude_m(self, downward_speed_mps: float) -> float:
    """Return the conservative AGL ignition gate for a relightable coast."""

    speed = max(float(downward_speed_mps), 0.0)
    gravity = abs(float(self.model.opt.gravity[2]))
    maximum_acceleration = (
      self.controller.limits.max_thrust_newtons
      / max(self.controller.wet_mass_kg, 1.0)
    )
    net_braking_acceleration = max(maximum_acceleration - gravity, 0.25)
    target_speed_squared = COAST_TARGET_TOUCHDOWN_SPEED_MPS ** 2
    stopping_distance = max(
      speed * speed - target_speed_squared,
      0.0,
    ) / (2.0 * net_braking_acceleration)
    ignition_delay_distance = (
      speed * COAST_IGNITION_DELAY_S
      + 0.5 * gravity * COAST_IGNITION_DELAY_S ** 2
    )
    return (
      COAST_STOPPING_DISTANCE_FACTOR * stopping_distance
      + ignition_delay_distance
      + COAST_FIXED_BURN_MARGIN_M
    )

  def estimated_powered_descent_height_m(self, staging_height_m: float) -> float:
    """Estimate where a from-rest ballistic coast reaches the ignition gate."""

    staging_height = max(float(staging_height_m), 0.0)
    if staging_height <= COAST_MIN_ENTRY_HEIGHT_M:
      return staging_height
    gravity = abs(float(self.model.opt.gravity[2]))
    low = 0.0
    high = staging_height
    for _ in range(40):
      candidate = 0.5 * (low + high)
      downward_speed = math.sqrt(
        max(2.0 * gravity * (staging_height - candidate), 0.0)
      )
      if candidate >= self.landing_burn_altitude_m(downward_speed):
        high = candidate
      else:
        low = candidate
    burn_height = high
    if staging_height - burn_height < COAST_MIN_ALTITUDE_SAVING_M:
      return staging_height
    return burn_height

  @classmethod
  def estimated_coast_burn_time_seconds(cls, height_m: float) -> float:
    """Integrate the energy-based post-relight descent-speed corridor."""

    height = max(float(height_m), 0.0)
    if height <= 0.0:
      return 0.0
    sample_count = max(64, int(math.ceil(height)) + 1)
    samples = np.linspace(0.0, height, sample_count)
    rates = []
    for sample in samples:
      scheduled_rate = cls.descent_rate_for_height_mps(sample)
      energy_rate = math.sqrt(
        COAST_TARGET_TOUCHDOWN_SPEED_MPS ** 2
        + 2.0
        * COAST_BURN_REFERENCE_DECELERATION_MPS2
        * max(sample - COAST_BURN_REFERENCE_TERMINAL_HEIGHT_M, 0.0)
      )
      rates.append(max(scheduled_rate, energy_rate))
    inverse_rates = 1.0 / np.asarray(rates, dtype=float)
    intervals = np.diff(samples)
    return float(
      np.sum(0.5 * intervals * (inverse_rates[:-1] + inverse_rates[1:]))
    )

  def _body_tilt_radians(self) -> float:
    rotation = self.data.xmat[self.rocket_body_id].reshape(3, 3)
    return math.acos(float(np.clip(rotation[2, 2], -1.0, 1.0)))

  def _can_begin_landing_coast(
    self,
    *,
    height_m: float,
    horizontal_error_m: float,
    horizontal_speed_mps: float,
  ) -> bool:
    if height_m < COAST_MIN_ENTRY_HEIGHT_M:
      return False
    burn_height = self.landing_burn_altitude_m(
      max(-float(self.center_of_mass_velocity_world()[2]), 0.0)
    )
    return (
      height_m - burn_height >= COAST_MIN_ALTITUDE_SAVING_M
      and horizontal_error_m < LANDING_DESCENT_CAPTURE_RADIUS_M
      and horizontal_speed_mps < LANDING_DESCENT_CAPTURE_HORIZONTAL_SPEED_MPS
      and self._body_tilt_radians() < COAST_ENTRY_MAX_TILT_RADIANS
      and float(np.linalg.norm(self.data.qvel[3:6]))
      < COAST_ENTRY_MAX_ANGULAR_RATE_RPS
    )

  def _coast_requires_early_ignition(
    self,
    *,
    height_m: float,
    horizontal_error_m: float,
    horizontal_speed_mps: float,
  ) -> bool:
    downward_speed = max(-float(self.center_of_mass_velocity_world()[2]), 0.0)
    return (
      height_m <= self.landing_burn_altitude_m(downward_speed)
      or horizontal_error_m > COAST_ABORT_MAX_HORIZONTAL_ERROR_M
      or horizontal_speed_mps > COAST_ABORT_MAX_HORIZONTAL_SPEED_MPS
      or self._body_tilt_radians() > COAST_ABORT_MAX_TILT_RADIANS
      or float(np.linalg.norm(self.data.qvel[3:6]))
      > COAST_ABORT_MAX_ANGULAR_RATE_RPS
    )

  def _ignite_landing_burn(self) -> bool:
    if not self.controller.relight(
      throttle=self.controller.limits.max_throttle
    ):
      return False
    self._invalidate_mpc_solution()
    self.hover_enabled = True
    self.hover_target_position = self.center_of_mass_position_world()
    self.hover_target_velocity = self.center_of_mass_velocity_world()
    self.controller.center_lateral()
    self.landing_burn_from_coast = True
    self.landing_phase = LandingPhase.DESCEND
    return True

  def fuel_takeover_threshold_kg(self) -> float:
    return min(
      self.controller.initial_fuel_mass_kg,
      AUTO_LAND_FUEL_MARGIN * self.estimated_landing_fuel_kg(),
    )

  def _check_fuel_reserve_takeover(self) -> None:
    if (
      self.data.time - self.last_fuel_takeover_check_time
      < AUTO_LAND_FUEL_CHECK_PERIOD_S
    ):
      return
    self.last_fuel_takeover_check_time = float(self.data.time)
    if self.controller.engine_state is EngineState.COAST and self.landing_active:
      return
    if self.controller.engine_state is not EngineState.LIT:
      self.last_estimated_landing_fuel_kg = 0.0
      return
    estimate = self.estimated_landing_fuel_kg()
    self.last_estimated_landing_fuel_kg = estimate
    if (
      self.fuel_takeover_triggered
      or self.landing_active
      or self.landing_phase in (LandingPhase.COMPLETE, LandingPhase.ABORTED)
    ):
      return
    height = float(self.data.qpos[2] - LANDING_PAD_POSITION[2])
    if height <= AUTO_LAND_TAKEOVER_MIN_HEIGHT_M:
      return
    takeover_threshold = min(
      self.controller.initial_fuel_mass_kg,
      AUTO_LAND_FUEL_MARGIN * estimate,
    )
    if self.controller.fuel_mass_kg <= takeover_threshold:
      self.start_landing(fuel_takeover=True)

  def _update_landing_guidance(self) -> None:
    if not self.landing_active:
      return
    expected_engine_state = (
      EngineState.COAST
      if self.landing_phase is LandingPhase.COAST
      else EngineState.LIT
    )
    if self.controller.engine_state is not expected_engine_state:
      self._invalidate_mpc_solution()
      self.hover_enabled = False
      self.hover_target_velocity[:] = 0.0
      self.landing_phase = LandingPhase.ABORTED
      return

    position = self.center_of_mass_position_world()
    velocity = self.center_of_mass_velocity_world()
    body_position = self.data.qpos[0:3]
    body_velocity = self.data.qvel[0:3]
    landed_com_position = self.landing_center_of_mass_position()
    horizontal_error = float(np.linalg.norm(position[0:2]))
    horizontal_speed = float(np.linalg.norm(velocity[0:2]))
    height = max(float(body_position[2] - LANDING_PAD_POSITION[2]), 0.0)
    horizontal_lead = self.landing_horizontal_lead_for_height_m(height)

    if self.landing_phase is LandingPhase.ALIGN:
      self.hover_target_velocity[:] = 0.0
      self.hover_target_position[0:2] = self._bounded_landing_horizontal_target(
        position,
        landed_com_position,
        horizontal_lead,
      )
      vertical_offset = float(self.landing_staging_altitude - position[2])
      self.hover_target_position[2] = position[2] + float(
        np.clip(
          vertical_offset,
          -LANDING_ALIGNMENT_VERTICAL_LEAD_M,
          LANDING_ALIGNMENT_VERTICAL_LEAD_M,
        )
      )
      aligned = (
        horizontal_error < LANDING_DESCENT_CAPTURE_RADIUS_M
        and horizontal_speed
        < LANDING_DESCENT_CAPTURE_HORIZONTAL_SPEED_MPS
        and abs(position[2] - self.landing_staging_altitude)
        < LANDING_DESCENT_CAPTURE_ALTITUDE_ERROR_M
        and abs(velocity[2]) < LANDING_DESCENT_CAPTURE_VERTICAL_SPEED_MPS
      )
      if aligned:
        if self._can_begin_landing_coast(
          height_m=height,
          horizontal_error_m=horizontal_error,
          horizontal_speed_mps=horizontal_speed,
        ) and self.controller.begin_coast():
          self._invalidate_mpc_solution()
          self.controller.center_lateral()
          self.hover_target_position = landed_com_position.copy()
          self.hover_target_position[2] += self.landing_burn_altitude_m(0.0)
          self.hover_target_velocity[:] = 0.0
          self.landing_phase = LandingPhase.COAST
        else:
          self.landing_phase = LandingPhase.DESCEND

    if self.landing_phase is LandingPhase.COAST:
      self.hover_target_position[0:2] = landed_com_position[0:2]
      self.hover_target_position[2] = (
        landed_com_position[2]
        + self.landing_burn_altitude_m(
          max(-float(velocity[2]), 0.0)
        )
      )
      self.hover_target_velocity[:] = velocity
      if self._coast_requires_early_ignition(
        height_m=height,
        horizontal_error_m=horizontal_error,
        horizontal_speed_mps=horizontal_speed,
      ):
        if not self._ignite_landing_burn():
          self.hover_enabled = False
          self.landing_phase = LandingPhase.ABORTED

    if self.landing_phase is LandingPhase.DESCEND:
      self.hover_target_position[0:2] = self._bounded_landing_horizontal_target(
        position,
        landed_com_position,
        horizontal_lead,
      )
      previous_target_z = float(self.hover_target_position[2])
      requested_rate = self._descent_rate_mps()
      next_target_z = max(
        landed_com_position[2],
        previous_target_z - requested_rate * self.model.opt.timestep,
      )
      self.hover_target_position[2] = next_target_z
      self.hover_target_velocity[:] = 0.0
      self.hover_target_velocity[2] = -max(
        (previous_target_z - next_target_z) / self.model.opt.timestep,
        0.0,
      )
      ready_for_cutoff = (
        height < (0.30 if self.fuel_takeover_active else 0.15)
        and -0.50 <= body_velocity[2] <= 0.15
        and horizontal_error < (1.00 if self.fuel_takeover_active else 0.50)
        and horizontal_speed < (0.60 if self.fuel_takeover_active else 0.30)
      )
      if ready_for_cutoff:
        self._invalidate_mpc_solution()
        self.controller.kill_engine()
        self.hover_enabled = False
        self.hover_target_velocity[:] = 0.0
        self.controller.center_lateral()
        self.landing_burn_from_coast = False
        self.landing_phase = LandingPhase.COMPLETE

  def move_hover_target(self, delta_world: np.ndarray) -> None:
    if self.hover_enabled and not self.landing_active:
      self.hover_target_position += np.asarray(delta_world, dtype=float)
      position = self.center_of_mass_position_world()
      horizontal_offset = self.hover_target_position[0:2] - position[0:2]
      horizontal_distance = float(np.linalg.norm(horizontal_offset))
      if horizontal_distance > HOVER_TARGET_HORIZONTAL_LEAD_M:
        self.hover_target_position[0:2] = (
          position[0:2]
          + horizontal_offset
          * HOVER_TARGET_HORIZONTAL_LEAD_M
          / horizontal_distance
        )
      self.hover_target_position[2] = position[2] + float(
        np.clip(
          self.hover_target_position[2] - position[2],
          -HOVER_TARGET_VERTICAL_LEAD_M,
          HOVER_TARGET_VERTICAL_LEAD_M,
        )
      )

  def _fallback_hover_guidance(
    self,
    *,
    target_position: np.ndarray | None = None,
    target_velocity: np.ndarray | None = None,
    feedforward_acceleration: np.ndarray | None = None,
    position_gain_scale: float = 1.0,
  ) -> None:
    """High-rate constrained trajectory tracking and MPC fallback control."""

    if not self.hover_enabled:
      return
    if self.controller.engine_state is not EngineState.LIT:
      self.disable_hover()
      return

    position = self.center_of_mass_position_world()
    velocity = self.center_of_mass_velocity_world()
    reference_position = (
      self.hover_target_position
      if target_position is None
      else np.asarray(target_position, dtype=float)
    )
    reference_velocity = (
      self.hover_target_velocity
      if target_velocity is None
      else np.asarray(target_velocity, dtype=float)
    )
    reference_acceleration = (
      np.zeros(3, dtype=float)
      if feedforward_acceleration is None
      else np.asarray(feedforward_acceleration, dtype=float)
    )
    position_error = reference_position - position
    velocity_error = reference_velocity - velocity
    desired_acceleration = (
      reference_acceleration
      + position_gain_scale * HOVER_POSITION_KP * position_error
      + HOVER_VELOCITY_KD * velocity_error
    )
    required_force = self.controller.wet_mass_kg * (
      desired_acceleration - self.model.opt.gravity
    )

    vertical_force = float(required_force[2])
    horizontal_force = required_force[0:2]
    horizontal_magnitude = float(np.linalg.norm(horizontal_force))
    max_angle = math.radians(self.controller.limits.pointing_half_angle_deg)

    if vertical_force <= 0.0:
      self.controller.center_lateral()
      required_magnitude = 0.0
    else:
      requested_angle = math.atan2(horizontal_magnitude, vertical_force)
      constrained_angle = min(requested_angle, max_angle)
      if horizontal_magnitude > 0.0 and max_angle > 0.0:
        self.controller.lateral_command[:] = (
          horizontal_force
          / horizontal_magnitude
          * (constrained_angle / max_angle)
        )
      else:
        self.controller.center_lateral()
      required_magnitude = vertical_force / max(math.cos(constrained_angle), 1e-6)

    self.controller.throttle = float(
      np.clip(
        required_magnitude / self.controller.limits.nominal_max_newtons,
        self.controller.limits.min_throttle,
        self.controller.limits.max_throttle,
      )
    )

  def _mpc_state(self) -> np.ndarray:
    state = np.zeros(14, dtype=float)
    state[MASS] = self.controller.wet_mass_kg
    state[POSITION] = self.center_of_mass_position_world()
    state[VELOCITY] = self.center_of_mass_velocity_world()
    state[QUATERNION] = normalize_quaternion(self.data.qpos[3:7])
    state[ANGULAR_VELOCITY] = self.data.qvel[3:6]
    return state

  def _mpc_target_state(self) -> np.ndarray:
    target = self._mpc_state()
    target[POSITION] = self.hover_target_position
    target[VELOCITY] = self.hover_target_velocity
    target[QUATERNION] = (1.0, 0.0, 0.0, 0.0)
    target[ANGULAR_VELOCITY] = 0.0
    return target

  def _current_actuator_control(self) -> np.ndarray:
    return np.array(
      [
        self.controller.thrust_magnitude_newtons(),
        self.engine_gimbal_radians[0],
        self.engine_gimbal_radians[1],
        self.roll_control_torque_nm,
      ],
      dtype=float,
    )

  @staticmethod
  def _solve_mpc_job(
    controller: SixDofMPC,
    generation: int,
    request_time: float,
    state: np.ndarray,
    target: np.ndarray,
    previous_control: np.ndarray,
    max_gimbal_radians: float,
    central_differences: bool,
  ) -> tuple[int, float, np.ndarray, np.ndarray, MPCResult]:
    if isinstance(controller, SixDofMPC):
      result = controller.solve(
        state,
        target,
        previous_control,
        max_gimbal_radians=max_gimbal_radians,
        central_differences=central_differences,
      )
    else:
      result = controller.solve(state, target, previous_control)
    return (
      generation,
      request_time,
      target[POSITION].copy(),
      target[VELOCITY].copy(),
      result,
    )

  def _sample_mpc_prediction(
    self,
    result: MPCResult,
    elapsed_seconds: float,
  ) -> tuple[np.ndarray, np.ndarray] | None:
    states = np.asarray(result.predicted_states, dtype=float)
    if states.ndim != 2 or states.shape[0] != 14 or states.shape[1] == 0:
      return None
    horizon_seconds = self._mpc_config.prediction_dt * (states.shape[1] - 1)
    if elapsed_seconds < 0.0 or elapsed_seconds > horizon_seconds:
      return None
    if states.shape[1] == 1:
      sampled = states[:, 0].copy()
      sampled[QUATERNION] = normalize_quaternion(sampled[QUATERNION])
      return sampled, np.zeros(3, dtype=float)
    floating_index = elapsed_seconds / self._mpc_config.prediction_dt
    lower_index = min(int(math.floor(floating_index)), states.shape[1] - 2)
    fraction = float(np.clip(floating_index - lower_index, 0.0, 1.0))
    sampled = (
      (1.0 - fraction) * states[:, lower_index]
      + fraction * states[:, lower_index + 1]
    )
    sampled[QUATERNION] = quaternion_slerp(
      states[QUATERNION, lower_index],
      states[QUATERNION, lower_index + 1],
      fraction,
    )
    acceleration = (
      states[VELOCITY, lower_index + 1]
      - states[VELOCITY, lower_index]
    ) / self._mpc_config.prediction_dt
    return sampled, acceleration

  def _async_prediction_matches_current_state(
    self, predicted_state: np.ndarray
  ) -> bool:
    current = self._mpc_state()
    position_mismatch = float(
      np.linalg.norm(current[POSITION] - predicted_state[POSITION])
    )
    velocity_mismatch = float(
      np.linalg.norm(current[VELOCITY] - predicted_state[VELOCITY])
    )
    quaternion_dot = abs(
      float(np.dot(current[QUATERNION], predicted_state[QUATERNION]))
    )
    attitude_mismatch = 2.0 * math.acos(
      float(np.clip(quaternion_dot, -1.0, 1.0))
    )
    angular_rate_mismatch = float(
      np.linalg.norm(
        current[ANGULAR_VELOCITY] - predicted_state[ANGULAR_VELOCITY]
      )
    )
    return (
      position_mismatch <= ASYNC_MPC_MAX_POSITION_MISMATCH_M
      and velocity_mismatch <= ASYNC_MPC_MAX_VELOCITY_MISMATCH_MPS
      and attitude_mismatch <= ASYNC_MPC_MAX_ATTITUDE_MISMATCH_RADIANS
      and angular_rate_mismatch <= ASYNC_MPC_MAX_ANGULAR_RATE_MISMATCH_RPS
    )

  def _reject_async_mpc_result(self, result: MPCResult, reason: str) -> None:
    self.last_mpc_result = result
    self._active_async_mpc_result = None
    self._active_async_mpc_request_time = -math.inf
    self._active_async_mpc_target_position[:] = 0.0
    self._active_async_mpc_target_velocity[:] = 0.0
    self.mpc_using_fallback = True
    self.async_mpc_rejection_reason = reason

  def _accept_async_mpc_result(
    self,
    result: MPCResult,
    request_time: float,
    requested_target_position: np.ndarray,
    requested_target_velocity: np.ndarray,
  ) -> None:
    if not result.success:
      failure_reason = result.status
      if (
        result.status in ("optimal", "optimal_inaccurate")
        and result.scaled_dynamics_defect
        > self._mpc_config.maximum_scaled_defect
      ):
        failure_reason = "dynamics_defect"
      self._reject_async_mpc_result(result, failure_reason)
      return
    age = float(self.data.time - request_time)
    if age > ASYNC_MPC_MAX_ACCEPT_AGE_S:
      self._reject_async_mpc_result(result, "stale")
      return
    sampled = self._sample_mpc_prediction(result, age)
    if sampled is None or not self._async_prediction_matches_current_state(
      sampled[0]
    ):
      self._reject_async_mpc_result(result, "state_mismatch")
      return
    target_shift = float(
      np.linalg.norm(
        self.hover_target_position
        - np.asarray(requested_target_position, dtype=float)
      )
    )
    maximum_target_shift = (
      ASYNC_MPC_MAX_LANDING_TARGET_SHIFT_M
      if self.landing_active
      else ASYNC_MPC_MAX_HOVER_TARGET_SHIFT_M
    )
    if target_shift > maximum_target_shift:
      self._reject_async_mpc_result(result, "target_shift")
      return
    self.last_mpc_result = result
    self._active_async_mpc_result = result
    self._active_async_mpc_request_time = request_time
    self._active_async_mpc_target_position[:] = requested_target_position
    self._active_async_mpc_target_velocity[:] = requested_target_velocity
    self.mpc_using_fallback = False
    self.async_mpc_rejection_reason = ""

  def _track_async_mpc_trajectory(self) -> bool:
    result = self._active_async_mpc_result
    if result is None:
      return False
    elapsed = float(self.data.time - self._active_async_mpc_request_time)
    sampled = self._sample_mpc_prediction(result, elapsed)
    if sampled is None:
      self._reject_async_mpc_result(result, "trajectory_expired")
      return False
    current_predicted_state = sampled[0]
    if not self._async_prediction_matches_current_state(
      current_predicted_state
    ):
      self._reject_async_mpc_result(result, "trajectory_mismatch")
      return False
    tracking_lookahead = (
      ASYNC_MPC_LANDING_TRACKING_LOOKAHEAD_S
      if self.landing_active
      else ASYNC_MPC_HOVER_TRACKING_LOOKAHEAD_S
    )
    tracking_sample = self._sample_mpc_prediction(
      result,
      elapsed + tracking_lookahead,
    )
    if tracking_sample is None:
      tracking_sample = sampled
    predicted_state, feedforward_acceleration = tracking_sample
    target_position_delta = (
      self.hover_target_position - self._active_async_mpc_target_position
    )
    target_velocity_delta = (
      self.hover_target_velocity - self._active_async_mpc_target_velocity
    )
    maximum_target_shift = (
      ASYNC_MPC_MAX_LANDING_TARGET_SHIFT_M
      if self.landing_active
      else ASYNC_MPC_MAX_HOVER_TARGET_SHIFT_M
    )
    if np.linalg.norm(target_position_delta) > maximum_target_shift:
      self._reject_async_mpc_result(result, "target_shift")
      return False
    reference_position = predicted_state[POSITION] + target_position_delta
    target_blend = (
      ASYNC_MPC_LANDING_TARGET_BLEND
      if self.landing_active
      else ASYNC_MPC_HOVER_TARGET_BLEND
    )
    reference_position += target_blend * (
      self.hover_target_position - reference_position
    )
    reference_position[2] = max(
      reference_position[2], self._mpc_config.minimum_com_height_m
    )
    reference_velocity = predicted_state[VELOCITY] + target_velocity_delta
    velocity_blend = (
      ASYNC_MPC_LANDING_VELOCITY_BLEND
      if self.landing_active
      else ASYNC_MPC_HOVER_VELOCITY_BLEND
    )
    reference_velocity += velocity_blend * (
      self.hover_target_velocity - reference_velocity
    )
    feedforward_acceleration = self._limit_vector_magnitude(
      feedforward_acceleration,
      ASYNC_MPC_MAX_FEEDFORWARD_ACCELERATION_MPS2,
    )
    # Early horizontal acceleration includes counter-gimbal force used to tilt
    # the long stage. The inner loop derives horizontal acceleration from
    # position and measured velocity instead of treating that force as a
    # desired body direction. Vertical feed-forward does not have this
    # non-minimum-phase ambiguity.
    feedforward_acceleration[0:2] = 0.0
    self._fallback_hover_guidance(
      target_position=reference_position,
      target_velocity=reference_velocity,
      feedforward_acceleration=feedforward_acceleration,
      position_gain_scale=(
        ASYNC_MPC_LANDING_POSITION_GAIN_SCALE
        if self.landing_active
        else ASYNC_MPC_HOVER_POSITION_GAIN_SCALE
      ),
    )
    self._allocate_attitude_control(self.controller.thrust_direction_world())
    self.mpc_using_fallback = False
    return True

  def _set_lateral_indicator_from_world_direction(
    self, direction_world: np.ndarray
  ) -> None:
    direction = np.asarray(direction_world, dtype=float)
    horizontal = direction[0:2]
    horizontal_magnitude = float(np.linalg.norm(horizontal))
    if horizontal_magnitude < 1e-9:
      self.controller.center_lateral()
      return
    max_angle = self._active_gimbal_limit_radians()
    angle = math.atan2(horizontal_magnitude, max(float(direction[2]), 1e-9))
    self.controller.lateral_command[:] = (
      horizontal
      / horizontal_magnitude
      * min(angle / max(max_angle, 1e-9), 1.0)
    )

  def _apply_mpc_result(self, result: MPCResult) -> None:
    self.last_mpc_result = result
    if not result.success:
      self.mpc_using_fallback = True
      return
    command = result.control
    self.controller.throttle = float(
      np.clip(
        command[THRUST] / self.controller.limits.nominal_max_newtons,
        self.controller.limits.min_throttle,
        self.controller.limits.max_throttle,
      )
    )
    gimbal = np.asarray(command[GIMBAL], dtype=float).copy()
    max_gimbal = self._active_gimbal_limit_radians()
    magnitude = float(np.linalg.norm(gimbal))
    if magnitude > max_gimbal:
      gimbal *= max_gimbal / magnitude
    self.engine_gimbal_command_radians[:] = gimbal
    self.roll_control_torque_command_nm = float(
      np.clip(
        command[ROLL_TORQUE],
        -MAX_ROLL_CONTROL_TORQUE_NM,
        MAX_ROLL_CONTROL_TORQUE_NM,
      )
    )
    rotation = self.data.xmat[self.rocket_body_id].reshape(3, 3)
    self._set_lateral_indicator_from_world_direction(
      rotation @ gimbal_direction_body(self.engine_gimbal_command_radians)
    )
    self.mpc_using_fallback = False

  def _poll_mpc_result(self) -> None:
    if self._mpc_future is None or not self._mpc_future.done():
      return
    metadata = self._mpc_future_metadata
    try:
      (
        generation,
        request_time,
        requested_target_position,
        requested_target_velocity,
        result,
      ) = self._mpc_future.result()
    except Exception:
      if metadata is None:
        generation = self._mpc_generation
        request_time = float(self.last_mpc_request_time)
        requested_target_position = self.hover_target_position.copy()
        requested_target_velocity = self.hover_target_velocity.copy()
      else:
        (
          generation,
          request_time,
          requested_target_position,
          requested_target_velocity,
        ) = metadata
      result = MPCResult(
        success=False,
        control=self._current_actuator_control(),
        predicted_states=np.empty((14, 0)),
        status="worker_error",
        solve_time_seconds=0.0,
        iterations=0,
        scaled_dynamics_defect=math.inf,
        scaled_virtual_control=math.inf,
      )
    self._mpc_future = None
    self._mpc_future_metadata = None
    if generation != self._mpc_generation:
      if self.mpc is not None:
        self.mpc.reset()
      return
    if self.asynchronous_mpc:
      self._accept_async_mpc_result(
        result,
        request_time,
        requested_target_position,
        requested_target_velocity,
      )
    else:
      self._apply_mpc_result(result)

  def _request_mpc_solution(self) -> None:
    if self.mpc is None:
      return
    state = self._mpc_state()
    target = self._mpc_target_state()
    previous = self._current_actuator_control()
    max_gimbal_radians = self._mpc_gimbal_limit_radians()
    central_differences = self.landing_active
    self.last_mpc_request_time = float(self.data.time)
    if self._mpc_executor is None:
      if isinstance(self.mpc, SixDofMPC):
        result = self.mpc.solve(
          state,
          target,
          previous,
          max_gimbal_radians=max_gimbal_radians,
          central_differences=central_differences,
        )
      else:
        result = self.mpc.solve(state, target, previous)
      self._apply_mpc_result(result)
      return
    self._mpc_future = self._mpc_executor.submit(
      self._solve_mpc_job,
      self.mpc,
      self._mpc_generation,
      self.last_mpc_request_time,
      state,
      target,
      previous,
      max_gimbal_radians,
      central_differences,
    )
    self._mpc_future_metadata = (
      self._mpc_generation,
      self.last_mpc_request_time,
      target[POSITION].copy(),
      target[VELOCITY].copy(),
    )

  def _update_hover_controller(self, *, force: bool = False) -> None:
    if not self.hover_enabled:
      return
    if self.controller.engine_state is not EngineState.LIT:
      self.disable_hover()
      return
    self._poll_mpc_result()
    if self.terminal_controller_active:
      self.mpc_using_fallback = True
      self._fallback_hover_guidance()
      self._allocate_attitude_control(self.controller.thrust_direction_world())
      return
    due = (
      force
      or self.data.time - self.last_mpc_request_time >= MPC_UPDATE_PERIOD_S
    )
    if (
      self.enable_mpc
      and due
      and self._mpc_future is None
    ):
      self._request_mpc_solution()
    if self.asynchronous_mpc:
      if not self._track_async_mpc_trajectory():
        self._fallback_hover_guidance()
        self._allocate_attitude_control(
          self.controller.thrust_direction_world()
        )
      return
    if not self.enable_mpc or self.mpc_using_fallback:
      self._fallback_hover_guidance()
      self._allocate_attitude_control(self.controller.thrust_direction_world())

  @staticmethod
  def _desired_attitude_rotation(desired_up: np.ndarray) -> np.ndarray:
    body_z = np.asarray(desired_up, dtype=float)
    body_z /= max(float(np.linalg.norm(body_z)), 1e-12)
    heading_reference = np.array([1.0, 0.0, 0.0], dtype=float)
    body_x = heading_reference - np.dot(heading_reference, body_z) * body_z
    if np.linalg.norm(body_x) < 1e-6:
      heading_reference = np.array([0.0, 1.0, 0.0], dtype=float)
      body_x = heading_reference - np.dot(heading_reference, body_z) * body_z
    body_x /= np.linalg.norm(body_x)
    body_y = np.cross(body_z, body_x)
    body_y /= np.linalg.norm(body_y)
    body_x = np.cross(body_y, body_z)
    return np.column_stack((body_x, body_y, body_z))

  def _attitude_control_torque_body(self, desired_up: np.ndarray) -> np.ndarray:
    rotation = self.data.xmat[self.rocket_body_id].reshape(3, 3)
    desired_rotation = self._desired_attitude_rotation(desired_up)
    error_matrix = (
      desired_rotation.T @ rotation - rotation.T @ desired_rotation
    )
    attitude_error = 0.5 * np.array(
      [error_matrix[2, 1], error_matrix[0, 2], error_matrix[1, 0]],
      dtype=float,
    )
    angular_velocity_body = self.data.qvel[3:6]
    return (
      -ATTITUDE_KP_BODY * attitude_error
      - ATTITUDE_KD_BODY * angular_velocity_body
    )

  def _allocate_attitude_control(self, desired_up: np.ndarray) -> None:
    if self.controller.engine_state is not EngineState.LIT:
      self.engine_gimbal_command_radians[:] = 0.0
      self.roll_control_torque_command_nm = 0.0
      return
    desired_torque = self._attitude_control_torque_body(desired_up)
    thrust = self.controller.thrust_magnitude_newtons()
    lever_arm = abs(
      float(
        ENGINE_POSITION_BODY[2]
        - self.current_mass_properties().center_of_mass_body_m[2]
      )
    )
    requested_lateral_force = np.array(
      [-desired_torque[1] / lever_arm, desired_torque[0] / lever_arm],
      dtype=float,
    )
    max_angle = self._active_gimbal_limit_radians()
    max_lateral_force = thrust * math.sin(max_angle)
    requested_magnitude = float(np.linalg.norm(requested_lateral_force))
    if requested_magnitude > max_lateral_force > 0.0:
      requested_lateral_force *= max_lateral_force / requested_magnitude
      requested_magnitude = max_lateral_force
    if thrust <= 0.0 or requested_magnitude < 1e-9:
      self.engine_gimbal_command_radians[:] = 0.0
    else:
      angle = math.asin(min(requested_magnitude / thrust, math.sin(max_angle)))
      self.engine_gimbal_command_radians[:] = (
        requested_lateral_force / requested_magnitude * angle
      )
    self.roll_control_torque_command_nm = float(
      np.clip(
        desired_torque[2],
        -MAX_ROLL_CONTROL_TORQUE_NM,
        MAX_ROLL_CONTROL_TORQUE_NM,
      )
    )

  def _active_gimbal_limit_radians(self) -> float:
    mechanical_limit = math.radians(
      self.controller.limits.pointing_half_angle_deg
    )
    if not self.hover_enabled:
      return mechanical_limit
    landing_limit = min(mechanical_limit, LANDING_GIMBAL_LIMIT_RADIANS)
    if not self.landing_active:
      return landing_limit
    if self.landing_phase is not LandingPhase.DESCEND:
      return landing_limit
    height = float(self.data.qpos[2] - LANDING_PAD_POSITION[2])
    if height <= 1.0:
      return math.radians(0.75)
    if height <= 2.5:
      return math.radians(1.5)
    if height <= 5.0:
      return math.radians(3.0)
    return landing_limit

  def _mpc_gimbal_limit_radians(self) -> float:
    active_limit = self._active_gimbal_limit_radians()
    if self.landing_active:
      return active_limit
    return min(active_limit, HOVER_MPC_GIMBAL_LIMIT_RADIANS)

  @staticmethod
  def _limit_vector_magnitude(vector: np.ndarray, limit: float) -> np.ndarray:
    result = np.asarray(vector, dtype=float).copy()
    magnitude = float(np.linalg.norm(result))
    if magnitude > limit > 0.0:
      result *= limit / magnitude
    return result

  def _update_engine_gimbal_actuator(self) -> None:
    """Track the commanded gimbal with lag and terminal authority limits."""

    target = (
      self.engine_gimbal_command_radians
      if self.controller.engine_state is EngineState.LIT
      else np.zeros(2, dtype=float)
    )
    limit = self._active_gimbal_limit_radians()
    target = self._limit_vector_magnitude(target, limit)
    terminal = (
      self.landing_phase is LandingPhase.DESCEND
      and float(self.data.qpos[2] - LANDING_PAD_POSITION[2]) <= 5.0
    )
    if terminal and np.linalg.norm(target) < TERMINAL_GIMBAL_DEADBAND_RADIANS:
      target[:] = 0.0
    time_constant = (
      TERMINAL_GIMBAL_TIME_CONSTANT_S
      if terminal
      else GIMBAL_TIME_CONSTANT_S
    )
    blend = 1.0 - math.exp(-self.model.opt.timestep / time_constant)
    self.engine_gimbal_radians += blend * (
      target - self.engine_gimbal_radians
    )
    self.engine_gimbal_radians[:] = self._limit_vector_magnitude(
      self.engine_gimbal_radians, limit
    )
    if np.linalg.norm(self.engine_gimbal_radians) < 1e-8:
      self.engine_gimbal_radians[:] = 0.0

  def _update_roll_actuator(self) -> None:
    """Apply a first-order valve/manifold response to the RCS command."""

    target = (
      self.roll_control_torque_command_nm
      if self.controller.engine_state in (EngineState.LIT, EngineState.COAST)
      else 0.0
    )
    blend = 1.0 - math.exp(
      -self.model.opt.timestep / ROLL_RCS_TIME_CONSTANT_S
    )
    self.roll_control_torque_nm += blend * (
      target - self.roll_control_torque_nm
    )
    if abs(self.roll_control_torque_nm) < 1e-6:
      self.roll_control_torque_nm = 0.0

  def roll_rcs_force_pair_body(self) -> tuple[np.ndarray, np.ndarray]:
    """Return opposed body-frame forces producing the current roll moment."""

    force = float(
      np.clip(
        self.roll_control_torque_nm / (2.0 * ROLL_RCS_LEVER_ARM_M),
        -ROLL_RCS_MAX_THRUSTER_FORCE_N,
        ROLL_RCS_MAX_THRUSTER_FORCE_N,
      )
    )
    return (
      np.array([0.0, force, 0.0], dtype=float),
      np.array([0.0, -force, 0.0], dtype=float),
    )

  def _apply_roll_rcs(self, rotation: np.ndarray) -> None:
    positive_force_body, negative_force_body = self.roll_rcs_force_pair_body()
    if abs(float(positive_force_body[1])) < 1e-9:
      return
    zero_torque = np.zeros(3, dtype=float)
    for site_id, force_body in (
      (self.roll_rcs_xp_site_id, positive_force_body),
      (self.roll_rcs_xm_site_id, negative_force_body),
    ):
      mujoco.mj_applyFT(
        self.model,
        self.data,
        rotation @ force_body,
        zero_torque,
        self.data.site_xpos[site_id],
        self.rocket_body_id,
        self.data.qfrc_applied,
      )

  def _apply_control(self) -> None:
    self.data.qfrc_applied[:] = 0.0
    self.data.xfrc_applied[self.rocket_body_id, :] = 0.0
    rotation = self.data.xmat[self.rocket_body_id].reshape(3, 3)
    if self.controller.engine_state is EngineState.LIT:
      thrust_body = (
        self.controller.thrust_magnitude_newtons()
        * gimbal_direction_body(self.engine_gimbal_radians)
      )
      mujoco.mj_applyFT(
        self.model,
        self.data,
        rotation @ thrust_body,
        np.zeros(3, dtype=float),
        self.data.site_xpos[self.thrust_site_id],
        self.rocket_body_id,
        self.data.qfrc_applied,
      )
    self._apply_roll_rcs(rotation)

  def thrust_arrow_world(
    self,
  ) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Return plume-facing arrow endpoints and normalized thrust magnitude.

    The arrow exits the engine bell like the visible plume. It therefore points
    along the exhaust direction; the force applied to the vehicle points in the
    opposite direction.
    """

    thrust = self.controller.thrust_magnitude_newtons()
    maximum = self.controller.limits.max_thrust_newtons
    if self.controller.engine_state is not EngineState.LIT or thrust <= 0.0:
      return None

    rotation = self.data.xmat[self.rocket_body_id].reshape(3, 3)
    force_direction_world = rotation @ gimbal_direction_body(
      self.engine_gimbal_radians
    )
    plume_direction_world = -force_direction_world
    magnitude_fraction = float(np.clip(thrust / maximum, 0.0, 1.0))
    origin = self.data.site_xpos[self.thrust_site_id].copy()
    tip = (
      origin
      + plume_direction_world
      * THRUST_ARROW_MAX_LENGTH_M
      * magnitude_fraction
    )
    return origin, tip, magnitude_fraction

  def _update_model_mass(self, *, force: bool = False) -> None:
    current_mass = self.controller.wet_mass_kg
    if not force and math.isclose(current_mass, self.last_applied_mass, abs_tol=1e-9):
      return
    if (
      not force
      and self.data.time - self.last_mass_update_time < MASS_UPDATE_PERIOD_S
    ):
      return

    mass_properties = self.mass_model.properties(current_mass)
    self.model.body_mass[self.rocket_body_id] = mass_properties.mass_kg
    self.model.body_ipos[self.rocket_body_id] = (
      mass_properties.center_of_mass_body_m
    )
    self.model.body_inertia[self.rocket_body_id] = (
      mass_properties.inertia_at_com_kgm2
    )

    # mj_setConst temporarily evaluates the model at qpos0 and leaves MjData
    # there. Preserve the live free-body state so a mass update cannot teleport
    # the rocket back to its initial pose.
    qpos = self.data.qpos.copy()
    qvel = self.data.qvel.copy()
    simulation_time = float(self.data.time)
    mujoco.mj_setConst(self.model, self.data)
    self.data.qpos[:] = qpos
    self.data.qvel[:] = qvel
    self.data.time = simulation_time
    mujoco.mj_forward(self.model, self.data)
    self.last_mass_update_time = self.data.time
    self.last_applied_mass = current_mass

  def _update_flame(self) -> None:
    lit = self.controller.engine_state is EngineState.LIT
    span = self.controller.limits.max_throttle - self.controller.limits.min_throttle
    normalized = (
      (self.controller.throttle - self.controller.limits.min_throttle) / span
      if span > 0.0
      else 0.0
    )
    half_length = 0.25 + 0.75 * normalized
    direction_body = gimbal_direction_body(self.engine_gimbal_radians)
    self.model.geom_size[self.flame_geom_id, 1] = half_length
    self.model.geom_pos[self.flame_geom_id, :] = (
      ENGINE_POSITION_BODY - direction_body * half_length
    )
    cross = np.cross(np.array([0.0, 0.0, 1.0]), direction_body)
    dot = float(np.clip(direction_body[2], -1.0, 1.0))
    scale = math.sqrt(max(2.0 * (1.0 + dot), 1e-12))
    self.model.geom_quat[self.flame_geom_id, :] = (
      scale / 2.0,
      cross[0] / scale,
      cross[1] / scale,
      cross[2] / scale,
    )
    self.model.geom_rgba[self.flame_geom_id, 3] = 0.58 if lit else 0.0

  def step(self) -> None:
    self._check_fuel_reserve_takeover()
    self._update_landing_guidance()
    self._update_landing_legs()
    if self.landing_phase is LandingPhase.COAST:
      self.engine_gimbal_command_radians[:] = 0.0
      self.controller.center_lateral()
      desired_torque = self._attitude_control_torque_body(
        np.array([0.0, 0.0, 1.0])
      )
      self.roll_control_torque_command_nm = float(
        np.clip(
          desired_torque[2],
          -MAX_ROLL_CONTROL_TORQUE_NM,
          MAX_ROLL_CONTROL_TORQUE_NM,
        )
      )
    elif self.hover_enabled:
      self._update_hover_controller()
    else:
      self._poll_mpc_result()
      self._allocate_attitude_control(self.controller.thrust_direction_world())
    self._update_engine_gimbal_actuator()
    self._update_roll_actuator()
    self.controller.consume_fuel(self.model.opt.timestep)
    self._update_model_mass()
    self._apply_control()
    self._update_flame()
    mujoco.mj_step(self.model, self.data)

  def telemetry_lines(self) -> tuple[str, ...]:
    position = self.data.qpos[0:3]
    velocity = self.data.qvel[0:3]
    if self.landing_phase is LandingPhase.ALIGN:
      mode_line = (
        "MODE AUTO LAND: FUEL RESERVE ALIGN"
        if self.fuel_takeover_active
        else "MODE AUTO LAND: ALIGN OVER PAD"
      )
    elif self.landing_phase is LandingPhase.COAST:
      coast_source = "FUEL RESERVE " if self.fuel_takeover_active else ""
      mode_line = (
        f"MODE AUTO LAND: {coast_source}COAST    "
        f"IGNITE BELOW {self.hover_target_position[2] - self.landing_center_of_mass_position()[2]:.1f} m"
      )
    elif self.landing_phase is LandingPhase.DESCEND:
      descent_source = "FUEL RESERVE " if self.fuel_takeover_active else ""
      mode_line = (
        f"MODE AUTO LAND: {descent_source}DESCEND    "
        f"TARGET Z {self.hover_target_position[2]:.1f} m    "
        f"TARGET VZ {self.hover_target_velocity[2]:.2f} m/s"
      )
    elif self.landing_phase is LandingPhase.COMPLETE:
      mode_line = "MODE LANDED"
    elif self.landing_phase is LandingPhase.ABORTED:
      mode_line = "MODE LANDING ABORTED"
    elif self.hover_enabled:
      mode_line = (
        "MODE HOVER HOLD    "
        f"TARGET [{self.hover_target_position[0]:.1f}, "
        f"{self.hover_target_position[1]:.1f}, "
        f"{self.hover_target_position[2]:.1f}] m"
      )
    else:
      mode_line = "MODE MANUAL"
    rotation = self.data.xmat[self.rocket_body_id].reshape(3, 3)
    body_tilt = math.degrees(
      math.acos(float(np.clip(rotation[2, 2], -1.0, 1.0)))
    )
    gimbal_angle = math.degrees(float(np.linalg.norm(self.engine_gimbal_radians)))
    if self.landing_phase is LandingPhase.COAST:
      control_line = "CTRL BALLISTIC COAST: ENGINE ARMED"
    elif self.hover_enabled and self.terminal_controller_active:
      control_line = "CTRL 6-DOF TERMINAL"
    elif self.hover_enabled and self.enable_mpc:
      if self.mpc_using_fallback:
        fallback_detail = (
          f": {self.async_mpc_rejection_reason.upper()}"
          if self.async_mpc_rejection_reason
          else ""
        )
        control_line = f"CTRL 6-DOF FALLBACK{fallback_detail}"
      elif self.last_mpc_result is None:
        timing = "ASYNC" if self.asynchronous_mpc else "SYNC"
        control_line = f"CTRL SCVX MPC {timing}: WARMING"
      else:
        timing = "ASYNC+INNER" if self.asynchronous_mpc else "SYNC"
        control_line = (
          f"CTRL SCVX MPC {timing}: "
          f"{self.last_mpc_result.status.upper()}  "
          f"{self.last_mpc_result.solve_time_seconds * 1000:.0f} ms"
        )
    elif self.hover_enabled:
      control_line = "CTRL 6-DOF FALLBACK"
    else:
      control_line = "CTRL 6-DOF TVC"
    if self.landing_leg_deployment >= 1.0 - 1e-9:
      leg_line = "LEGS DEPLOYED"
    elif self.landing_legs_deploy_commanded:
      leg_line = f"LEGS DEPLOYING {self.landing_leg_deployment * 100:.0f}%"
    else:
      leg_line = "LEGS STOWED"
    return (
      (
        f"ENGINE {self.controller.engine_state.name}    "
        f"THROTTLE {self.controller.throttle * 100:5.1f}%    "
        f"THRUST {self.controller.thrust_magnitude_newtons() / 1000:5.1f} kN"
      ),
      (
        f"GIMBAL {gimbal_angle:4.1f} deg    TILT {body_tilt:4.1f} deg    "
        f"FUEL {self.controller.fuel_mass_kg:7.1f} kg    "
        f"HEIGHT AGL {position[2] - ROCKET_LANDED_COM_Z_M:6.1f} m    "
        f"VZ {velocity[2]:6.1f} m/s"
      ),
      (
        f"{mode_line}    {control_line}    "
        f"RCS {self.roll_control_torque_nm / 1000:+5.1f} kN m    "
        f"LAND EST {self.last_estimated_landing_fuel_kg:.0f} kg    "
        f"{leg_line}"
      ),
      "3-D arrow: plume direction, vehicle thrust is opposite",
      "H hover | L auto-land | arrows altitude/throttle | WASD target/thrust | K kill | R reset",
    )


class RocketWindow:
  """Small GLFW-based MuJoCo viewer with a clickable engine-kill button."""

  @staticmethod
  def _fit_window_size_to_work_area(
    work_width: int, work_height: int
  ) -> tuple[int, int]:
    """Keep the initial window inside the monitor's usable desktop area."""

    available_width = max(int(work_width), 1)
    available_height = max(int(work_height), 1)
    width = min(
      WINDOW_WIDTH,
      available_width,
      max(640, int(0.92 * available_width)),
    )
    height = min(
      WINDOW_HEIGHT,
      available_height,
      max(520, int(0.90 * available_height)),
    )
    return width, height

  @classmethod
  def _initial_window_size(cls) -> tuple[int, int]:
    monitor = glfw.get_primary_monitor()
    if monitor is None:
      return WINDOW_WIDTH, WINDOW_HEIGHT
    _, _, work_width, work_height = glfw.get_monitor_workarea(monitor)
    return cls._fit_window_size_to_work_area(work_width, work_height)

  @staticmethod
  def _font_scale_for_display(
    window_width: int,
    window_height: int,
    framebuffer_width: int,
    framebuffer_height: int,
  ) -> int:
    """Choose MuJoCo glyph resolution matching logical-to-pixel scale."""

    scale_x = (
      framebuffer_width / window_width if window_width > 0 else 1.0
    )
    scale_y = (
      framebuffer_height / window_height if window_height > 0 else 1.0
    )
    display_scale = max(
      scale_x,
      scale_y,
      0.5,
    )
    requested_scale = 100.0 * display_scale
    available_scales = (50, 100, 150, 200, 250, 300)
    eligible = [
      scale for scale in available_scales if scale <= requested_scale + 1e-6
    ]
    return eligible[-1] if eligible else available_scales[0]

  @staticmethod
  def _text_width_pixels(text: str, character_widths) -> int:
    widths = character_widths
    if len(widths) == 0:
      return 0
    fallback_index = min(ord("?"), len(widths) - 1)
    fallback_width = int(widths[fallback_index])
    return sum(
      int(widths[code]) if code < len(widths) else fallback_width
      for code in (ord(character) for character in text)
    )

  @classmethod
  def _wrap_overlay_lines(
    cls,
    lines: tuple[str, ...] | list[str],
    maximum_width_pixels: int,
    character_widths,
  ) -> tuple[str, ...]:
    """Word-wrap overlay text using MuJoCo's actual rasterized glyph widths."""

    wrapped: list[str] = []
    maximum_width = max(int(maximum_width_pixels), 1)
    for line in lines:
      if cls._text_width_pixels(line, character_widths) <= maximum_width:
        wrapped.append(line)
        continue
      words = line.split()
      if not words:
        wrapped.append("")
        continue
      current = words[0]
      for word in words[1:]:
        candidate = f"{current} {word}"
        if (
          cls._text_width_pixels(candidate, character_widths)
          <= maximum_width
        ):
          current = candidate
        else:
          wrapped.append(current)
          current = word
      wrapped.append(current)
    return tuple(wrapped)

  def __init__(self, simulation: RocketSimulation) -> None:
    self.simulation = simulation
    if not glfw.init():
      raise RuntimeError("GLFW could not initialize a graphics context.")

    glfw.window_hint(glfw.SAMPLES, 4)
    window_width, window_height = self._initial_window_size()
    self.window = glfw.create_window(
      window_width, window_height, APP_TITLE, None, None
    )
    if self.window is None:
      glfw.terminate()
      raise RuntimeError("GLFW could not create the MuJoCo window.")

    glfw.make_context_current(self.window)
    glfw.swap_interval(1)

    self.camera = mujoco.MjvCamera()
    self.option = mujoco.MjvOption()
    self.scene = mujoco.MjvScene(self.simulation.model, maxgeom=10_000)
    window_width, window_height = glfw.get_window_size(self.window)
    framebuffer_width, framebuffer_height = glfw.get_framebuffer_size(
      self.window
    )
    self.font_scale = self._font_scale_for_display(
      window_width,
      window_height,
      framebuffer_width,
      framebuffer_height,
    )
    self.context = mujoco.MjrContext(
      self.simulation.model, self.font_scale
    )
    mujoco.mjv_defaultCamera(self.camera)
    mujoco.mjv_defaultOption(self.option)
    self.camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    self.camera.trackbodyid = self.simulation.rocket_body_id
    self.camera.distance = 72.0
    self.camera.azimuth = 135.0
    self.camera.elevation = -14.0

    self.mouse_left = False
    self.mouse_middle = False
    self.mouse_right = False
    self.last_cursor_x = 0.0
    self.last_cursor_y = 0.0
    self.mouse_direction_command = np.zeros(2, dtype=float)
    self.throttle_slider_dragging = False
    self.ui_pointer_captured = False
    self.previous_command_keys = {
      glfw.KEY_H: False,
      glfw.KEY_I: False,
      glfw.KEY_K: False,
      glfw.KEY_L: False,
      glfw.KEY_R: False,
    }
    self.status_message = "Press Up, I, or IGNITE ENGINE to start"
    self.status_until = time.monotonic() + 5.0

    glfw.set_key_callback(self.window, self._on_key)
    glfw.set_mouse_button_callback(self.window, self._on_mouse_button)
    glfw.set_cursor_pos_callback(self.window, self._on_cursor_move)
    glfw.set_scroll_callback(self.window, self._on_scroll)

  @staticmethod
  def _control_panel_rect_window(
    window_width: int,
  ) -> tuple[float, float]:
    panel_width = float(
      np.clip(
        GUI_PANEL_WIDTH_FRACTION * window_width,
        GUI_PANEL_MIN_WIDTH,
        GUI_PANEL_MAX_WIDTH,
      )
    )
    panel_width = min(panel_width, max(float(window_width) - 24.0, 1.0))
    margin = min(GUI_PANEL_MARGIN, max((window_width - panel_width) / 2.0, 0.0))
    return float(window_width) - margin - panel_width, panel_width

  @classmethod
  def _engine_button_rect_window(
    cls, window_width: int
  ) -> tuple[float, float, float, float]:
    panel_x, panel_width = cls._control_panel_rect_window(window_width)
    return (panel_x, 22.0, panel_width, 52.0)

  @classmethod
  def _hover_button_rect_window(
    cls, window_width: int
  ) -> tuple[float, float, float, float]:
    panel_x, panel_width = cls._control_panel_rect_window(window_width)
    return (panel_x, 84.0, panel_width, 46.0)

  @classmethod
  def _land_button_rect_window(
    cls, window_width: int
  ) -> tuple[float, float, float, float]:
    panel_x, panel_width = cls._control_panel_rect_window(window_width)
    return (panel_x, 140.0, panel_width, 46.0)

  @classmethod
  def _direction_button_rects_window(
    cls, window_width: int,
  ) -> dict[str, tuple[float, float, float, float]]:
    panel_x, panel_width = cls._control_panel_rect_window(window_width)
    button_width = min(52.0, max((panel_width - 16.0) / 3.0, 1.0))
    gap = 8.0
    row_width = 3.0 * button_width + 2.0 * gap
    row_x = panel_x + 0.5 * (panel_width - row_width)
    center_x = panel_x + 0.5 * (panel_width - button_width)
    return {
      "W": (center_x, 214.0, button_width, 42.0),
      "A": (row_x, 264.0, button_width, 42.0),
      "S": (row_x + button_width + gap, 264.0, button_width, 42.0),
      "D": (row_x + 2.0 * (button_width + gap), 264.0, button_width, 42.0),
    }

  @classmethod
  def _thrust_slider_rect_window(
    cls, window_width: int,
  ) -> tuple[float, float, float, float]:
    panel_x, panel_width = cls._control_panel_rect_window(window_width)
    return (panel_x, 354.0, panel_width, 20.0)

  @classmethod
  def _controller_indicator_rect_window(
    cls, window_width: int,
  ) -> tuple[float, float, float, float]:
    panel_x, panel_width = cls._control_panel_rect_window(window_width)
    return (panel_x, 404.0, panel_width, 42.0)

  def _controller_indicator_style(
    self,
  ) -> tuple[str, tuple[float, float, float, float]]:
    simulation = self.simulation
    if simulation.landing_phase is LandingPhase.COAST:
      return "COAST ACTIVE", (0.08, 0.42, 0.70, 0.96)
    if simulation.hover_enabled:
      if simulation.terminal_controller_active:
        return "TERMINAL ACTIVE", (0.20, 0.32, 0.68, 0.96)
      if (
        simulation.enable_mpc
        and not simulation.mpc_using_fallback
        and simulation.last_mpc_result is not None
        and simulation.last_mpc_result.success
      ):
        return "MPC ACTIVE", (0.02, 0.55, 0.38, 0.96)
      return "FALLBACK ACTIVE", (0.82, 0.38, 0.02, 0.96)
    return "MANUAL TVC", (0.18, 0.24, 0.31, 0.94)

  def _overlay_font_for_label(self, label: str, rectangle_width: int):
    """Use the large font only when its measured glyphs fit the rectangle."""

    available_width = max(int(rectangle_width) - 12, 1)
    if (
      self._text_width_pixels(label, self.context.charWidthBig)
      <= available_width
    ):
      return mujoco.mjtFont.mjFONT_BIG
    return mujoco.mjtFont.mjFONT_NORMAL

  @classmethod
  def _point_in_engine_button(
    cls, cursor_x: float, cursor_y: float, window_width: int
  ) -> bool:
    x, y, width, height = cls._engine_button_rect_window(window_width)
    return x <= cursor_x <= x + width and y <= cursor_y <= y + height

  @classmethod
  def _point_in_hover_button(
    cls, cursor_x: float, cursor_y: float, window_width: int
  ) -> bool:
    x, y, width, height = cls._hover_button_rect_window(window_width)
    return x <= cursor_x <= x + width and y <= cursor_y <= y + height

  @classmethod
  def _point_in_land_button(
    cls, cursor_x: float, cursor_y: float, window_width: int
  ) -> bool:
    x, y, width, height = cls._land_button_rect_window(window_width)
    return x <= cursor_x <= x + width and y <= cursor_y <= y + height

  @classmethod
  def _direction_command_for_point(
    cls, cursor_x: float, cursor_y: float, window_width: int
  ) -> np.ndarray | None:
    commands = {
      "W": np.array([0.0, 1.0]),
      "A": np.array([-1.0, 0.0]),
      "S": np.array([0.0, -1.0]),
      "D": np.array([1.0, 0.0]),
    }
    for name, (x, y, width, height) in cls._direction_button_rects_window(
      window_width
    ).items():
      if x <= cursor_x <= x + width and y <= cursor_y <= y + height:
        return commands[name].copy()
    return None

  @classmethod
  def _point_in_thrust_slider(
    cls, cursor_x: float, cursor_y: float, window_width: int
  ) -> bool:
    x, y, width, height = cls._thrust_slider_rect_window(window_width)
    return (
      x <= cursor_x <= x + width
      and y - 10.0 <= cursor_y <= y + height + 10.0
    )

  def _throttle_from_slider_x(self, cursor_x: float, window_width: int) -> float:
    x, _, width, _ = self._thrust_slider_rect_window(window_width)
    normalized = float(np.clip((cursor_x - x) / width, 0.0, 1.0))
    limits = self.simulation.controller.limits
    return limits.min_throttle + normalized * (
      limits.max_throttle - limits.min_throttle
    )

  def _thrust_display_values(self) -> tuple[float, float, str]:
    controller = self.simulation.controller
    if controller.engine_state is not EngineState.LIT:
      return 0.0, 0.0, "OFF"
    displayed_throttle = controller.throttle
    limits = controller.limits
    slider_fraction = float(
      np.clip(
        (displayed_throttle - limits.min_throttle)
        / (limits.max_throttle - limits.min_throttle),
        0.0,
        1.0,
      )
    )
    owner = (
      "AUTO"
      if self.simulation.hover_enabled or self.simulation.landing_active
      else "MANUAL"
    )
    return displayed_throttle, slider_fraction, owner

  @staticmethod
  def _direction_button_levels(command: np.ndarray) -> dict[str, float]:
    x_command, y_command = np.asarray(command, dtype=float)
    return {
      "W": max(float(y_command), 0.0),
      "A": max(float(-x_command), 0.0),
      "S": max(float(-y_command), 0.0),
      "D": max(float(x_command), 0.0),
    }

  def _set_status(self, message: str, duration: float = 2.0) -> None:
    self.status_message = message
    self.status_until = time.monotonic() + duration

  def _ignite(self) -> None:
    if self.simulation.controller.ignite():
      self._set_status("ENGINE IGNITED - minimum thrust is active")
    elif self.simulation.controller.engine_state is EngineState.SHUTDOWN:
      self._set_status("Engine was killed; press R before reigniting")
    elif self.simulation.controller.engine_state is EngineState.FUEL_OUT:
      self._set_status("Fuel depleted; press R to reset")

  def _kill_engine(self) -> None:
    if self.simulation.controller.kill_engine():
      self.simulation.disable_hover()
      self._set_status("ENGINE KILLED - thrust jumped directly to zero")
    else:
      self._set_status("Engine is already off")

  def _reset_flight(self) -> None:
    self.simulation.reset()
    self._set_status("FLIGHT RESET - engine off, fuel full", duration=3.0)

  def _toggle_hover(self) -> None:
    if self.simulation.hover_enabled:
      self.simulation.disable_hover()
      self._set_status("HOVER HOLD OFF - manual controls active")
    elif self.simulation.enable_hover():
      self._set_status("HOVER HOLD ON - current position captured", duration=3.0)
    elif self.simulation.controller.engine_state is EngineState.SHUTDOWN:
      self._set_status("Press R before enabling hover")
    else:
      self._set_status("Hover unavailable: engine has no fuel")

  def _toggle_landing(self) -> None:
    if self.simulation.landing_active:
      self.simulation.cancel_landing()
      self._set_status("AUTO LAND CANCELED - holding current position")
    elif self.simulation.start_landing():
      self._set_status("AUTO LAND ON - aligning over the pad", duration=3.0)
    elif self.simulation.controller.engine_state is EngineState.SHUTDOWN:
      self._set_status("Press R before starting another landing")
    else:
      self._set_status("Auto land unavailable: engine has no fuel")

  def _set_manual_throttle_from_slider(
    self, cursor_x: float, window_width: int
  ) -> bool:
    if self.simulation.landing_active or self.simulation.hover_enabled:
      self._set_status("Autopilot currently owns the thrust slider")
      return False
    if self.simulation.controller.engine_state in (
      EngineState.SHUTDOWN,
      EngineState.FUEL_OUT,
    ):
      self._set_status("Press R before commanding thrust")
      return False
    if self.simulation.controller.engine_state is EngineState.OFF:
      self._ignite()
    self.simulation.controller.throttle = self._throttle_from_slider_x(
      cursor_x, window_width
    )
    self._set_status(
      f"MANUAL THRUST {self.simulation.controller.throttle * 100:.1f}%",
      duration=1.0,
    )
    return True

  def _apply_throttle_input(self, throttle_axis: float, dt: float) -> None:
    if throttle_axis == 0.0:
      return
    if self.simulation.landing_active:
      return
    if self.simulation.hover_enabled:
      self.simulation.move_hover_target(
        np.array([0.0, 0.0, throttle_axis * HOVER_TARGET_SPEED_MPS * dt])
      )
      return
    if throttle_axis > 0.0 and self.simulation.controller.engine_state is EngineState.OFF:
      self._ignite()
    self.simulation.controller.change_throttle(
      throttle_axis * THROTTLE_RATE_PER_SECOND * dt
    )

  def _on_key(
    self,
    window,
    key: int,
    scancode: int,
    action: int,
    mods: int,
  ) -> None:
    del scancode, mods
    if action == glfw.PRESS and key == glfw.KEY_ESCAPE:
      glfw.set_window_should_close(window, True)
      return

    command_actions = {
      glfw.KEY_H: self._toggle_hover,
      glfw.KEY_I: self._ignite,
      glfw.KEY_K: self._kill_engine,
      glfw.KEY_L: self._toggle_landing,
      glfw.KEY_R: self._reset_flight,
    }
    if key not in command_actions:
      return
    if action == glfw.PRESS:
      command_actions[key]()
      self.previous_command_keys[key] = True
    elif action == glfw.RELEASE:
      self.previous_command_keys[key] = False

  def _on_mouse_button(self, window, button: int, action: int, mods: int) -> None:
    del mods
    self.mouse_left = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
    self.mouse_middle = (
      glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS
    )
    self.mouse_right = (
      glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
    )

    if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS:
      cursor_x, cursor_y = glfw.get_cursor_pos(window)
      window_width, _ = glfw.get_window_size(window)
      self.ui_pointer_captured = False
      if self._point_in_engine_button(cursor_x, cursor_y, window_width):
        self.ui_pointer_captured = True
        if self.simulation.controller.engine_state is EngineState.OFF:
          self._ignite()
        elif self.simulation.controller.engine_state in (
          EngineState.LIT,
          EngineState.COAST,
        ):
          self._kill_engine()
        else:
          self._set_status("Press R to start a new flight")
      elif self._point_in_hover_button(cursor_x, cursor_y, window_width):
        self.ui_pointer_captured = True
        self._toggle_hover()
      elif self._point_in_land_button(cursor_x, cursor_y, window_width):
        self.ui_pointer_captured = True
        self._toggle_landing()
      else:
        direction_command = self._direction_command_for_point(
          cursor_x, cursor_y, window_width
        )
        if direction_command is not None:
          self.ui_pointer_captured = True
          self.mouse_direction_command[:] = direction_command
        elif self._point_in_thrust_slider(cursor_x, cursor_y, window_width):
          self.ui_pointer_captured = True
          self.throttle_slider_dragging = self._set_manual_throttle_from_slider(
            cursor_x, window_width
          )

    if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.RELEASE:
      self.mouse_direction_command[:] = 0.0
      self.throttle_slider_dragging = False
      self.ui_pointer_captured = False

    self.last_cursor_x, self.last_cursor_y = glfw.get_cursor_pos(window)

  def _on_cursor_move(self, window, xpos: float, ypos: float) -> None:
    if self.throttle_slider_dragging:
      window_width, _ = glfw.get_window_size(window)
      self._set_manual_throttle_from_slider(xpos, window_width)
      self.last_cursor_x, self.last_cursor_y = xpos, ypos
      return
    if self.ui_pointer_captured:
      self.last_cursor_x, self.last_cursor_y = xpos, ypos
      return
    if not (self.mouse_left or self.mouse_middle or self.mouse_right):
      self.last_cursor_x, self.last_cursor_y = xpos, ypos
      return

    dx = xpos - self.last_cursor_x
    dy = ypos - self.last_cursor_y
    self.last_cursor_x, self.last_cursor_y = xpos, ypos
    _, height = glfw.get_window_size(window)
    if height <= 0:
      return

    shift_down = (
      glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
      or glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
    )
    if self.mouse_right:
      action = (
        mujoco.mjtMouse.mjMOUSE_MOVE_H
        if shift_down
        else mujoco.mjtMouse.mjMOUSE_MOVE_V
      )
    elif self.mouse_left:
      action = (
        mujoco.mjtMouse.mjMOUSE_ROTATE_H
        if shift_down
        else mujoco.mjtMouse.mjMOUSE_ROTATE_V
      )
    else:
      action = mujoco.mjtMouse.mjMOUSE_ZOOM

    mujoco.mjv_moveCamera(
      self.simulation.model,
      action,
      dx / height,
      dy / height,
      self.scene,
      self.camera,
    )

  def _on_scroll(self, window, xoffset: float, yoffset: float) -> None:
    del window, xoffset
    mujoco.mjv_moveCamera(
      self.simulation.model,
      mujoco.mjtMouse.mjMOUSE_ZOOM,
      0.0,
      -0.05 * yoffset,
      self.scene,
      self.camera,
    )

  def _read_flight_controls(self, dt: float) -> None:
    command_actions = {
      glfw.KEY_H: self._toggle_hover,
      glfw.KEY_I: self._ignite,
      glfw.KEY_K: self._kill_engine,
      glfw.KEY_L: self._toggle_landing,
      glfw.KEY_R: self._reset_flight,
    }
    for key, command in command_actions.items():
      is_down = glfw.get_key(self.window, key) == glfw.PRESS
      if is_down and not self.previous_command_keys[key]:
        command()
      self.previous_command_keys[key] = is_down

    lateral_x = float(
      (glfw.get_key(self.window, glfw.KEY_D) == glfw.PRESS)
      - (glfw.get_key(self.window, glfw.KEY_A) == glfw.PRESS)
    ) + self.mouse_direction_command[0]
    lateral_y = float(
      (glfw.get_key(self.window, glfw.KEY_W) == glfw.PRESS)
      - (glfw.get_key(self.window, glfw.KEY_S) == glfw.PRESS)
    ) + self.mouse_direction_command[1]
    lateral_target = np.array([lateral_x, lateral_y], dtype=float)
    target_norm = float(np.linalg.norm(lateral_target))
    if target_norm > 1.0:
      lateral_target /= target_norm

    if self.simulation.landing_active:
      pass
    elif self.simulation.hover_enabled:
      self.simulation.move_hover_target(
        np.array(
          [
            lateral_target[0] * HOVER_TARGET_SPEED_MPS * dt,
            lateral_target[1] * HOVER_TARGET_SPEED_MPS * dt,
            0.0,
          ]
        )
      )
    else:
      lateral_delta = lateral_target - self.simulation.controller.lateral_command
      delta_norm = float(np.linalg.norm(lateral_delta))
      max_delta = LATERAL_SLEW_RATE_PER_SECOND * dt
      if delta_norm > max_delta > 0.0:
        lateral_delta *= max_delta / delta_norm
      self.simulation.controller.lateral_command += lateral_delta

    throttle_axis = float(
      (glfw.get_key(self.window, glfw.KEY_UP) == glfw.PRESS)
      - (glfw.get_key(self.window, glfw.KEY_DOWN) == glfw.PRESS)
    )
    self._apply_throttle_input(throttle_axis, dt)

  def _draw_button(self, framebuffer_width: int, framebuffer_height: int) -> None:
    window_width, window_height = glfw.get_window_size(self.window)
    if window_width <= 0 or window_height <= 0:
      return
    scale_x = framebuffer_width / window_width
    scale_y = framebuffer_height / window_height
    x, y, width, height = self._engine_button_rect_window(window_width)
    rect = mujoco.MjrRect(
      int(x * scale_x),
      int((window_height - y - height) * scale_y),
      int(width * scale_x),
      int(height * scale_y),
    )
    state = self.simulation.controller.engine_state
    if state is EngineState.OFF:
      color = (0.05, 0.48, 0.20, 0.95)
      label = "IGNITE ENGINE"
    elif state is EngineState.LIT:
      color = (0.72, 0.06, 0.04, 0.95)
      label = "KILL ENGINE"
    elif state is EngineState.COAST:
      color = (0.72, 0.34, 0.02, 0.95)
      label = "COAST - KILL"
    elif state is EngineState.SHUTDOWN:
      color = (0.18, 0.19, 0.21, 0.90)
      label = "ENGINE KILLED"
    else:
      color = (0.18, 0.19, 0.21, 0.90)
      label = "FUEL DEPLETED"
    mujoco.mjr_rectangle(rect, *color)
    mujoco.mjr_overlay(
      self._overlay_font_for_label(label, rect.width),
      mujoco.mjtGridPos.mjGRID_TOPLEFT,
      rect,
      label,
      "",
      self.context,
    )

    land_x, land_y, land_width, land_height = self._land_button_rect_window(
      window_width
    )
    land_rect = mujoco.MjrRect(
      int(land_x * scale_x),
      int((window_height - land_y - land_height) * scale_y),
      int(land_width * scale_x),
      int(land_height * scale_y),
    )
    if self.simulation.landing_phase is LandingPhase.ALIGN:
      land_color = (0.78, 0.38, 0.02, 0.96)
      land_label = (
        "FUEL AUTO: ALIGN"
        if self.simulation.fuel_takeover_active
        else "LAND: ALIGN"
      )
    elif self.simulation.landing_phase is LandingPhase.COAST:
      land_color = (0.12, 0.42, 0.68, 0.96)
      land_label = (
        "FUEL AUTO: COAST"
        if self.simulation.fuel_takeover_active
        else "LAND: COAST"
      )
    elif self.simulation.landing_phase is LandingPhase.DESCEND:
      land_color = (0.78, 0.38, 0.02, 0.96)
      land_label = (
        "FUEL AUTO: LAND"
        if self.simulation.fuel_takeover_active
        else "LAND: DESCEND"
      )
    elif self.simulation.landing_phase is LandingPhase.COMPLETE:
      land_color = (0.08, 0.48, 0.18, 0.94)
      land_label = "LANDED"
    else:
      land_color = (0.40, 0.26, 0.08, 0.92)
      land_label = "AUTO LAND"
    mujoco.mjr_rectangle(land_rect, *land_color)
    mujoco.mjr_overlay(
      self._overlay_font_for_label(land_label, land_rect.width),
      mujoco.mjtGridPos.mjGRID_TOPLEFT,
      land_rect,
      land_label,
      "",
      self.context,
    )

    hover_x, hover_y, hover_width, hover_height = self._hover_button_rect_window(
      window_width
    )
    hover_rect = mujoco.MjrRect(
      int(hover_x * scale_x),
      int((window_height - hover_y - hover_height) * scale_y),
      int(hover_width * scale_x),
      int(hover_height * scale_y),
    )
    if self.simulation.hover_enabled:
      hover_color = (0.02, 0.42, 0.70, 0.95)
      hover_label = "HOVER ACTIVE"
    else:
      hover_color = (0.15, 0.25, 0.34, 0.92)
      hover_label = "HOVER HOLD"
    mujoco.mjr_rectangle(hover_rect, *hover_color)
    mujoco.mjr_overlay(
      self._overlay_font_for_label(hover_label, hover_rect.width),
      mujoco.mjtGridPos.mjGRID_TOPLEFT,
      hover_rect,
      hover_label,
      "",
      self.context,
    )

    direction_levels = self._direction_button_levels(
      self.simulation.controller.lateral_command
    )
    for direction, (button_x, button_y, button_width, button_height) in (
      self._direction_button_rects_window(window_width).items()
    ):
      button_rect = mujoco.MjrRect(
        int(button_x * scale_x),
        int((window_height - button_y - button_height) * scale_y),
        int(button_width * scale_x),
        int(button_height * scale_y),
      )
      level = float(np.clip(direction_levels[direction], 0.0, 1.0))
      button_color = (
        0.13 - 0.08 * level,
        0.18 + 0.30 * level,
        0.24 + 0.55 * level,
        0.94,
      )
      mujoco.mjr_rectangle(button_rect, *button_color)
      mujoco.mjr_overlay(
        mujoco.mjtFont.mjFONT_BIG,
        mujoco.mjtGridPos.mjGRID_TOPLEFT,
        button_rect,
        direction,
        "",
        self.context,
      )

    slider_x, slider_y, slider_width, slider_height = (
      self._thrust_slider_rect_window(window_width)
    )
    slider_rect = mujoco.MjrRect(
      int(slider_x * scale_x),
      int((window_height - slider_y - slider_height) * scale_y),
      int(slider_width * scale_x),
      int(slider_height * scale_y),
    )
    displayed_throttle, slider_fraction, slider_owner = (
      self._thrust_display_values()
    )
    mujoco.mjr_rectangle(slider_rect, 0.10, 0.12, 0.15, 0.96)
    fill_rect = mujoco.MjrRect(
      slider_rect.left,
      slider_rect.bottom,
      max(1, int(slider_rect.width * slider_fraction)),
      slider_rect.height,
    )
    autopilot_owns_thrust = (
      self.simulation.hover_enabled or self.simulation.landing_active
    )
    fill_color = (
      (0.04, 0.48, 0.78, 0.96)
      if autopilot_owns_thrust
      else (0.95, 0.48, 0.05, 0.96)
    )
    mujoco.mjr_rectangle(fill_rect, *fill_color)
    knob_width = max(8, int(10 * scale_x))
    knob_left = int(
      slider_rect.left + slider_fraction * slider_rect.width - knob_width / 2
    )
    knob_rect = mujoco.MjrRect(
      knob_left,
      slider_rect.bottom - max(2, int(3 * scale_y)),
      knob_width,
      slider_rect.height + max(4, int(6 * scale_y)),
    )
    mujoco.mjr_rectangle(knob_rect, 0.95, 0.96, 0.98, 1.0)

    slider_label_rect = mujoco.MjrRect(
      int(slider_x * scale_x),
      int((window_height - 348.0) * scale_y),
      int(slider_width * scale_x),
      int(30.0 * scale_y),
    )
    mujoco.mjr_overlay(
      mujoco.mjtFont.mjFONT_NORMAL,
      mujoco.mjtGridPos.mjGRID_TOPLEFT,
      slider_label_rect,
      f"THRUST {displayed_throttle * 100:.1f}%  {slider_owner}",
      "",
      self.context,
    )

    indicator_x, indicator_y, indicator_width, indicator_height = (
      self._controller_indicator_rect_window(window_width)
    )
    indicator_rect = mujoco.MjrRect(
      int(indicator_x * scale_x),
      int((window_height - indicator_y - indicator_height) * scale_y),
      int(indicator_width * scale_x),
      int(indicator_height * scale_y),
    )
    indicator_label, indicator_color = self._controller_indicator_style()
    mujoco.mjr_rectangle(indicator_rect, *indicator_color)
    mujoco.mjr_overlay(
      self._overlay_font_for_label(indicator_label, indicator_rect.width),
      mujoco.mjtGridPos.mjGRID_TOPLEFT,
      indicator_rect,
      indicator_label,
      "",
      self.context,
    )

  def _render(self) -> None:
    framebuffer_width, framebuffer_height = glfw.get_framebuffer_size(self.window)
    window_width, _ = glfw.get_window_size(self.window)
    viewport = mujoco.MjrRect(0, 0, framebuffer_width, framebuffer_height)
    mujoco.mjv_updateScene(
      self.simulation.model,
      self.simulation.data,
      self.option,
      None,
      self.camera,
      mujoco.mjtCatBit.mjCAT_ALL.value,
      self.scene,
    )
    self._append_thrust_arrow()
    mujoco.mjr_render(viewport, self.scene, self.context)
    telemetry = list(self.simulation.telemetry_lines())
    if time.monotonic() < self.status_until:
      telemetry.append(f"STATUS: {self.status_message}")
    scale_x = (
      framebuffer_width / window_width if window_width > 0 else 1.0
    )
    panel_x, _ = self._control_panel_rect_window(window_width)
    telemetry_width = max(int((panel_x - 12.0) * scale_x), 1)
    telemetry = list(
      self._wrap_overlay_lines(
        telemetry,
        telemetry_width,
        self.context.charWidth,
      )
    )
    telemetry_viewport = mujoco.MjrRect(
      viewport.left,
      viewport.bottom,
      telemetry_width,
      viewport.height,
    )
    mujoco.mjr_overlay(
      mujoco.mjtFont.mjFONT_NORMAL,
      mujoco.mjtGridPos.mjGRID_TOPLEFT,
      telemetry_viewport,
      "\n".join(telemetry),
      "",
      self.context,
    )
    self._draw_button(framebuffer_width, framebuffer_height)

  def _append_thrust_arrow(self) -> None:
    """Append the live thrust visualization to the current MuJoCo scene."""

    arrow = self.simulation.thrust_arrow_world()
    if arrow is None or self.scene.ngeom >= self.scene.maxgeom:
      return

    origin, tip, magnitude_fraction = arrow
    rgba = np.array(
      [
        1.0 - 0.70 * magnitude_fraction,
        0.45 + 0.45 * magnitude_fraction,
        0.08 + 0.92 * magnitude_fraction,
        0.96,
      ],
      dtype=np.float32,
    )
    width = (
      THRUST_ARROW_MIN_WIDTH_M
      + (THRUST_ARROW_MAX_WIDTH_M - THRUST_ARROW_MIN_WIDTH_M)
      * magnitude_fraction
    )
    geom = self.scene.geoms[self.scene.ngeom]
    mujoco.mjv_initGeom(
      geom,
      mujoco.mjtGeom.mjGEOM_ARROW.value,
      np.zeros(3, dtype=np.float64),
      origin,
      np.eye(3, dtype=np.float64).reshape(-1),
      rgba,
    )
    mujoco.mjv_connector(
      geom,
      mujoco.mjtGeom.mjGEOM_ARROW.value,
      width,
      origin,
      tip,
    )
    self.scene.ngeom += 1

  def run(self) -> None:
    previous_time = time.monotonic()
    accumulator = 0.0
    try:
      while not glfw.window_should_close(self.window):
        now = time.monotonic()
        wall_dt = min(now - previous_time, 0.05)
        previous_time = now
        self._read_flight_controls(wall_dt)
        accumulator += wall_dt

        while accumulator >= self.simulation.model.opt.timestep:
          self.simulation.step()
          accumulator -= self.simulation.model.opt.timestep

        self._render()
        glfw.swap_buffers(self.window)
        glfw.poll_events()
    finally:
      self.context.free()
      glfw.destroy_window(self.window)
      glfw.terminate()
      self.simulation.close()


def build_argument_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--async-mpc",
    action="store_true",
    help=(
      "solve MPC guidance on a background worker while a deterministic "
      "200 Hz inner loop tracks its latency-compensated trajectory"
    ),
  )
  return parser


def main() -> None:
  args = build_argument_parser().parse_args()
  simulation = RocketSimulation(
    enable_mpc=True,
    asynchronous_mpc=args.async_mpc,
  )
  simulation.warm_up_mpc()
  RocketWindow(simulation).run()


if __name__ == "__main__":
  main()
