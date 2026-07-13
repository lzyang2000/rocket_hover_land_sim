import math

import numpy as np

from rocket_landing.controller import EngineState, RocketController, ThrustLimits


def test_engine_is_zero_before_ignition_and_bounded_afterward() -> None:
  controller = RocketController()
  assert controller.thrust_magnitude_newtons() == 0.0

  assert controller.ignite()
  assert controller.engine_state is EngineState.LIT
  assert (
    controller.limits.min_thrust_newtons
    <= controller.thrust_magnitude_newtons()
    <= controller.limits.max_thrust_newtons
  )


def test_throttle_cannot_enter_forbidden_gap_after_ignition() -> None:
  controller = RocketController()
  controller.ignite()
  controller.change_throttle(-10.0)
  assert controller.throttle == controller.limits.min_throttle
  assert controller.thrust_magnitude_newtons() == controller.limits.min_thrust_newtons


def test_throttle_respects_upper_bound() -> None:
  controller = RocketController()
  controller.ignite()
  controller.change_throttle(10.0)
  assert controller.throttle == controller.limits.max_throttle
  assert controller.thrust_magnitude_newtons() == controller.limits.max_thrust_newtons


def test_lateral_command_stays_inside_pointing_cone() -> None:
  limits = ThrustLimits(pointing_half_angle_deg=20.0)
  controller = RocketController(limits)
  controller.nudge_lateral(100.0, 100.0)

  direction = controller.thrust_direction_world()
  assert np.linalg.norm(direction) == pytest_approx(1.0)
  assert controller.pointing_angle_deg() == pytest_approx(20.0)
  assert direction[2] >= math.cos(math.radians(20.0))


def test_fuel_consumption_matches_paper_equation() -> None:
  controller = RocketController()
  controller.ignite()
  thrust = controller.thrust_magnitude_newtons()
  burned = controller.consume_fuel(0.5)
  expected = controller.limits.alpha_kg_per_newton_second * thrust * 0.5
  assert burned == pytest_approx(expected)
  assert controller.fuel_mass_kg == pytest_approx(
    controller.initial_fuel_mass_kg - expected
  )


def test_engine_kill_jumps_directly_to_zero_and_requires_reset() -> None:
  controller = RocketController()
  controller.ignite()
  assert controller.thrust_magnitude_newtons() > 0.0
  assert controller.kill_engine()
  assert controller.engine_state is EngineState.SHUTDOWN
  assert controller.thrust_magnitude_newtons() == 0.0


def test_ballistic_coast_can_relight_but_kill_remains_permanent() -> None:
  controller = RocketController()
  controller.ignite()
  controller.throttle = 0.55

  assert controller.begin_coast()
  assert controller.engine_state is EngineState.COAST
  assert controller.thrust_magnitude_newtons() == 0.0
  fuel_before = controller.fuel_mass_kg
  assert controller.consume_fuel(2.0) == 0.0
  assert controller.fuel_mass_kg == fuel_before

  assert controller.relight(throttle=controller.limits.max_throttle)
  assert controller.engine_state is EngineState.LIT
  assert controller.throttle == controller.limits.max_throttle
  assert controller.begin_coast()
  assert controller.kill_engine()
  assert controller.engine_state is EngineState.SHUTDOWN
  assert not controller.relight()
  assert not controller.ignite()
  assert not controller.ignite()
  assert not controller.kill_engine()
  controller.reset()
  assert controller.engine_state is EngineState.OFF
  assert controller.thrust_magnitude_newtons() == 0.0


def pytest_approx(value: float):
  import pytest

  return pytest.approx(value, rel=1e-9, abs=1e-9)
