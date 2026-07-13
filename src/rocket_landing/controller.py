"""Paper-inspired thrust constraints and manual guidance commands."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
import math

import numpy as np


class EngineState(Enum):
  OFF = auto()
  LIT = auto()
  COAST = auto()
  SHUTDOWN = auto()
  FUEL_OUT = auto()


@dataclass(frozen=True)
class ThrustLimits:
  """Paper-style thrust bounds scaled to a Falcon 9 landing configuration."""

  nominal_max_newtons: float = 720_000.0
  min_throttle: float = 0.20
  max_throttle: float = 0.80
  pointing_half_angle_deg: float = 20.0
  alpha_kg_per_newton_second: float = 5.0e-4

  @property
  def min_thrust_newtons(self) -> float:
    return self.nominal_max_newtons * self.min_throttle

  @property
  def max_thrust_newtons(self) -> float:
    return self.nominal_max_newtons * self.max_throttle


class RocketController:
  """Stateful manual controller with a latched, minimum-thrust engine."""

  def __init__(
    self,
    limits: ThrustLimits | None = None,
    *,
    dry_mass_kg: float = 21_000.0,
    fuel_mass_kg: float = 9_000.0,
  ) -> None:
    self.limits = limits or ThrustLimits()
    self.dry_mass_kg = dry_mass_kg
    self.initial_fuel_mass_kg = fuel_mass_kg
    self.reset()

  def reset(self) -> None:
    self.engine_state = EngineState.OFF
    self.throttle = self.limits.min_throttle
    self.lateral_command = np.zeros(2, dtype=float)
    self.fuel_mass_kg = self.initial_fuel_mass_kg

  @property
  def wet_mass_kg(self) -> float:
    return self.dry_mass_kg + self.fuel_mass_kg

  def ignite(self) -> bool:
    """Ignite once. A killed engine cannot restart without resetting."""

    if self.engine_state is not EngineState.OFF or self.fuel_mass_kg <= 0.0:
      return False
    self.engine_state = EngineState.LIT
    self.throttle = float(
      np.clip(self.throttle, self.limits.min_throttle, self.limits.max_throttle)
    )
    return True

  def begin_coast(self) -> bool:
    """Temporarily stop thrust while keeping the engine armed to relight."""

    if self.engine_state is not EngineState.LIT:
      return False
    self.engine_state = EngineState.COAST
    return True

  def relight(self, *, throttle: float | None = None) -> bool:
    """Relight an armed coasting engine without weakening permanent kill."""

    if self.engine_state is not EngineState.COAST or self.fuel_mass_kg <= 0.0:
      return False
    if throttle is not None:
      self.throttle = float(throttle)
    self.throttle = float(
      np.clip(self.throttle, self.limits.min_throttle, self.limits.max_throttle)
    )
    self.engine_state = EngineState.LIT
    return True

  def kill_engine(self) -> bool:
    """Discontinuously transition from valid positive thrust to zero."""

    if self.engine_state not in (EngineState.LIT, EngineState.COAST):
      return False
    self.engine_state = EngineState.SHUTDOWN
    return True

  def change_throttle(self, delta: float) -> None:
    self.throttle = float(
      np.clip(
        self.throttle + delta,
        self.limits.min_throttle,
        self.limits.max_throttle,
      )
    )

  def nudge_lateral(self, dx: float, dy: float) -> None:
    self.lateral_command += np.array([dx, dy], dtype=float)
    length = float(np.linalg.norm(self.lateral_command))
    if length > 1.0:
      self.lateral_command /= length

  def center_lateral(self) -> None:
    self.lateral_command[:] = 0.0

  def thrust_magnitude_newtons(self) -> float:
    if self.engine_state is not EngineState.LIT:
      return 0.0
    return float(
      np.clip(
        self.throttle * self.limits.nominal_max_newtons,
        self.limits.min_thrust_newtons,
        self.limits.max_thrust_newtons,
      )
    )

  def thrust_direction_world(self) -> np.ndarray:
    """Return a unit vector inside the pointing cone about world +Z."""

    command_norm = float(np.linalg.norm(self.lateral_command))
    if command_norm == 0.0:
      return np.array([0.0, 0.0, 1.0])

    horizontal_direction = self.lateral_command / command_norm
    angle = math.radians(self.limits.pointing_half_angle_deg) * command_norm
    return np.array(
      [
        horizontal_direction[0] * math.sin(angle),
        horizontal_direction[1] * math.sin(angle),
        math.cos(angle),
      ]
    )

  def thrust_vector_world(self) -> np.ndarray:
    return self.thrust_magnitude_newtons() * self.thrust_direction_world()

  def pointing_angle_deg(self) -> float:
    direction = self.thrust_direction_world()
    return min(
      math.degrees(math.acos(float(np.clip(direction[2], -1.0, 1.0)))),
      self.limits.pointing_half_angle_deg,
    )

  def consume_fuel(self, dt: float) -> float:
    """Integrate m_dot = -alpha * ||T|| and return fuel burned."""

    if dt <= 0.0 or self.engine_state is not EngineState.LIT:
      return 0.0

    requested = (
      self.limits.alpha_kg_per_newton_second
      * self.thrust_magnitude_newtons()
      * dt
    )
    burned = min(requested, self.fuel_mass_kg)
    self.fuel_mass_kg -= burned
    if self.fuel_mass_kg <= 0.0:
      self.fuel_mass_kg = 0.0
      self.engine_state = EngineState.FUEL_OUT
    return burned
