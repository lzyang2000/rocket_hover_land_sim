"""Fuel-dependent center-of-mass and inertia model for the rocket stage."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MassProperties:
  """Rigid-body mass properties expressed in the vehicle body frame."""

  mass_kg: float
  center_of_mass_body_m: np.ndarray
  inertia_at_com_kgm2: np.ndarray


@dataclass(frozen=True)
class PropellantTank:
  """Axisymmetric effective liquid column used during the landing reserve."""

  name: str
  mass_fraction: float
  bottom_z_m: float
  height_m: float
  radius_m: float

  def properties(
    self, initial_propellant_mass_kg: float, fill_fraction: float
  ) -> tuple[float, float, np.ndarray]:
    fill = float(np.clip(fill_fraction, 0.0, 1.0))
    mass = initial_propellant_mass_kg * self.mass_fraction * fill
    liquid_length = self.height_m * fill
    center_z = self.bottom_z_m + 0.5 * liquid_length
    transverse = mass * (
      3.0 * self.radius_m**2 + liquid_length**2
    ) / 12.0
    axial = 0.5 * mass * self.radius_m**2
    return mass, center_z, np.array([transverse, transverse, axial])


@dataclass(frozen=True)
class RocketMassModel:
  """Calibrated dry-stage plus LOX/RP-1 residual mass model.

  The dry-stage properties are inferred so that the fully fueled landing
  configuration exactly matches ``initial_inertia_kgm2`` and has its center of
  mass at the body origin. Liquid columns shorten toward the tank bottoms as
  propellant drains, moving the combined center of mass and changing inertia.
  """

  dry_mass_kg: float = 21_000.0
  initial_propellant_mass_kg: float = 9_000.0
  initial_inertia_kgm2: tuple[float, float, float] = (
    4_300_000.0,
    4_300_000.0,
    60_000.0,
  )
  tanks: tuple[PropellantTank, ...] = (
    PropellantTank(
      name="LOX",
      mass_fraction=0.72,
      bottom_z_m=-2.0,
      height_m=16.0,
      radius_m=1.65,
    ),
    PropellantTank(
      name="RP-1",
      mass_fraction=0.28,
      bottom_z_m=-14.0,
      height_m=12.0,
      radius_m=1.65,
    ),
  )

  def __post_init__(self) -> None:
    if self.dry_mass_kg <= 0.0 or self.initial_propellant_mass_kg <= 0.0:
      raise ValueError("Dry and initial propellant masses must be positive.")
    fraction_sum = sum(tank.mass_fraction for tank in self.tanks)
    if not np.isclose(fraction_sum, 1.0):
      raise ValueError("Propellant tank mass fractions must sum to one.")
    if any(
      tank.mass_fraction <= 0.0
      or tank.height_m <= 0.0
      or tank.radius_m <= 0.0
      for tank in self.tanks
    ):
      raise ValueError("Tank fractions and dimensions must be positive.")
    if np.any(np.asarray(self.initial_inertia_kgm2, dtype=float) <= 0.0):
      raise ValueError("Initial principal inertias must be positive.")
    if np.any(self.dry_inertia_at_com_kgm2 <= 0.0):
      raise ValueError("Tank assumptions leave a non-physical dry inertia.")

  @property
  def initial_mass_kg(self) -> float:
    return self.dry_mass_kg + self.initial_propellant_mass_kg

  def _tank_properties(
    self, fill_fraction: float
  ) -> tuple[tuple[float, float, np.ndarray], ...]:
    return tuple(
      tank.properties(self.initial_propellant_mass_kg, fill_fraction)
      for tank in self.tanks
    )

  @property
  def dry_center_of_mass_z_m(self) -> float:
    initial_tanks = self._tank_properties(1.0)
    propellant_first_moment = sum(
      mass * center_z for mass, center_z, _ in initial_tanks
    )
    return -propellant_first_moment / self.dry_mass_kg

  @property
  def dry_inertia_at_com_kgm2(self) -> np.ndarray:
    initial_inertia = np.asarray(self.initial_inertia_kgm2, dtype=float)
    dry_com_z = self.dry_center_of_mass_z_m
    tank_contribution = np.zeros(3, dtype=float)
    for mass, center_z, intrinsic_inertia in self._tank_properties(1.0):
      tank_contribution += intrinsic_inertia
      tank_contribution[0:2] += mass * center_z**2
    dry_parallel_axis = np.array(
      [
        self.dry_mass_kg * dry_com_z**2,
        self.dry_mass_kg * dry_com_z**2,
        0.0,
      ]
    )
    return initial_inertia - tank_contribution - dry_parallel_axis

  def properties(self, total_mass_kg: float) -> MassProperties:
    mass = float(np.clip(total_mass_kg, self.dry_mass_kg, self.initial_mass_kg))
    propellant_mass = mass - self.dry_mass_kg
    fill_fraction = propellant_mass / self.initial_propellant_mass_kg
    tank_properties = self._tank_properties(fill_fraction)
    dry_com_z = self.dry_center_of_mass_z_m

    first_moment = self.dry_mass_kg * dry_com_z + sum(
      tank_mass * center_z
      for tank_mass, center_z, _ in tank_properties
    )
    combined_com_z = first_moment / mass

    inertia = self.dry_inertia_at_com_kgm2.copy()
    dry_offset = dry_com_z - combined_com_z
    inertia[0:2] += self.dry_mass_kg * dry_offset**2
    for tank_mass, center_z, intrinsic_inertia in tank_properties:
      inertia += intrinsic_inertia
      tank_offset = center_z - combined_com_z
      inertia[0:2] += tank_mass * tank_offset**2

    return MassProperties(
      mass_kg=mass,
      center_of_mass_body_m=np.array([0.0, 0.0, combined_com_z]),
      inertia_at_com_kgm2=inertia,
    )
