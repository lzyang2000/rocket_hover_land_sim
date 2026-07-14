import numpy as np
import pytest

from rocket_landing.mass_properties import (
  MassProperties,
  RocketMassModel,
  combine_axial_mass_properties,
)
from rocket_landing.mpc import MPCConfig, SixDofMPC
from rocket_landing.sim import RocketSimulation


def test_tank_model_matches_initial_vehicle_and_moves_center_of_mass() -> None:
  model = RocketMassModel()
  wet = model.properties(model.initial_mass_kg)
  half_reserve = model.properties(
    model.dry_mass_kg + 0.5 * model.initial_propellant_mass_kg
  )
  dry = model.properties(model.dry_mass_kg)

  assert wet.center_of_mass_body_m == pytest.approx(np.zeros(3))
  assert wet.inertia_at_com_kgm2 == pytest.approx(model.initial_inertia_kgm2)
  assert half_reserve.center_of_mass_body_m[2] < -1.0
  assert dry.center_of_mass_body_m[2] == pytest.approx(
    model.dry_center_of_mass_z_m
  )
  assert dry.inertia_at_com_kgm2[0] > (
    wet.inertia_at_com_kgm2[0] * model.dry_mass_kg / model.initial_mass_kg
  )
  assert dry.inertia_at_com_kgm2[2] < wet.inertia_at_com_kgm2[2]


def test_mujoco_and_mpc_share_fuel_dependent_mass_properties() -> None:
  simulation = RocketSimulation()
  simulation.controller.fuel_mass_kg = 4_500.0
  simulation._update_model_mass(force=True)
  installed = simulation.applied_mass_properties()
  expected = simulation.mass_model.properties(25_500.0)

  assert installed.mass_kg == pytest.approx(expected.mass_kg)
  assert installed.center_of_mass_body_m == pytest.approx(
    expected.center_of_mass_body_m
  )
  assert installed.inertia_at_com_kgm2 == pytest.approx(
    expected.inertia_at_com_kgm2
  )

  predictor = SixDofMPC(
    MPCConfig(
      initial_mass_kg=simulation.mass_model.initial_mass_kg,
      dry_mass_kg=simulation.mass_model.dry_mass_kg,
      initial_inertia_kgm2=simulation.mass_model.initial_inertia_kgm2,
      mass_model=simulation.mass_model,
      horizon_steps=4,
      successive_iterations=1,
    )
  )
  predicted = predictor.mass_properties(25_500.0)
  assert predicted.center_of_mass_body_m == pytest.approx(
    installed.center_of_mass_body_m
  )
  assert predicted.inertia_at_com_kgm2 == pytest.approx(
    installed.inertia_at_com_kgm2
  )


def test_attached_upper_stack_shifts_com_and_increases_pitch_inertia() -> None:
  primary = MassProperties(
    mass_kg=400_000.0,
    center_of_mass_body_m=np.zeros(3),
    inertia_at_com_kgm2=np.array([60e6, 60e6, 1e6]),
  )
  combined = combine_axial_mass_properties(
    primary,
    attached_mass_kg=120_000.0,
    attached_center_z_m=34.0,
    attached_inertia_at_com_kgm2=np.array([9e6, 9e6, 0.2e6]),
  )

  assert combined.mass_kg == pytest.approx(520_000.0)
  assert combined.center_of_mass_body_m[2] == pytest.approx(
    120_000.0 * 34.0 / 520_000.0
  )
  assert combined.inertia_at_com_kgm2[0] > primary.inertia_at_com_kgm2[0]
