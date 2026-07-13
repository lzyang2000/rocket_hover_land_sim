import mujoco
import numpy as np

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
