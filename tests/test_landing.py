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
  for _ in range(3000):
    simulation.step()
    saw_valid_mpc_command |= (
      simulation.last_mpc_result is not None
      and simulation.last_mpc_result.success
      and not simulation.mpc_using_fallback
    )
    if simulation.landing_phase is LandingPhase.COMPLETE:
      break

  assert simulation.landing_phase is LandingPhase.COMPLETE
  assert saw_valid_mpc_command
  assert simulation.controller.engine_state is EngineState.SHUTDOWN
  assert np.linalg.norm(simulation.data.qpos[0:2]) < 0.10
  assert np.linalg.norm(simulation.data.qvel[0:2]) < 0.10
