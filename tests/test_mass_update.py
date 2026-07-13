from rocket_landing.sim import RocketSimulation


def test_propellant_mass_updates_do_not_reset_free_joint_position() -> None:
  simulation = RocketSimulation()
  simulation.controller.ignite()
  simulation.controller.throttle = 0.80

  previous_altitude = float(simulation.data.qpos[2])
  largest_downward_jump = 0.0
  for _ in range(1200):
    simulation.step()
    altitude = float(simulation.data.qpos[2])
    largest_downward_jump = min(
      largest_downward_jump, altitude - previous_altitude
    )
    previous_altitude = altitude

  assert largest_downward_jump > -0.05
  assert simulation.data.qpos[2] > 100.0
