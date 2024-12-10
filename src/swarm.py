"""
Modified from the `pid_velocity.py` example from gym-pybullet-drones

Run as `python -m src.swarm --num_drones=<number>`
"""

import time

import argparse
import numpy as np

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync

from gym_pybullet_drones.envs.VelocityAviary import VelocityAviary

DEFAULT_NUM_DRONES = 4
DEFAULT_DRONE = DroneModel("cf2x")
DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = False
DEFAULT_PLOT = False
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_OBSTACLES = False
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 48
DEFAULT_DURATION_SEC = 60
DEFAULT_OUTPUT_FOLDER = "results"
DEFAULT_COLAB = False
DEFAULT_INIT_POS = 0

# protocol constants
COLLISION_THRESHOLD = 0.4
STIFFNESS_GAIN = 0.7
FLOCKING_GAIN = 0.7
COLLISION_AVOIDANCE_GAIN = 2.0
GAMMA = 0.5


def get_circle_points(num_points: int, radius: float, height: float) -> np.ndarray:
    points = np.zeros((num_points, 3))
    for i in range(num_points):
        angle = i * (2 * np.pi / num_points)
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        points[i, :] = [x, y, height]

    return points


def get_bounded_random_points(num_points: int, max_radius: float, max_height: float):
    angles = np.random.uniform(0, 2 * np.pi, size=num_points)
    distances = np.random.uniform(0, max_radius, size=num_points)
    heights = np.random.uniform(0, max_height, size=num_points)

    x = distances * np.cos(angles)
    y = distances * np.sin(angles)

    points = np.column_stack((x, y, heights))
    return points


def run(
    num_drones: int = DEFAULT_NUM_DRONES,
    init_pos: int = DEFAULT_INIT_POS,
):
    # use defaults
    drone = DEFAULT_DRONE
    gui = DEFAULT_GUI
    record_video = DEFAULT_RECORD_VIDEO
    plot = DEFAULT_PLOT
    user_debug_gui = DEFAULT_USER_DEBUG_GUI
    obstacles = DEFAULT_OBSTACLES
    simulation_freq_hz = DEFAULT_SIMULATION_FREQ_HZ
    control_freq_hz = DEFAULT_CONTROL_FREQ_HZ
    duration_sec = DEFAULT_DURATION_SEC
    output_folder = DEFAULT_OUTPUT_FOLDER
    colab = DEFAULT_COLAB

    #### Initialize the simulation #############################
    match init_pos:
        case 0:
            CUR_POS = np.zeros((num_drones, 3))
            for j in range(num_drones):
                CUR_POS[j, :] = [0.5 * j, 0, 3.0]
        case 1:
            CUR_POS = get_circle_points(num_drones, 3.0, 3.0)
        case 2:
            CUR_POS = get_bounded_random_points(num_drones, 3.0, 3.0)
        case _:
            raise ValueError(f"Invalid initial position type {init_pos}")

    CUR_VELS = np.zeros((num_drones, 4))
    INIT_XYZS = CUR_POS.copy()
    INIT_RPYS = np.zeros((num_drones, 3))
    PHY = Physics.PYB

    #### Create the environment ################################
    env = VelocityAviary(
        drone_model=drone,
        num_drones=num_drones,
        initial_xyzs=INIT_XYZS,
        initial_rpys=INIT_RPYS,
        physics=Physics.PYB,
        neighbourhood_radius=10,
        pyb_freq=simulation_freq_hz,
        ctrl_freq=control_freq_hz,
        gui=gui,
        record=record_video,
        obstacles=obstacles,
        user_debug_gui=user_debug_gui,
    )

    #### Obtain the PyBullet Client ID from the environment ####
    PYB_CLIENT = env.getPyBulletClient()
    DRONE_IDS = env.getDroneIds()

    #### Velocity target is calculated in real time acc to protocol ########################
    TARGET_VEL = np.zeros((num_drones, 4))

    #### Initialize the logger #################################
    logger = Logger(
        logging_freq_hz=control_freq_hz,
        num_drones=num_drones,
        output_folder=output_folder,
        colab=colab,
    )

    def are_colliding(i: int, j: int) -> bool:
        distance = np.linalg.norm(CUR_POS[i] - CUR_POS[j])
        return distance <= COLLISION_THRESHOLD

    input("\nPress enter to start:")

    #### Run the simulation ####################################
    action = np.zeros((num_drones, 4))
    START = time.time()
    for i in range(0, int(duration_sec * env.CTRL_FREQ)):

        ############################################################
        # for j in range(num_drones): env._showDroneLocalAxes(j)

        #### Step the simulation ###################################
        obs, reward, terminated, truncated, info = env.step(action)

        # read position and velocity
        CUR_POS = obs[:, :3]
        CUR_VELS = obs[:, 10:13]

        # consensus protocol obeying Reynolds' rules
        # treating all drones as neighbours
        for j in range(num_drones):
            TARGET_VEL[j, :] = [0, 0, 0, 0]
            for n in range(num_drones):
                if n == j:
                    continue
                if are_colliding(j, n):
                    TARGET_VEL[j, :3] -= COLLISION_AVOIDANCE_GAIN * (
                        CUR_POS[n] - CUR_POS[j]
                    )
                else:
                    TARGET_VEL[j, :3] += (
                        STIFFNESS_GAIN
                        * FLOCKING_GAIN
                        * (
                            (CUR_POS[n] - CUR_POS[j])
                            + GAMMA * (CUR_VELS[n] - CUR_VELS[j])
                        )
                    )

                speed = np.linalg.norm(TARGET_VEL[j, :3])
                TARGET_VEL[j, :3] /= speed
                TARGET_VEL[j, 3] = speed / env.SPEED_LIMIT

        #### Compute control for the current way point #############
        action = TARGET_VEL

        #### Log the simulation ####################################
        for j in range(num_drones):
            logger.log(
                drone=j,
                timestamp=i / env.CTRL_FREQ,
                state=obs[j],
                control=np.hstack([TARGET_VEL[j, :3], np.zeros(9)]),
            )

        #### Printout ##############################################
        env.render()

        #### Sync the simulation ###################################
        if gui:
            sync(i, START, env.CTRL_TIMESTEP)

    #### Close the environment #################################
    input("\nPress enter to close:")
    env.close()

    #### Plot the simulation results ###########################
    logger.save_as_csv("vel")  # Optional CSV save
    if plot:
        logger.plot()


if __name__ == "__main__":
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(
        description="Velocity control example using VelocityAviary"
    )
    parser.add_argument(
        "--num_drones",
        default=DEFAULT_NUM_DRONES,
        type=int,
        help="Number of drones (default: 4)",
        metavar="",
    )
    parser.add_argument(
        "--init_pos",
        default=DEFAULT_INIT_POS,
        type=int,
        help="Type of initial position (on a line: 0, on a circle: 1, random: 2)",
        choices=(0, 1, 2),
        metavar="",
    )

    run(**vars(parser.parse_args()))
