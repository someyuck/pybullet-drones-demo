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

# protocol constants
COLLISION_THRESHOLD = 0.4
STIFFNESS_GAIN = 0.7
FLOCKING_GAIN = 0.7
COLLISION_AVOIDANCE_GAIN = 2.0
GAMMA = 0.5


def run(
    num_drones: int = DEFAULT_NUM_DRONES,
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
    CUR_POS = np.zeros((num_drones, 3))
    for j in range(num_drones):
        CUR_POS[j, :] = [0.5 * j, 0, 3.0]
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
    MAX_SPEED = 0.03 * (5 * env.MAX_SPEED_KMH / 18)  # m/s
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

    input()

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
                TARGET_VEL[j, 3] = np.linalg.norm(TARGET_VEL[j, :3]) / MAX_SPEED

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
    input("Press enter to close:")
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

    run(**vars(parser.parse_args()))
