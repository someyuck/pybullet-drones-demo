"""
Modified from the `pid_velocity.py` example from gym-pybullet-drones
"""

import time
# import argparse
import numpy as np

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync

from gym_pybullet_drones.envs.VelocityAviary import VelocityAviary

DEFAULT_DRONE = DroneModel("cf2x")
DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = False
DEFAULT_PLOT = False
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_OBSTACLES = False
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 48
DEFAULT_DURATION_SEC = 5
DEFAULT_OUTPUT_FOLDER = "results"
DEFAULT_COLAB = False


def run():
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
    INIT_XYZS = np.array([[0, 0, 0.1], [0.3, 0, 0.1], [0.6, 0, 0.1], [0.9, 0, 0.1]])
    INIT_RPYS = np.array(
        [[0, 0, 0], [0, 0, np.pi / 3], [0, 0, np.pi / 4], [0, 0, np.pi / 2]]
    )
    PHY = Physics.PYB

    #### Create the environment ################################
    env = VelocityAviary(
        drone_model=drone,
        num_drones=4,
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

    #### Compute number of control steps in the simlation ######
    PERIOD = duration_sec
    NUM_WP = control_freq_hz * PERIOD
    wp_counters = np.array([0 for i in range(4)])

    #### Initialize the velocity target ########################
    TARGET_VEL = np.zeros((4, NUM_WP, 4))
    for i in range(NUM_WP):
        TARGET_VEL[0, i, :] = (
            [-0.5, 1, 0, 0.99] if i < (NUM_WP / 8) else [0.5, -1, 0, 0.99]
        )
        TARGET_VEL[1, i, :] = (
            [0, 1, 0, 0.99] if i < (NUM_WP / 8 + NUM_WP / 6) else [0, -1, 0, 0.99]
        )
        TARGET_VEL[2, i, :] = (
            [0.2, 1, 0.2, 0.99]
            if i < (NUM_WP / 8 + 2 * NUM_WP / 6)
            else [-0.2, -1, -0.2, 0.99]
        )
        TARGET_VEL[3, i, :] = (
            [0, 1, 0.5, 0.99]
            if i < (NUM_WP / 8 + 3 * NUM_WP / 6)
            else [0, -1, -0.5, 0.99]
        )

    #### Initialize the logger #################################
    logger = Logger(
        logging_freq_hz=control_freq_hz,
        num_drones=4,
        output_folder=output_folder,
        colab=colab,
    )

    #### Run the simulation ####################################
    action = np.zeros((4, 4))
    START = time.time()
    for i in range(0, int(duration_sec * env.CTRL_FREQ)):

        ############################################################
        # for j in range(3): env._showDroneLocalAxes(j)

        #### Step the simulation ###################################
        obs, reward, terminated, truncated, info = env.step(action)

        #### Compute control for the current way point #############
        for j in range(4):
            action[j, :] = TARGET_VEL[j, wp_counters[j], :]

        #### Go to the next way point and loop #####################
        for j in range(4):
            wp_counters[j] = wp_counters[j] + 1 if wp_counters[j] < (NUM_WP - 1) else 0

        #### Log the simulation ####################################
        for j in range(4):
            logger.log(
                drone=j,
                timestamp=i / env.CTRL_FREQ,
                state=obs[j],
                control=np.hstack([TARGET_VEL[j, wp_counters[j], 0:3], np.zeros(9)]),
            )

        #### Printout ##############################################
        env.render()

        #### Sync the simulation ###################################
        if gui:
            sync(i, START, env.CTRL_TIMESTEP)

    #### Close the environment #################################
    env.close()

    #### Plot the simulation results ###########################
    logger.save_as_csv("vel")  # Optional CSV save
    if plot:
        logger.plot()


if __name__ == "__main__":
    #### Define and parse (optional) arguments for the script ##
    # parser = argparse.ArgumentParser(
    #     description="Velocity control example using VelocityAviary"
    # )
    # ARGS = parser.parse_args()

    run()
