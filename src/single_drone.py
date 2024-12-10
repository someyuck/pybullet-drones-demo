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
DEFAULT_DURATION_SEC = 10
DEFAULT_OUTPUT_FOLDER = "results"
DEFAULT_COLAB = False

# motion params
DEFAULT_TARGET_POINTS: np.ndarray = np.array([[0, 3, 1.0]])


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

    target_points = DEFAULT_TARGET_POINTS

    #### Initialize the simulation #############################
    CUR_POS = np.array([[0, 0, 0]])

    INIT_XYZS = np.array([[0, 0, 0]])
    INIT_RPYS = np.array([[0, 0, 0]])
    PHY = Physics.PYB

    #### Create the environment ################################
    env = VelocityAviary(
        drone_model=drone,
        num_drones=1,
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
    TARGET_VEL = np.zeros((1, NUM_WP, 4))
    direction_2d = np.array([target_points[0][0], target_points[0][1], 0.0])
    distance_to_target = np.linalg.norm(direction_2d)
    direction_2d /= distance_to_target
    ascent_speed = target_points[0][2] / ((NUM_WP / 8) / control_freq_hz)
    horizontal_speed = distance_to_target / (5 * (NUM_WP / 8) / control_freq_hz)

    for i in range(NUM_WP):
        if i < NUM_WP / 8:
            # try to reach z-level
            TARGET_VEL[0, i, :] = [0, 0, 1, ascent_speed]
        elif i < (6 * NUM_WP / 8):
            # 2d motion
            TARGET_VEL[0, i, :] = [
                horizontal_speed * direction_2d[0],
                horizontal_speed * direction_2d[1],
                0,
                horizontal_speed,
            ]
        else:
            # chill
            TARGET_VEL[0, i, :] = [0, 0, 0, 0.0]

    #### Initialize the logger #################################
    logger = Logger(
        logging_freq_hz=control_freq_hz,
        num_drones=1,
        output_folder=output_folder,
        colab=colab,
    )

    #### Run the simulation ####################################
    action = np.zeros((1, 4))
    START = time.time()
    for i in range(0, int(duration_sec * env.CTRL_FREQ)):

        ############################################################
        # for j in range(3): env._showDroneLocalAxes(j)

        #### Step the simulation ###################################
        obs, reward, terminated, truncated, info = env.step(action)

        #### Compute control for the current way point #############
        action[0, :] = TARGET_VEL[0, wp_counters[0], :]

        #### Go to the next way point and loop #####################
        wp_counters[0] = wp_counters[0] + 1 if wp_counters[0] < (NUM_WP - 1) else 0

        #### Log the simulation ####################################
        logger.log(
            drone=0,
            timestamp=i / env.CTRL_FREQ,
            state=obs[0],
            control=np.hstack([TARGET_VEL[0, wp_counters[0], 0:3], np.zeros(9)]),
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
