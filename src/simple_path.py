"""
Modified from the `pid_velocity.py` example from gym-pybullet-drones

Run as `python -m src.simple_path --num_drones={1 or 2}`
"""

import time

import argparse
import numpy as np

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync

from gym_pybullet_drones.envs.VelocityAviary import VelocityAviary

DEFAULT_NUM_DRONES = 1
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

    target_points = (
        np.array([[0, 3, 1.0]])
        if num_drones == 1
        else np.array([[0, 3, 1.0], [3, 0, 1.0]])
    )

    #### Initialize the simulation #############################
    CUR_POS = (
        np.array([[0, 0, 0]]) if num_drones == 1 else np.array([[0, 0, 0], [0, -1, 0]])
    )

    INIT_XYZS = CUR_POS.copy()
    INIT_RPYS = (
        np.array([[0, 0, 0]]) if num_drones == 1 else np.array([[0, 0, 0], [0, 0, 0]])
    )
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

    #### Compute number of control steps in the simlation ######
    PERIOD = duration_sec
    NUM_WP = control_freq_hz * PERIOD
    wp_counters = np.array([0 for i in range(num_drones)])

    #### Initialize the velocity target ########################
    MAX_SPEED = 0.03 * (5 * env.MAX_SPEED_KMH / 18)  # m/s

    TARGET_VEL = np.zeros((num_drones, NUM_WP, 4))
    for j in range(num_drones):
        direction_2d = np.array(
            [
                target_points[j][0] - INIT_XYZS[j][0],
                target_points[j][1] - INIT_XYZS[j][1],
                0.0,
            ]
        )
        distance_to_target = np.linalg.norm(direction_2d)
        direction_2d /= distance_to_target
        ascent_speed = (target_points[j][2] - INIT_XYZS[j][0]) / (PERIOD / 8)
        horizontal_speed = distance_to_target / (5 * PERIOD / 8)

        for i in range(NUM_WP):
            if i < NUM_WP / 8:
                # try to reach z-level
                TARGET_VEL[j, i, :] = [0, 0, 1, ascent_speed / MAX_SPEED]
            elif i < (6 * NUM_WP / 8):
                # 2d motion
                TARGET_VEL[j, i, :] = [
                    direction_2d[0],
                    direction_2d[1],
                    0,
                    horizontal_speed / MAX_SPEED,
                ]
            else:
                # chill
                TARGET_VEL[j, i, :] = [0, 0, 0, 0.0]

    #### Initialize the logger #################################
    logger = Logger(
        logging_freq_hz=control_freq_hz,
        num_drones=num_drones,
        output_folder=output_folder,
        colab=colab,
    )

    #### Run the simulation ####################################
    action = np.zeros((num_drones, 4))
    START = time.time()
    for i in range(0, int(duration_sec * env.CTRL_FREQ)):

        ############################################################
        # for j in range(num_drones): env._showDroneLocalAxes(j)

        #### Step the simulation ###################################
        obs, reward, terminated, truncated, info = env.step(action)

        #### Compute control for the current way point #############
        for j in range(num_drones):
            action[j, :] = TARGET_VEL[j, wp_counters[j], :]

        #### Go to the next way point and loop #####################
        for j in range(num_drones):
            wp_counters[j] = wp_counters[j] + 1 if wp_counters[j] < (NUM_WP - 1) else 0

        #### Log the simulation ####################################
        for j in range(num_drones):
            logger.log(
                drone=j,
                timestamp=i / env.CTRL_FREQ,
                state=obs[j],
                control=np.hstack([TARGET_VEL[0, wp_counters[j], 0:3], np.zeros(9)]),
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
        help="Number of drones (default: 1)",
        choices=(1, 2),
        metavar="",
    )

    run(**vars(parser.parse_args()))
