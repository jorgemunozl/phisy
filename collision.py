"""
Use the momentum conservation
And the collision is perfectly elastic
We only want to simulate the number of collisions
"""
import logging
import matplotlib.pyplot as plt
import os
import numpy as np

from physi.mechanics import body
from physi.mechanics.body import point_body

COLLISION_COUNT = 0
flag = True

# massive body
body_slow = point_body(
    mass=1000000.0,
    v_0=-10.0,
    x_0=20,
)

# fast body
body_fast = point_body(
    mass=1.0,
    v_0=0.0,
    x_0=10.0,
)

A = np.array([[-1, 1], [body_fast.mass, body_slow.mass]])

logger = logging.getLogger("collision")


def update_velocties(v_1, v_2):
    """
    v_1 is the fast body's velocity
    v_2 is the slow body's velocity
    """
    logger.debug(
        "Solving elastic collision: v_fast=%+.4f, v_slow=%+.4f (m_fast=%.1f, m_slow=%.1f)",
        v_1,
        v_2,
        body_fast.mass,
        body_slow.mass,
    )
    v_1_2 = v_1 - v_2
    sum = body_fast.mass * v_1 + body_slow.mass * v_2
    b = np.array([v_1_2, sum])
    v_1_prime, v_2_prime = np.linalg.solve(A, b)
    logger.debug(
        "New velocities after collision: v_fast=%+.4f, v_slow=%+.4f",
        v_1_prime,
        v_2_prime,
    )
    return v_1_prime, v_2_prime




def calculate_collision(x_1, x_2, v_1, v_2):
    """
    x_1 is the fast body's position
    x_2 is the slow body's position
    """
    global flag
    global COLLISION_COUNT
    if not flag:
        return 0, 0
    logger.debug(
        "State -> x_fast=%+.4f, x_slow=%+.4f, v_fast=%+.4f, v_slow=%+.4f",
        x_1,
        x_2,
        v_1,
        v_2,
    )

    time_collision = (x_1 - x_2) / (v_2 - v_1)
    logger.info("Time to body-body collision: %+.4f", time_collision)

    if time_collision <= 0:
        # Calculate the time that it will take to fast body to reach the wall
        logger.info("Body-body collision not possible -> checking wall rebound instead")
        if v_1 == 0 and v_2 > 0:
            logger.info("ATTENTION: Fast body cannot reach the wall -> collision not possible")
            flag = False
            return 0, 0
        if v_1 != 0:
            time_collision_wall = -x_1 / v_1
        logger.info("Time for fast body to reach the wall: %+.4f", time_collision_wall)
        if time_collision_wall <= 0:
            logger.info("Fast body cannot reach the wall -> collision not possible")
            flag = False
            return 0, 0
        COLLISION_COUNT += 1
        x_2_new = x_2 + time_collision_wall * v_2
        v_1_new = -v_1
        logger.info(
            "Wall bounce: fast body reflects (v %+.4f -> %+.4f) at x=0, slow body drifts to x=%+.4f",
            v_1,
            v_1_new,
            x_2_new,
        )
        body_fast.v_0 = v_1_new
        body_fast.x_0 = x_2_new
        body_fast.x_0 = 0
        time_collision, pos_collision = calculate_collision(0, x_2_new, v_1_new, v_2)
    else:
        COLLISION_COUNT += 1
        pos_collision = time_collision * v_1 + x_1
        logger.info("Collision position: %+.4f", pos_collision)
    return time_collision, pos_collision


def main():
    logging.basicConfig(
        level=logging.DEBUG if os.environ.get("COLLISION_DEBUG") else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logger.info("━" * 46)
    logger.info("Starting elastic collision simulation")
    logger.info("━" * 46)
    logger.info(
        "Bodies: fast {mass=%.1f, x=%.4f, v=%+.4f} | slow {mass=%.1f, x=%.4f, v=%+.4f}",
        body_fast.mass,
        body_fast.x_0,
        body_fast.v_0,
        body_slow.mass,
        body_slow.x_0,
        body_slow.v_0,
    )
    number_steps = 10
    global_time = 0.
    times, positions = [], []
    while flag:
        logger.info("━" * 46)

        time_i, pos_i = calculate_collision(
            body_fast.x_0, body_slow.x_0, body_fast.v_0, body_slow.v_0
        )
        global_time += time_i
        times.append(global_time)
        positions.append(pos_i)
        logger.info("-> time=%.4f, pos=%.4f", global_time, pos_i)
        logger.info("==" * 4)
        logger.info("Positions before collision: fast=%.4f, slow=%.4f", body_fast.x_0, body_slow.x_0)
        logger.info("Collision detected at %.4f", pos_i)
        logger.info("==" * 4)
        body_fast.x_0, body_slow.x_0 = pos_i, pos_i
        # Update the velocities after the collision
        logger.info(
            "Velocities BEFORE update: v_fast=%+.4f, v_slow=%+.4f",
            body_fast.v_0,
            body_slow.v_0,
        )

        body_fast.v_0, body_slow.v_0 = update_velocties(body_fast.v_0, body_slow.v_0)
        logger.info(
            "Velocities AFTER update: v_fast=%+.4f, v_slow=%+.4f",
            body_fast.v_0,
            body_slow.v_0,
        )
        if COLLISION_COUNT > 100000:
            logger.info("Collision count exceeded 10, stopping simulation")
            break
        logger.info("Collision count: %d", COLLISION_COUNT)


    logger.info("━" * 46)
    logger.info(
        f"Simulation finished after {number_steps} steps -> final x_fast=%.4f, x_slow=%.4f, v_fast=%+.4f, v_slow=%+.4f",
        body_fast.x_0,
        body_slow.x_0,
        body_fast.v_0,
        body_slow.v_0,
    )
    logger.info(f"Collision count: {COLLISION_COUNT}")
    plt.plot(times, positions)
    plt.xlabel("Time")
    plt.ylabel("Position")
    plt.show()


if __name__ == "__main__":
    main()
