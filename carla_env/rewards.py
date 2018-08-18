import numpy as np


def compute_reward_corl2017(env, prev, current):
    reward = 0.0

    cur_dist = current["distance_to_goal"]

    prev_dist = prev["distance_to_goal"]

    if env.config["verbose"]:
        print("Cur dist {}, prev dist {}".format(cur_dist, prev_dist))

    # Distance travelled toward the goal in m
    reward += np.clip(prev_dist - cur_dist, -10.0, 10.0)

    # Change in speed (km/h)
    reward += 0.05 * (current["forward_speed"] - prev["forward_speed"])

    # New collision damage
    reward -= .00002 * (
        current["collision_vehicles"] + current["collision_pedestrians"] +
        current["collision_other"] - prev["collision_vehicles"] -
        prev["collision_pedestrians"] - prev["collision_other"])

    # New sidewalk intersection
    reward -= 2 * (
        current["intersection_offroad"] - prev["intersection_offroad"])

    # New opposite lane intersection
    reward -= 2 * (
        current["intersection_otherlane"] - prev["intersection_otherlane"])

    return reward


def compute_reward_lane_keep(env, prev, current):
    # Reward based on movement
    desired_speed = 25
    speed = current["forward_speed"] * 3.8
    speed_reward = 0
    if speed > 1.0:
        speed_reward = 1.1 - abs((speed - desired_speed) / desired_speed)
    reward = speed_reward

    # Apply otherlane penalty
    otherlane = current["intersection_otherlane"]
    if otherlane > 0:
        reward -= 0.2 + speed_reward * otherlane * 1.2

    # Apply offroad penalty (full offroad will completely convert speed reward into a penalty)
    offroad = current["intersection_offroad"]
    if offroad > 0:
        reward -= 0.5 + speed_reward * offroad * 2

    # Collision penalty
    if current["collision_vehicles"] or current["collision_pedestrians"] or current["collision_other"]:
        reward -= 10 + (speed ** 2) / 2

    # Apply slight penalty for turning, more penalty for larger turns
    steer_delta = abs(current["control"]["steer"] - prev["control"]["steer"])
    reward -= (steer_delta ** 2) / 2

    # Apply slight penalty for slamming breaks/gas, more penalty for larger values
    throttle_delta = abs(current["control"]["throttle_brake"] - prev["control"]["throttle_brake"])
    reward -= (throttle_delta ** 2) / 10

    return reward


REWARD_CORL2017 = "corl2017"
REWARD_CUSTOM = "custom"
REWARD_LANE_KEEP = "lane_keep"


REWARD_FUNCTIONS = {
    "corl2017": compute_reward_corl2017,
    "lane_keep": compute_reward_lane_keep,
}


def compute_reward(env, prev, current):
    return REWARD_FUNCTIONS[env.config["reward_function"]](
        env, prev, current)
