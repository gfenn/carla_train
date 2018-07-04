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
    stepReward = 0.0

    # Reward speed times the percent on
    stepReward += current["forward_speed"]

    # # New collision damage
    # new_damage = (
    #     current["collision_vehicles"] + current["collision_pedestrians"] +
    #     current["collision_other"] - prev["collision_vehicles"] -
    #     prev["collision_pedestrians"] - prev["collision_other"])
    # if new_damage and not env.config["early_terminate_on_collision"]:
    #     stepReward -= 100.0

    # # Sidewalk intersection - lose based on how far off we are.  The step we move off
    # # is heavily penalized, but much of the reward can be "earned back" by moving
    # # back into the lane (not all of it).
    # stepReward -= current["intersection_offroad"]
    # if not env.config["early_terminate_on_bounds"] and percent_off > env.config["early_terminate_bounds_limit"]:
    #     stepReward -= 50

    return stepReward


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
