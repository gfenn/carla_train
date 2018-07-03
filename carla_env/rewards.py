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


def compute_reward_custom(env, prev, current):
    reward = 0.0

    cur_dist = current["distance_to_goal"]
    prev_dist = prev["distance_to_goal"]

    if env.config["verbose"]:
        print("Cur dist {}, prev dist {}".format(cur_dist, prev_dist))

    # Distance travelled toward the goal in m
    reward += np.clip(prev_dist - cur_dist, -10.0, 10.0)

    # Speed reward, up 30.0 (km/h)
    reward += np.clip(current["forward_speed"], 0.0, 30.0) / 10

    # New collision damage
    new_damage = (
        current["collision_vehicles"] + current["collision_pedestrians"] +
        current["collision_other"] - prev["collision_vehicles"] -
        prev["collision_pedestrians"] - prev["collision_other"])
    if new_damage:
        reward -= 100.0

    # Sidewalk intersection
    reward -= current["intersection_offroad"]

    # Opposite lane intersection
    reward -= current["intersection_otherlane"]

    # Reached goal
    if current["next_command"] == "REACH_GOAL":
        reward += 100.0

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


REWARD_FUNCTIONS = {
    "corl2017": compute_reward_corl2017,
    "custom": compute_reward_custom,
    "lane_keep": compute_reward_lane_keep,
}


def compute_reward(env, prev, current):
    return REWARD_FUNCTIONS[env.config["reward_function"]](
        env, prev, current)
