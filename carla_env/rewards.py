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
    desiredSpeed = 25
    speed = current["forward_speed"] * 3.8

    # Reward moving forward (up to target speed)
    if speed >= 1:
        stepReward += 0.1 + np.clip(speed, 0.0, desiredSpeed) / desiredSpeed

    # Penalize going over desired, will become negative at 1.5x desired
    if speed > desiredSpeed:
        stepReward -= 2 * (speed - desiredSpeed) / desiredSpeed

    # Apply a penalty if any of these conditions are met
    if not prev["applied_penalty"]:
        leaving_road = current["intersection_offroad"] > prev["intersection_offroad"]
        leaving_lane = current["intersection_otherlane"] > prev["intersection_otherlane"]
        collision_vehicle = current["collision_vehicles"] > prev["collision_vehicles"]
        collision_ped = current["collision_pedestrians"] > prev["collision_pedestrians"]
        collision_other = current["collision_other"] > prev["collision_other"]
        if leaving_road or leaving_lane:
            stepReward -= 1 + (speed ** 2) / 100.0
            current["applied_penalty"] = True
        elif collision_vehicle or collision_ped or collision_other:
            stepReward -= 1 + (speed ** 2) / 10.0
            current["applied_penalty"] = True

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
