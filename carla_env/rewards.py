import numpy as np
import random


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


DEBUGGER = True
DESIRED_SPEED = 25
SPEED_CONVERSION = 3.6


# Each step should provide a penalty based on how far away we are from the desired speed
def instantaneous_speed_penalty(prev, current):
    speed = current["forward_speed"] * SPEED_CONVERSION
    delta = abs(speed - DESIRED_SPEED)
    return (delta ** 2) / (DESIRED_SPEED ** 2)


# The goal here is to provide an instantaneous penalty to when the speed is changing "in the wrong direction"
def changing_speed_penalty(prev, current):
    current_speed = current["forward_speed"] * SPEED_CONVERSION
    prev_speed = prev["forward_speed"] * SPEED_CONVERSION

    # If moving TOWARDS then desired speed - no penalty
    if current_speed <= prev_speed <= DESIRED_SPEED:
        return 0
    if DESIRED_SPEED <= prev_speed <= current_speed:
        return 0

    # If moving AWAY from the desired speed - that change is a penalty
    return abs(current_speed - prev_speed)


# The goal behind stagnancy is to penalize the vehicle for staying stationary too long.  The effect of
# stagnancy builds up the longer the vehicle is not moving, rapidly becoming a large penalty.
def stagnancy_penalty(prev, current):
    # Set defaults if not there
    if "_stagnancy" not in prev:
        prev["_stagnancy"] = {
            "collection_steps": 0,
            "stagnancy": 0.0
        }
    stagnancy = prev["_stagnancy"]

    # Determine the stagnancy this step.  This equation has a maximum stagnancy when the vehicle is not moving,
    # rapidly trailing down to 0 at about 4km/hr.  When moving faster, stagnancy is negative, which is used to
    # counter previous steps of stagnancy.  This transition setup prevents the vehicle from learning to "poke"
    # back in to clear the stagnancy and then going back to 0 again.  The only way to avoid a stagnancy penalty
    # that is large is to move, on average, faster than 4km/hr.
    current_speed = max(0.01, current["forward_speed"] * SPEED_CONVERSION)
    instantaneous_stagnancy = 1.0 - (current_speed ** 0.5) / 2

    # New stagnancy between 0 and 10.  Steadily builds up (or returns to 0) based on speeds
    new_stagnancy = min(10, max(0, stagnancy["stagnancy"] + instantaneous_stagnancy))

    # Set the new stagnancy dictionary
    current["_stagnancy"] = {
        "collection_steps": stagnancy["collection_steps"] + 1,
        "stagnancy": new_stagnancy
    }

    # Only return stagnancy once the vehicle has had a bit of time to start moving at beginning of episode
    if current["_stagnancy"]["collection_steps"] >= 50:
        return new_stagnancy
    return 0.0


# Applies a penalty if the vehicle is becoming MORE out of lane/road than it was previously.
def leaving_lane_penalty(prev, current):
    if current["intersection_offroad"] > prev["intersection_offroad"]:
        return 1.0
    if current["intersection_otherlane"] > prev["intersection_otherlane"]:
        return 1.0
    return 0.0


# Applies a steady penalty based on how far out of lane the vehicle is.
def instantaneous_out_of_lane_penalty(prev, current):
    return current["intersection_offroad"] + current["intersection_otherlane"]


# Applies a penalty to dramatic changes in steering
def steer_delta_penalty(prev, current):
    steer_delta = abs(current["control"]["steer"] - prev["control"]["steer"])
    return steer_delta ** 2


# Applies a penalty to dramatic changes in throttle
def throttle_delta_penalty(prev, current):
    throttle_delta = abs(current["control"]["throttle_brake"] - prev["control"]["throttle_brake"])
    return throttle_delta ** 2


# Any collision is very bad - apply penalty on these events.
def collision_detected_penalty(prev, current):
    if current["collision_vehicles"] > prev["collision_vehicles"]:
        return 1.0
    if current["collision_pedestrians"] > prev["collision_pedestrians"]:
        return 1.0
    if current["collision_other"] > prev["collision_other"]:
        return 1.0
    return 0.0


def compute_reward_evolving(env, prev, current):
    # If it was a failure - stop early
    if current["done"]:
        return 0.0

    # Determine the list of penalties
    penalties = dict()
    # Basic driving practices
    penalties["instantaneous_speed"] = instantaneous_speed_penalty(prev, current)
    penalties["changing_speed"] = changing_speed_penalty(prev, current) * 0.1
    penalties["stagnancy"] = stagnancy_penalty(prev, current)
    penalties["steer_delta"] = steer_delta_penalty(prev, current) * 0.05
    penalties["throttle_delta"] = throttle_delta_penalty(prev, current) * 0.01

    # Dangerous events
    penalties["leaving_lane"] = leaving_lane_penalty(prev, current)
    penalties["instantaneous_out_of_lane"] = instantaneous_out_of_lane_penalty(prev, current)
    penalties["collision_detected"] = collision_detected_penalty(prev, current) * 100.0

    # Start with 1 point and whittle it down
    reward = 1
    for penalty_key in penalties:
        reward -= penalties[penalty_key]

    if DEBUGGER:
        items = list()
        for penalty_key in penalties:
            penalty_value = penalties[penalty_key]
            items.append("{key}: {value:>07.4f}".format(key=penalty_key, value=penalty_value))
        items.append("Step Reward: {:+>.4f}".format(reward))
        print(",  ".join(items))

    # Finally we want to apply a bit of noise to try and prevent strange overfitting events
    reward += random.random() * 0.1 - 0.05
    return reward





def compute_reward_refined_lane(env, prev, current):
    return compute_reward_evolving(env, prev, current)
    # # Finished? Failed?
    # if current["done"]:
    #     if current["finished"]:
    #         return 0.0
    #     if current["failed"]:
    #         return -200.0
    #
    # # Reward based on movement
    # desired_speed = 25
    # reward = 0
    # speed = current["forward_speed"] * 3.6
    #
    # # If partially offroad, apply a significant penalty
    # offroad = current["intersection_offroad"]
    # if offroad > 0:
    #     reward -= 1.0 + offroad * 100
    #
    # # If partially out of the lane, apply a lighter penalty
    # otherlane = current["intersection_otherlane"]
    # if otherlane > 0:
    #     reward -= 0.1 + otherlane * 10
    #
    # # Provide a speed reward - reward peaks at desired speed, becoming negative if speed too high
    # if speed < 1:
    #     reward -= 1.0
    # elif speed < 5:
    #     reward -= 0.5
    # else:
    #     reward += 1 - abs((speed - desired_speed) / desired_speed)
    #
    # # Push a penalty if exiting road for first time
    # if prev["intersection_offroad"] == 0 and current["intersection_offroad"] > 0:
    #     reward -= 20.0
    # if prev["intersection_otherlane"] == 0 and current["intersection_otherlane"] > 0:
    #     reward -= 5.0
    #
    # # Collision penalty
    # if current["collision_vehicles"] or current["collision_pedestrians"] or current["collision_other"]:
    #     reward -= 200.0 + ((3.6 * prev["forward_speed"]) ** 2) / 2
    #
    # # Apply slight penalty for turning, more penalty for larger turns
    # steer_delta = abs(current["control"]["steer"] - prev["control"]["steer"])
    # reward -= (steer_delta ** 2) / 2
    #
    # # Apply slight penalty for slamming breaks/gas, more penalty for larger values
    # throttle_delta = abs(current["control"]["throttle_brake"] - prev["control"]["throttle_brake"])
    # reward -= (throttle_delta ** 2) / 10
    #
    # return reward


REWARD_CORL2017 = "corl2017"
REWARD_CUSTOM = "custom"
REWARD_LANE_KEEP = "lane_keep"


REWARD_FUNCTIONS = {
    "corl2017": compute_reward_corl2017,
    "lane_keep": compute_reward_refined_lane,
}


def compute_reward(env, prev, current):
    return REWARD_FUNCTIONS[env.config["reward_function"]](
        env, prev, current)
