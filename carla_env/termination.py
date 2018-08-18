
def terminate_on_collision(curr, prev):
    collided = (
        curr["collision_vehicles"] > 0 or curr["collision_pedestrians"] > 0 or
        curr["collision_other"] > 0)
    return bool(collided or curr["total_reward"] < -100)


# Returns true when moving farther out of bounds.  Uses prev instead of 0 since
# some episodes start slightly OOB.  This will allow them to operate as normal
# so long as they move inwards.
def terminate_on_otherlane(curr, prev):
    bounded = (
        curr["intersection_otherlane"] > 0.99
    )
    return bool(bounded)


def terminate_on_offroad(curr, prev):
    bounded = (
        curr["intersection_offroad"] > 0.50
    )
    return bool(bounded)

def terminate_no_movement(curr, prev):
    # Increase your tracking
    tracking_no_move = prev.get("tracking_no_movement", 0)
    curr["tracking_no_movement"] = tracking_no_move + 1

    # Apply filter
    BETA = 0.99
    running_total = prev.get("tracking_no_movement_running", 0)
    new_running_total = running_total * BETA + curr["forward_speed"] * (1 - BETA)
    curr["tracking_no_movement_running"] = new_running_total

    # Bad?
    return tracking_no_move > 200 and new_running_total < 0.5


EARLY_TERMINATIONS = "early_terminations"
TERMINATE_ON_COLLISION = "on_collision"
TERMINATE_ON_OTHERLANE = "on_otherlane"
TERMINATE_ON_OFFROAD = "on_offroad"
TERMINATE_NO_MOVEMENT = "no_movement"


TERMINATION_FUNCTIONS = {
    TERMINATE_ON_COLLISION: terminate_on_collision,
    TERMINATE_ON_OTHERLANE: terminate_on_otherlane,
    TERMINATE_ON_OFFROAD: terminate_on_offroad,
    TERMINATE_NO_MOVEMENT: terminate_no_movement
}


def compute_termination(env, curr, prev):
    terms = env.config[EARLY_TERMINATIONS]
    if terms:
        for term in terms:
            if TERMINATION_FUNCTIONS[term](curr, prev):
                return True
    return False
