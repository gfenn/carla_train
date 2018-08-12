
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
        curr["intersection_otherlane"] > prev["intersection_otherlane"]
    )
    return bool(bounded)


def terminate_on_offroad(curr, prev):
    bounded = (
        curr["intersection_offroad"] > prev["intersection_offroad"]
    )
    return bool(bounded)


EARLY_TERMINATIONS = "early_terminations"
TERMINATE_ON_COLLISION = "on_collision"
TERMINATE_ON_OTHERLANE = "on_otherlane"
TERMINATE_ON_OFFROAD = "on_offroad"


TERMINATION_FUNCTIONS = {
    TERMINATE_ON_COLLISION: terminate_on_collision,
    TERMINATE_ON_OTHERLANE: terminate_on_otherlane,
    TERMINATE_ON_OFFROAD: terminate_on_offroad,
}


def compute_termination(env, curr, prev):
    terms = env.config[EARLY_TERMINATIONS]
    if terms:
        for term in terms:
            if TERMINATION_FUNCTIONS[term](curr, prev):
                return True
    return False
