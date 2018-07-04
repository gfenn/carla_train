
def terminate_on_collision(py_measurements):
    m = py_measurements
    collided = (
        m["collision_vehicles"] > 0 or m["collision_pedestrians"] > 0 or
        m["collision_other"] > 0)
    return bool(collided or m["total_reward"] < -100)


def terminate_on_leave_bounds(py_measurements):
    m = py_measurements
    bounded = (
        m["intersection_offroad"] > 0 or
        m["intersection_otherlane"] > 0
    )
    return bool(bounded)


EARLY_TERMINATIONS = "early_terminations"
TERMINATE_ON_COLLISION = "on_collision"
TERMINATE_ON_LEAVE_BOUNDS = "on_leave_bounds"


TERMINATION_FUNCTIONS = {
    TERMINATE_ON_COLLISION: terminate_on_collision,
    TERMINATE_ON_LEAVE_BOUNDS: terminate_on_leave_bounds,
}


def compute_termination(env, py_measurements):
    terms = env.config[EARLY_TERMINATIONS]
    if terms:
        for term in terms:
            if TERMINATION_FUNCTIONS[term](py_measurements):
                return True
    return False
