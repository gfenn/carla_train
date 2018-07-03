
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


TERMINATION_FUNCTIONS = {
    "on_collision": terminate_on_collision,
    "on_leave_bounds": terminate_on_leave_bounds,
}


def compute_termination(env, py_measurements):
    terms = env.config["early_terminations"]
    if terms:
        for term in terms:
            if TERMINATION_FUNCTIONS[term](py_measurements):
                return True
    return False
