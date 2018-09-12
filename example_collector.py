

FPS = 10
OUT_LANE_RESET = 2 * FPS

class EpisodeCollector:

    def __init__(self):
        self.episodes = []
        self.valid_episodes = 0
        self.min_steps = 50
        self.next()

    def next(self):
        if len(self.episodes) > 0 and self.episodes[-1].steps >= self.min_steps:
            self.valid_episodes += 1
        self.episodes.append(EpisodeData())

    def step(self, py_measurements):
        self.episodes[-1].step(py_measurements)

    def results(self):
        data = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        total = 0.0
        for episode in self.episodes:
            # Ensure a valid episode
            if episode.steps < self.min_steps:
                continue

            total += 1
            ep_data = episode.csv_data()
            for i in range(len(data)):
                data[i] += ep_data[i]

        # Determine averages
        for i in range(len(data)):
            data[i] = data[i] / total
        return data

    def save(self, file):
        file.write(self.episodes[0].csv_header())
        for episode in self.episodes:
            if episode.steps >= self.min_steps:
                file.write("\n")
                file.write(",".join(str(x) for x in episode.csv_data()))
        file.write("\n")
        file.write(",".join(str(x) for x in self.results()))


class EpisodeData:

    def __init__(self):
        self.steps = 0
        self.offroad_counter = 0.0
        self.otherlane_counter = 0.0
        self.was_collision = False
        self.max_speed = 0
        self.speed_counter = 0.0
        self.is_out_of_lane = False
        self.out_of_lane_instances = 0
        self.out_of_lane_cooldown = 0
        self.reward = 0

    def step(self, py_measurements):
        # Simple measurements
        self.steps += 1
        self.offroad_counter += py_measurements["intersection_offroad"]
        self.otherlane_counter += py_measurements["intersection_otherlane"]
        self.was_collision = self.was_collision or (py_measurements["collision_vehicles"] > 0
                                                    or py_measurements["collision_pedestrians"] > 0
                                                    or py_measurements["collision_other"] > 0)
        speed = py_measurements["forward_speed"] * 3.8
        self.speed_counter += speed
        self.max_speed = max(self.max_speed, speed)
        self.reward += py_measurements["reward"]

        # Out of lane counter
        is_out_of_lane = py_measurements["intersection_offroad"] > 0.05 \
                         or py_measurements["intersection_otherlane"] > 0.05
        if is_out_of_lane and self.out_of_lane_cooldown <= 0:
            self.out_of_lane_cooldown = OUT_LANE_RESET + 1
            self.out_of_lane_instances += 1
        if self.out_of_lane_cooldown > 0:
            self.out_of_lane_cooldown -= 1
        self.is_out_of_lane = is_out_of_lane

    def csv_header(self):
        return "Steps,Offroad Percent,Otherlane Percent,Was Collision,Average Speed,Max Speed,OOL Instances,Reward"

    def csv_data(self):
        return [
            self.steps,
            self.offroad_counter / self.steps,
            self.otherlane_counter / self.steps,
            1.0 if self.was_collision else 0.0,
            self.speed_counter / self.steps,
            self.max_speed,
            self.out_of_lane_instances,
            self.reward]
