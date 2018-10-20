

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
        data = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
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

    def save_metrics_file(self, file):
        file.write(self.episodes[0].metrics_csv_header())
        for episode in self.episodes:
            if episode.steps >= self.min_steps:
                file.write("\n")
                file.write(",".join(str(x) for x in episode.metrics_csv_data()))
        file.write("\n")
        file.write(",".join(str(x) for x in self.results()))

    def save_crashes_file(self, file):
        file.write(self.episodes[0].crashes_csv_header())
        for episode in self.episodes:
            if episode.steps >= self.min_steps:
                crash = episode.crashes_csv_data()
                if crash is not None:
                    file.write("\n")
                    file.write(",".join(str(x) for x in crash))

    def save_out_of_lane_file(self, file):
        file.write(self.episodes[0].out_of_lane_csv_header())
        for episode in self.episodes:
            if episode.steps >= self.min_steps:
                instances = episode.out_of_lane_csv_data()
                for instance in instances:
                    file.write("\n")
                    file.write(",".join(str(x) for x in instance))


class EpisodeData:

    def __init__(self):
        self.steps = 0
        self.offroad_counter = 0.0
        self.otherlane_counter = 0.0
        self.collision_vehicle = 0
        self.collision_pedestrian = 0
        self.collision_other = 0
        self.max_speed = 0.0
        self.speed_counter = 0.0
        self.is_out_of_lane = False
        self.out_of_lane_instances = 0
        self.out_of_lane_cooldown = 0
        self.reward = 0
        self.crash_location = None
        self.out_of_lane_instances = []

    def step(self, py_measurements):
        # Simple measurements
        self.steps += 1
        self.offroad_counter += py_measurements["intersection_offroad"]
        self.otherlane_counter += py_measurements["intersection_otherlane"]
        if py_measurements["collision_vehicles"] > 0:
            self.collision_vehicle = True
        if py_measurements["collision_pedestrians"] > 0:
            self.collision_pedestrian = True
        if py_measurements["collision_other"] > 0:
            self.collision_other = True
        if self.collision_vehicle or self.collision_pedestrian or self.collision_other:
            self.crash_location = [py_measurements["x"], py_measurements["y"]]
        speed = py_measurements["forward_speed"] * 3.6
        self.speed_counter += speed
        self.max_speed = max(self.max_speed, speed)
        self.reward += py_measurements["reward"]

        # Out of lane counter
        is_out_of_lane = py_measurements["intersection_offroad"] > 0.05 \
                         or py_measurements["intersection_otherlane"] > 0.05
        if is_out_of_lane and self.out_of_lane_cooldown <= 0:
            self.out_of_lane_cooldown = OUT_LANE_RESET + 1
            self.out_of_lane_instances += 1
            self.out_of_lane_instances.append([py_measurements["x"], py_measurements["y"]])
        if self.out_of_lane_cooldown > 0:
            self.out_of_lane_cooldown -= 1
        self.is_out_of_lane = is_out_of_lane

    def metrics_csv_header(self):
        return "Steps,Offroad Percent,Otherlane Percent,Average Speed,Max Speed,OOL Instances," \
               "Collision - Vehicle,Collision - Pedestrian,Collision - Other,Reward"

    def metrics_csv_data(self):
        return [
            self.steps,
            self.offroad_counter / self.steps,
            self.otherlane_counter / self.steps,
            self.speed_counter / self.steps,
            self.max_speed,
            self.out_of_lane_instances,
            self.collision_vehicle,
            self.collision_pedestrian,
            self.collision_other,
            self.reward]

    def crashes_csv_header(self):
        return "X,Y"

    def crashes_csv_data(self):
        return self.crash_location

    def out_of_lane_csv_header(self):
        return "X,Y"

    def out_of_lane_csv_data(self):
        return self.out_of_lane_instances
