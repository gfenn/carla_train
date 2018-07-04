"""OpenAI gym environment for Carla. Run this file for a demo."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import atexit
import cv2
import os
import json
import random
import signal
import subprocess
import time
import traceback

import numpy as np
try:
    import scipy.misc
except Exception:
    pass

import gym
from gym.spaces import Box, Discrete, Tuple

from carla_env.scenarios import DEFAULT_SCENARIO
from carla_env.rewards import compute_reward
import carla_env.termination as TERM

SERVER_BINARY = os.environ.get("CARLA_SERVER", os.path.expanduser("~/CARLA_0.7.0/CarlaUE4.sh"))
if not os.path.exists(SERVER_BINARY):
    print("Since no $CARLA_SERVER -> CarlaUE4.sh binary exists, env.config['server_binary'] is no longer optional set.")

# Set this where you want to save image outputs (or empty string to disable)
CARLA_OUT_PATH = os.environ.get("CARLA_OUT", os.path.expanduser("/tmp/carla_out"))
if CARLA_OUT_PATH and not os.path.exists(CARLA_OUT_PATH):
    os.makedirs(CARLA_OUT_PATH)

try:
    from carla.client import CarlaClient
    from carla.sensor import Camera
    from carla.settings import CarlaSettings
    from carla.planner.planner import Planner, REACH_GOAL, GO_STRAIGHT, \
        TURN_RIGHT, TURN_LEFT, LANE_FOLLOW
except Exception as e:
    print("Failed to import Carla python libs.")
    raise e

# Carla planner commands
COMMANDS_ENUM = {
    REACH_GOAL: "REACH_GOAL",
    GO_STRAIGHT: "GO_STRAIGHT",
    TURN_RIGHT: "TURN_RIGHT",
    TURN_LEFT: "TURN_LEFT",
    LANE_FOLLOW: "LANE_FOLLOW",
}

# Mapping from string repr to one-hot encoding index to feed to the model
COMMAND_ORDINAL = {
    "REACH_GOAL": 0,
    "GO_STRAIGHT": 1,
    "TURN_RIGHT": 2,
    "TURN_LEFT": 3,
    "LANE_FOLLOW": 4,
}

# Number of retries if the server doesn't respond
RETRIES_ON_ERROR = 5

# Dummy Z coordinate to use when we only care about (x, y)
GROUND_Z = 22

# Default environment configuration
ENV_CONFIG = {
    "server_binary": SERVER_BINARY,
    "carla_out_path": CARLA_OUT_PATH,
    "log_images": True,
    "enable_planner": True,
    "framestack": 2,  # note: only [1, 2] currently supported
    "convert_images_to_video": True,
    TERM.EARLY_TERMINATIONS: [TERM.TERMINATE_ON_COLLISION, TERM.TERMINATE_ON_LEAVE_BOUNDS],
    "verbose": True,
    "reward_function": "lane_keep",
    "render_x_res": 800,
    "render_y_res": 600,
    "x_res": 80,
    "y_res": 80,
    "server_map": "/Game/Maps/Town02",
    "scenarios": [DEFAULT_SCENARIO],
    "use_depth_camera": False,
    "discrete_actions": True,
    "squash_action_logits": False,
}


DISCRETE_ACTIONS = {
    # coast
    0: [0.0, 0.0],
    # turn left
    1: [0.0, -0.5],
    # turn right
    2: [0.0, 0.5],
    # forward
    3: [1.0, 0.0],
    # brake
    4: [-0.5, 0.0],
    # forward left
    5: [1.0, -0.5],
    # forward right
    6: [1.0, 0.5],
    # brake left
    7: [-0.5, -0.5],
    # brake right
    8: [-0.5, 0.5],
}


live_carla_processes = set()


def cleanup():
    print("Killing live carla processes", live_carla_processes)
    for pgid in live_carla_processes:
        os.killpg(pgid, signal.SIGKILL)


atexit.register(cleanup)


class CarlaEnv(gym.Env):

    def __init__(self, config=ENV_CONFIG):
        self.config = config
        self.city = self.config["server_map"].split("/")[-1]
        if self.config["enable_planner"]:
            self.planner = Planner(self.city)

        if config["discrete_actions"]:
            self.action_space = Discrete(len(DISCRETE_ACTIONS))
        else:
            self.action_space = Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        if config["use_depth_camera"]:
            image_space = Box(
                -1.0, 1.0, shape=(
                    config["y_res"], config["x_res"],
                    1 * config["framestack"]), dtype=np.float32)
        else:
            image_space = Box(
                0, 255, shape=(
                    config["y_res"], config["x_res"],
                    3 * config["framestack"]), dtype=np.uint8)
        # self.observation_space = Tuple(  # forward_speed, dist to goal
        #     [image_space,
        #      Discrete(len(COMMANDS_ENUM)),  # next_command
        #      Box(-128.0, 128.0, shape=(2,), dtype=np.float32)])
        self.observation_space = image_space

        # TODO(ekl) this isn't really a proper gym spec
        self._spec = lambda: None
        self._spec.id = "Carla-v0"

        self.server_port = None
        self.server_process = None
        self.client = None
        self.num_steps = 0
        self.total_reward = 0
        self.prev_measurement = None
        self.prev_image = None
        self.episode_id = None
        self.measurements_file = None
        self.weather = None
        self.scenario = None
        self.start_pos = None
        self.end_pos = None
        self.start_coord = None
        self.end_coord = None
        self.last_obs = None

    def init_server(self):
        print("Initializing new Carla server...")
        # Create a new server process and start the client.
        self.server_port = random.randint(10000, 60000)
        self.server_process = subprocess.Popen(
            [self.config["server_binary"], self.config["server_map"],
             "-windowed", "-ResX=800", "-ResY=600",
             "-carla-server",
             "-carla-world-port={}".format(self.server_port)],
            preexec_fn=os.setsid, stdout=open(os.devnull, "w"))
        live_carla_processes.add(os.getpgid(self.server_process.pid))

        for i in range(RETRIES_ON_ERROR):
            try:
                self.client = CarlaClient("localhost", self.server_port)
                return self.client.connect()
            except Exception as e:
                print("Error connecting: {}, attempt {}".format(e, i))
                time.sleep(2)

    def clear_server_state(self):
        print("Clearing Carla server state")
        try:
            if self.client:
                self.client.disconnect()
                self.client = None
        except Exception as e:
            print("Error disconnecting client: {}".format(e))
            pass
        if self.server_process:
            pgid = os.getpgid(self.server_process.pid)
            os.killpg(pgid, signal.SIGKILL)
            live_carla_processes.remove(pgid)
            self.server_port = None
            self.server_process = None

    def __del__(self):
        self.clear_server_state()

    def reset(self):
        error = None
        for _ in range(RETRIES_ON_ERROR):
            try:
                if not self.server_process:
                    self.init_server()
                return self._reset()
            except Exception as e:
                print("Error during reset: {}".format(traceback.format_exc()))
                self.clear_server_state()
                error = e
        raise error

    def _reset(self):
        self.num_steps = 0
        self.total_reward = 0
        self.prev_measurement = None
        self.prev_image = None
        self.episode_id = datetime.today().strftime("%Y-%m-%d_%H-%M-%S_%f")
        self.measurements_file = None

        # Create a CarlaSettings object. This object is a wrapper around
        # the CarlaSettings.ini file. Here we set the configuration we
        # want for the new episode.
        settings = CarlaSettings()
        self.scenario = random.choice(self.config["scenarios"])
        assert self.scenario["city"] == self.city, (self.scenario, self.city)
        self.weather = random.choice(self.scenario["weather_distribution"])
        settings.set(
            SynchronousMode=True,
            SendNonPlayerAgentsInfo=True,
            NumberOfVehicles=self.scenario["num_vehicles"],
            NumberOfPedestrians=self.scenario["num_pedestrians"],
            WeatherId=self.weather)
        settings.randomize_seeds()

        if self.config["use_depth_camera"]:
            camera1 = Camera("CameraDepth", PostProcessing="Depth")
            camera1.set_image_size(
                self.config["render_x_res"], self.config["render_y_res"])
            camera1.set_position(0.30, 0, 1.30)
            settings.add_sensor(camera1)

        camera2 = Camera("CameraRGB")
        camera2.set_image_size(
            self.config["render_x_res"], self.config["render_y_res"])
        camera2.set_position(0.30, 0, 1.30)
        settings.add_sensor(camera2)

        # Setup start and end positions
        scene = self.client.load_settings(settings)
        positions = scene.player_start_spots
        self.start_pos = positions[self.scenario["start_pos_id"]]
        self.end_pos = positions[self.scenario["end_pos_id"]]
        self.start_coord = [
            self.start_pos.location.x // 100, self.start_pos.location.y // 100]
        self.end_coord = [
            self.end_pos.location.x // 100, self.end_pos.location.y // 100]
        print(
            "Start pos {} ({}), end {} ({})".format(
                self.scenario["start_pos_id"], self.start_coord,
                self.scenario["end_pos_id"], self.end_coord))

        # Notify the server that we want to start the episode at the
        # player_start index. This function blocks until the server is ready
        # to start the episode.
        print("Starting new episode...")
        self.client.start_episode(self.scenario["start_pos_id"])

        image, py_measurements = self._read_observation()
        self.prev_measurement = py_measurements
        return self.encode_obs(self.preprocess_image(image), py_measurements)

    def encode_obs(self, image, py_measurements):
        assert self.config["framestack"] in [1, 2]
        prev_image = self.prev_image
        self.prev_image = image
        if prev_image is None:
            prev_image = image
        if self.config["framestack"] == 2:
            image = np.concatenate([prev_image, image], axis=2)
        # obs = (
        #     image,
        #     COMMAND_ORDINAL[py_measurements["next_command"]],
        #     [py_measurements["forward_speed"],
        #      py_measurements["distance_to_goal"]])
        obs = image
        self.last_obs = obs
        return obs

    def step(self, action):
        try:
            obs = self._step(action)
            return obs
        except Exception:
            print(
                "Error during step, terminating episode early",
                traceback.format_exc())
            self.clear_server_state()
            return (self.last_obs, 0.0, True, {})

    def _step(self, action):
        if self.config["discrete_actions"]:
            action = DISCRETE_ACTIONS[int(action)]
        assert len(action) == 2, "Invalid action {}".format(action)
        if self.config["squash_action_logits"]:
            forward = 2 * float(sigmoid(action[0]) - 0.5)
            throttle = float(np.clip(forward, 0, 1))
            brake = float(np.abs(np.clip(forward, -1, 0)))
            steer = 2 * float(sigmoid(action[1]) - 0.5)
        else:
            throttle = float(np.clip(action[0], 0, 1))
            brake = float(np.abs(np.clip(action[0], -1, 0)))
            steer = float(np.clip(action[1], -1, 1))
        reverse = False
        hand_brake = False

        if self.config["verbose"]:
            print(
                "steer", steer, "throttle", throttle, "brake", brake,
                "reverse", reverse)

        self.client.send_control(
            steer=steer, throttle=throttle, brake=brake, hand_brake=hand_brake,
            reverse=reverse)

        # Process observations
        image, py_measurements = self._read_observation()
        if self.config["verbose"]:
            print("Next command", py_measurements["next_command"])
        if type(action) is np.ndarray:
            py_measurements["action"] = [float(a) for a in action]
        else:
            py_measurements["action"] = action
        py_measurements["control"] = {
            "steer": steer,
            "throttle": throttle,
            "brake": brake,
            "reverse": reverse,
            "hand_brake": hand_brake,
        }
        reward = compute_reward(
            self, self.prev_measurement, py_measurements)
        self.total_reward += reward
        py_measurements["reward"] = reward
        py_measurements["total_reward"] = self.total_reward
        done = (self.num_steps > self.scenario["max_steps"] or
                py_measurements["next_command"] == "REACH_GOAL" or
                TERM.compute_termination(self, py_measurements))
        py_measurements["done"] = done
        self.prev_measurement = py_measurements

        # Write out measurements to file
        if self.config["carla_out_path"]:
            if not self.measurements_file:
                self.measurements_file = open(
                    os.path.join(
                        self.config["carla_out_path"],
                        "measurements_{}.json".format(self.episode_id)),
                    "w")
            self.measurements_file.write(json.dumps(py_measurements))
            self.measurements_file.write("\n")
            if done:
                self.measurements_file.close()
                self.measurements_file = None
                if self.config["convert_images_to_video"]:
                    self.images_to_video()

        self.num_steps += 1
        image = self.preprocess_image(image)
        return (
            self.encode_obs(image, py_measurements), reward, done,
            py_measurements)

    def images_to_video(self):
        videos_dir = os.path.join(self.config["carla_out_path"], "Videos")
        if not os.path.exists(videos_dir):
            os.makedirs(videos_dir)
        ffmpeg_cmd = (
            "ffmpeg -loglevel -8 -r 60 -f image2 -s {x_res}x{y_res} "
            "-start_number 0 -i "
            "{img}_%04d.jpg -vcodec libx264 {vid}.mp4 && rm -f {img}_*.jpg "
        ).format(
            x_res=self.config["render_x_res"],
            y_res=self.config["render_y_res"],
            vid=os.path.join(videos_dir, self.episode_id),
            img=os.path.join(self.config["carla_out_path"], "CameraRGB", self.episode_id))

        print("Executing ffmpeg command", ffmpeg_cmd)
        try:
            subprocess.call(ffmpeg_cmd, shell=True, timeout=20)
        except Exception:
            print("FFMPEG EXPIRED")

    def preprocess_image(self, image):
        if self.config["use_depth_camera"]:
            assert self.config["use_depth_camera"]
            data = (image.data - 0.5) * 2
            data = data.reshape(
                self.config["render_y_res"], self.config["render_x_res"], 1)
            data = cv2.resize(
                data, (self.config["x_res"], self.config["y_res"]),
                interpolation=cv2.INTER_AREA)
            data = np.expand_dims(data, 2)
        else:
            data = image.data.reshape(
                self.config["render_y_res"], self.config["render_x_res"], 3)
            data = cv2.resize(
                data, (self.config["x_res"], self.config["y_res"]),
                interpolation=cv2.INTER_AREA)
            data = (data.astype(np.float32) - 128) / 128
        return data

    def _read_observation(self):
        # Read the data produced by the server this frame.
        measurements, sensor_data = self.client.read_data()

        # Print some of the measurements.
        if self.config["verbose"]:
            print_measurements(measurements)

        observation = None
        if self.config["use_depth_camera"]:
            camera_name = "CameraDepth"
        else:
            camera_name = "CameraRGB"
        for name, image in sensor_data.items():
            if name == camera_name:
                observation = image

        cur = measurements.player_measurements

        if self.config["enable_planner"]:
            next_command = COMMANDS_ENUM[
                self.planner.get_next_command(
                    [cur.transform.location.x, cur.transform.location.y,
                     GROUND_Z],
                    [cur.transform.orientation.x, cur.transform.orientation.y,
                     GROUND_Z],
                    [self.end_pos.location.x, self.end_pos.location.y,
                     GROUND_Z],
                    [self.end_pos.orientation.x, self.end_pos.orientation.y,
                     GROUND_Z])
            ]
        else:
            next_command = "LANE_FOLLOW"

        if next_command == "REACH_GOAL":
            distance_to_goal = 0.0  # avoids crash in planner
        elif self.config["enable_planner"]:
            distance_to_goal = self.planner.get_shortest_path_distance(
                [cur.transform.location.x, cur.transform.location.y, GROUND_Z],
                [cur.transform.orientation.x, cur.transform.orientation.y,
                 GROUND_Z],
                [self.end_pos.location.x, self.end_pos.location.y, GROUND_Z],
                [self.end_pos.orientation.x, self.end_pos.orientation.y,
                 GROUND_Z]) / 100
        else:
            distance_to_goal = -1

        distance_to_goal_euclidean = float(np.linalg.norm(
            [cur.transform.location.x - self.end_pos.location.x,
             cur.transform.location.y - self.end_pos.location.y]) / 100)

        py_measurements = {
            "episode_id": self.episode_id,
            "step": self.num_steps,
            "x": cur.transform.location.x,
            "y": cur.transform.location.y,
            "x_orient": cur.transform.orientation.x,
            "y_orient": cur.transform.orientation.y,
            "forward_speed": cur.forward_speed,
            "distance_to_goal": distance_to_goal,
            "distance_to_goal_euclidean": distance_to_goal_euclidean,
            "collision_vehicles": cur.collision_vehicles,
            "collision_pedestrians": cur.collision_pedestrians,
            "collision_other": cur.collision_other,
            "intersection_offroad": cur.intersection_offroad,
            "intersection_otherlane": cur.intersection_otherlane,
            "weather": self.weather,
            "map": self.config["server_map"],
            "start_coord": self.start_coord,
            "end_coord": self.end_coord,
            "current_scenario": self.scenario,
            "x_res": self.config["x_res"],
            "y_res": self.config["y_res"],
            "num_vehicles": self.scenario["num_vehicles"],
            "num_pedestrians": self.scenario["num_pedestrians"],
            "max_steps": self.scenario["max_steps"],
            "next_command": next_command,
        }

        if self.config["carla_out_path"] and self.config["log_images"]:
            for name, image in sensor_data.items():
                out_dir = os.path.join(self.config["carla_out_path"], name)
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)
                out_file = os.path.join(
                    out_dir,
                    "{}_{:>04}.jpg".format(self.episode_id, self.num_steps))
                scipy.misc.imsave(out_file, image.data)

        assert observation is not None, sensor_data
        return observation, py_measurements


def print_measurements(measurements):
    number_of_agents = len(measurements.non_player_agents)
    player_measurements = measurements.player_measurements
    message = "Vehicle at ({pos_x:.1f}, {pos_y:.1f}), "
    message += "{speed:.2f} km/h, "
    message += "Collision: {{vehicles={col_cars:.0f}, "
    message += "pedestrians={col_ped:.0f}, other={col_other:.0f}}}, "
    message += "{other_lane:.0f}% other lane, {offroad:.0f}% off-road, "
    message += "({agents_num:d} non-player agents in the scene)"
    message = message.format(
        pos_x=player_measurements.transform.location.x / 100,  # cm -> m
        pos_y=player_measurements.transform.location.y / 100,
        speed=player_measurements.forward_speed,
        col_cars=player_measurements.collision_vehicles,
        col_ped=player_measurements.collision_pedestrians,
        col_other=player_measurements.collision_other,
        other_lane=100 * player_measurements.intersection_otherlane,
        offroad=100 * player_measurements.intersection_offroad,
        agents_num=number_of_agents)
    print(message)


def sigmoid(x):
    x = float(x)
    return np.exp(x) / (1 + np.exp(x))


if __name__ == "__main__":
    for _ in range(2):
        env = CarlaEnv()
        obs = env.reset()
        print("reset", obs)
        start = time.time()
        done = False
        i = 0
        total_reward = 0.0
        while not done:
            i += 1
            if ENV_CONFIG["discrete_actions"]:
                obs, reward, done, info = env.step(1)
            else:
                obs, reward, done, info = env.step([0, 1, 0])
            total_reward += reward
            print(i, "rew", reward, "total", total_reward, "done", done)
        print("{} fps".format(100 / (time.time() - start)))
