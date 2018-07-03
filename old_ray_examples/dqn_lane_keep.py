from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import ray
from ray.tune import register_env, run_experiments

os.environ["CARLA_OUT"] = "/media/grant/FastData/carla_out"

from env import CarlaEnv, ENV_CONFIG
from models import register_carla_model
from scenarios import TOWN2_LANE_KEEP

env_name = "carla_env"
env_config = ENV_CONFIG.copy()
env_config.update({
    "verbose": False,
    "x_res": 80,
    "y_res": 80,
    "use_depth_camera": False,
    "discrete_actions": True,
    "server_map": "/Game/Maps/Town02",
    "reward_function": "lane_keep",
    "enable_planner": False,
    "scenarios": TOWN2_LANE_KEEP,
})

register_env(env_name, lambda env_config: CarlaEnv(env_config))
register_carla_model()

ray.init()
run_experiments({
    "carla-dqn": {
        "run": "DQN",
        "env": "carla_env",
        "local_dir": "/media/grant/FastData/carla_out",
        "trial_resources": {"cpu": 4, "gpu": 1},
        "checkpoint_freq": 10,
        "max_failures": 2,
        # "gpu": True,
        "config": {
            "env_config": env_config,
            "model": {
                "custom_model": "carla",
                "custom_options": {
                    "image_shape": [80, 80, 6],
                },
                "conv_filters": [
                    [16, [8, 8], 4],
                    [32, [4, 4], 2],
                    [512, [10, 10], 1],
                ],
            },
            "timesteps_per_iteration": 100,
            "learning_starts": 1000,
            "schedule_max_timesteps": 100000,
            "exploration_fraction": 0.5,
            "exploration_final_eps": 0.1,
            "gamma": 0.95,
            "tf_session_args": {
              "gpu_options": {"allow_growth": True},
            },
        },
    },
})
