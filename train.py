import os
from carla_env.env import CarlaEnv, ENV_CONFIG
import carla_env.scenarios as scenarios
import carla_env.termination as terminations
import carla_env.rewards as rewards
from baselines import deepq
from model_builder import cnn_to_mlp
import deepq_learner

# Filenames
carla_out_path = "/media/grant/FastData/carla"
if not os.path.exists(carla_out_path):
    os.mkdir(carla_out_path)

# Build the OpenAI-gym ready environment
TRAIN_CONFIG = ENV_CONFIG.copy()
TRAIN_CONFIG.update({
    "verbose": False,
    "carla_out_path": carla_out_path,
    "save_images_rgb": False,
    "save_image_frequency": 10,
    "convert_images_to_video": False,
    "render_x_res": 258,
    "render_y_res": 258,
    "x_res": 258,
    "y_res": 258,
    "fps": 50,
    # "quality": "Epic",
    "quality": "Low",
    "use_depth_camera": False,
    "server_map": "/Game/Maps/Town02",
    "reward_function": rewards.REWARD_LANE_KEEP,
    "enable_planner": False,
    "framestack": 2,
    terminations.EARLY_TERMINATIONS: [
        # terminations.TERMINATE_ON_COLLISION,
        # terminations.TERMINATE_ON_OFFROAD,
        terminations.TERMINATE_NO_MOVEMENT,
        # terminations.TERMINATE_NOT_PERFECT
    ],
    "scenarios": scenarios.TOWN2_LANE_KEEP,
})


# Create an OpenAI-deepq baseline
MODEL_CONVS = [
    {"num_outputs": 32, "kernel_size": 3, "stride": 2},
    {"num_outputs": 32, "kernel_size": 3, "stride": 2, "max_pool": {"size": [2, 2], "stride": 2}},
    {"num_outputs": 64, "kernel_size": 3, "stride": 1},
    {"num_outputs": 64, "kernel_size": 3, "stride": 1, "max_pool": {"size": [2, 2], "stride": 2}},
    {"num_outputs": 128, "kernel_size": 3, "stride": 1},
    {"num_outputs": 128, "kernel_size": 3, "stride": 1, "max_pool": {"size": [2, 2], "stride": 2}},
]
MODEL_HIDDEN = [
    {"neurons": 2048, "dropout": 0.5},
    {"neurons": 2048}
]
MODEL_DUELING = True


def main():
    # Build env
    env = CarlaEnv(TRAIN_CONFIG)
    train_model = cnn_to_mlp(
        convs=MODEL_CONVS,
        hiddens=MODEL_HIDDEN,
        dueling=MODEL_DUELING,
        is_training=True
    )

    # Determine paths
    model_save_path = os.path.join(carla_out_path, "model.pkl")
    checkpoint_path = os.path.join(carla_out_path, "checkpoints")
    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)

    # Learn
    learn_config = deepq_learner.DEEPQ_CONFIG.copy()
    learn_config.update({
        "gpu_memory_fraction": 0.4,
        "lr": 1e-4,
        "max_timesteps": int(1e6),
        "buffer_size": int(1e4),
        "exploration_fraction": 0.1,
        "exploration_final_eps": 0.1,
        "train_freq": 4,
        "learning_starts": 100,
        "target_network_update_freq": 1000,
        "gamma": 0.99,
        "prioritized_replay": True,
        "prioritized_replay_alpha": 0.7,
        "checkpoint_freq": 10,
        "checkpoint_path": checkpoint_path,
        "print_freq": 1
    })
    learn = deepq_learner.DeepqLearner(env=env, q_func=train_model, config=learn_config)
    learn.run()

    env.close()

    # Save the file
    learn.save(model_save_path)


if __name__ == '__main__':
    main()
