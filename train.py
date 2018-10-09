import os
from carla_env.env import CarlaEnv, ENV_CONFIG
import carla_env.scenarios as scenarios
import carla_env.termination as terminations
import carla_env.rewards as rewards
from baselines import deepq
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
    "log_images": False,
    "convert_images_to_video": False,
    "render_x_res": 228,
    "render_y_res": 228,
    "x_res": 228,
    "y_res": 228,
    "fps": 50,
    "use_depth_camera": False,
    "server_map": "/Game/Maps/Town02",
    "reward_function": rewards.REWARD_LANE_KEEP,
    "enable_planner": False,
    "framestack": 3,
    terminations.EARLY_TERMINATIONS: [
        terminations.TERMINATE_ON_COLLISION,
        terminations.TERMINATE_ON_OFFROAD,
        # terminations.TERMINATE_ON_OTHERLANE,
        terminations.TERMINATE_NO_MOVEMENT],
    "scenarios": scenarios.TOWN2_LANE_KEEP,
})


# Create an OpenAI-deepq baseline
TRAIN_MODEL = deepq.models.cnn_to_mlp(
    convs=[(32, 3, 2), (32, 3, 2),
           (64, 3, 2), (64, 3, 2),
           (128, 3, 1), (128, 3, 1),
           (256, 3, 1), (256, 3, 1)],
    hiddens=[1024, 1024],
    dueling=True
)


def main():
    # Build env
    env = CarlaEnv(TRAIN_CONFIG)

    # Determine paths
    model_save_path = os.path.join(carla_out_path, "model.pkl")
    checkpoint_path = os.path.join(carla_out_path, "checkpoints")
    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)

    # Learn
    learn_config = deepq_learner.DEEPQ_CONFIG.copy()
    learn_config.update({
        "gpu_memory_fraction": 0.7,
        "lr": 1e-5,
        "max_timesteps": int(2e6),
        "buffer_size": int(8000),
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
    learn = deepq_learner.DeepqLearner(env=env, q_func=TRAIN_MODEL, config=learn_config)
    learn.run()

    env.close()

    # Save the file
    learn.save(model_save_path)


if __name__ == '__main__':
    main()
