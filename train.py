import os
from carla_env.env import CarlaEnv, ENV_CONFIG
import carla_env.scenarios as scenarios
import carla_env.termination as terminations
import carla_env.rewards as rewards
from baselines import deepq
import deepq_learner


def main():
    # Filenames
    carla_out_path = "/media/grant/FastData/carla"
    if not os.path.exists(carla_out_path):
        os.mkdir(carla_out_path)
    checkpoint_path = os.path.join(carla_out_path, "checkpoints")
    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)
    model_save_path = os.path.join(carla_out_path, "model.pkl")

    # Build the OpenAI-gym ready environment
    env_config = ENV_CONFIG.copy()
    env_config.update({
        "verbose": False,
        "carla_out_path": carla_out_path,
        "x_res": 160,
        "y_res": 160,
        "use_depth_camera": False,
        "discrete_actions": True,
        "server_map": "/Game/Maps/Town02",
        "reward_function": rewards.REWARD_LANE_KEEP,
        "enable_planner": False,
        "framestack": 8,
        terminations.EARLY_TERMINATIONS: [
            terminations.TERMINATE_ON_COLLISION,
            terminations.TERMINATE_ON_OFFROAD,
            terminations.TERMINATE_ON_OTHERLANE],
        "scenarios": scenarios.TOWN2_LANE_KEEP,
    })
    env = CarlaEnv(env_config)

    # Create an OpenAI-deepq baseline
    model = deepq.models.cnn_to_mlp(
        convs=[(32, 4, 2), (64, 3, 2), (128, 3, 2), (256, 10, 2)],
        hiddens=[512],
        dueling=True
    )

    # Learn
    learn_config = deepq_learner.DEEPQ_CONFIG.copy()
    learn_config.update({
        "gpu_memory_fraction": 0.55,
        "lr": 1e-4,
        "max_timesteps": int(1e6),
        "buffer_size": int(1e4),
        "exploration_fraction": 0.1,
        "exploration_final_eps": 0.05,
        "train_freq": 4,
        "learning_starts": 1000,
        "target_network_update_freq": 1000,
        "gamma": 0.99,
        "prioritized_replay": True,
        "prioritized_replay_alpha": 0.6,
        "checkpoint_freq": 10,
        "checkpoint_path": checkpoint_path,
        "print_freq": 1
    })
    learn = deepq_learner.DeepqLearner(env=env, q_func=model, config=learn_config)
    learn.run()

    env.close()

    # Save the file
    learn.save(model_save_path)


if __name__ == '__main__':
    main()
