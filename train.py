import os
from carla_env.env import CarlaEnv, ENV_CONFIG
import carla_env.scenarios as scenarios
import carla_env.termination as terminations
import carla_env.rewards as rewards
from baselines import deepq
import deepq_learner


def callback(lcl, _glb):
    # Can do things and stuff each frame
    pass


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
        "x_res": 80,
        "y_res": 80,
        "use_depth_camera": False,
        "discrete_actions": True,
        "server_map": "/Game/Maps/Town02",
        "reward_function": rewards.REWARD_LANE_KEEP,
        "enable_planner": False,
        terminations.EARLY_TERMINATIONS: [
            terminations.TERMINATE_ON_COLLISION,
            terminations.TERMINATE_ON_LEAVE_BOUNDS],
        "scenarios": scenarios.TOWN2_LANE_KEEP,
    })
    env = CarlaEnv(env_config)

    # Create an OpenAI-deepq baseline
    model = deepq.models.cnn_to_mlp(
        convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
        hiddens=[512],
        dueling=True
    )

    # Learn
    act = deepq_learner.learn(
        env,
        gpu_memory_fraction=0.7,
        q_func=model,
        lr=1e-4,
        max_timesteps=int(1e6),
        buffer_size=int(5e4),
        exploration_fraction=0.1,
        exploration_final_eps=0.02,
        train_freq=4,
        learning_starts=1000,
        target_network_update_freq=1000,
        gamma=0.9,
        prioritized_replay=True,
        prioritized_replay_alpha=0.6,
        checkpoint_freq=1,
        checkpoint_path=checkpoint_path,
        print_freq=1,
        callback=callback
    )
    env.close()

    # Save the file

    print("Saving model to " + model_save_path)
    act.save(model_save_path)


if __name__ == '__main__':
    main()
