import os
from carla_env.env import CarlaEnv, ENV_CONFIG
import carla_env.scenarios as scenarios
import carla_env.termination as terminations
import carla_env.rewards as rewards
from example_collector import EpisodeCollector
from baselines import deepq
import deepq_learner


MAX_EPISODES = 10

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
        "log_images": False,
        "convert_images_to_video": False,
        "x_res": 120,
        "y_res": 120,
        "use_depth_camera": False,
        "server_map": "/Game/Maps/Town01",
        "reward_function": rewards.REWARD_LANE_KEEP,
        "enable_planner": False,
        "framestack": 2,
        terminations.EARLY_TERMINATIONS: [
            terminations.TERMINATE_ON_COLLISION,
            terminations.TERMINATE_ON_OFFROAD,
            # terminations.TERMINATE_ON_OTHERLANE,
            terminations.TERMINATE_NO_MOVEMENT],
        "scenarios": scenarios.TOWN1_LANE_KEEP,
    })

    collector = EpisodeCollector()
    def on_step(py_measurements):
        collector.step(py_measurements)
    def on_next():
        collector.next()
        if collector.valid_episodes >= MAX_EPISODES:
            raise Exception("Done!")

    env = CarlaEnv(env_config)
    env.on_step = on_step
    env.on_next = on_next

    # Create an OpenAI-deepq baseline
    model = deepq.models.cnn_to_mlp(
        convs=[(32, 3, 2), (32, 3, 2), (64, 5, 2), (128, 5, 1)],
        hiddens=[512, 512],
        dueling=True
    )

    # Learn
    learn_config = deepq_learner.DEEPQ_CONFIG.copy()
    learn_config.update({
        "gpu_memory_fraction": 0.5,
        "lr": 1e-60,
        "max_timesteps": int(1e8),
        "buffer_size": int(1e3),
        "exploration_fraction": 0.000001,
        "exploration_final_eps": 0.000001,
        "train_freq": 4000000,
        "learning_starts": 1000000,
        "target_network_update_freq": 10000000,
        "gamma": 0.99,
        "prioritized_replay": True,
        "prioritized_replay_alpha": 0.6,
        "checkpoint_freq": 100000000,
        "checkpoint_path": checkpoint_path,
        "print_freq": 1
    })
    learn = deepq_learner.DeepqLearner(env=env, q_func=model, config=learn_config)

    print("Running training....")
    try:
        learn.run()
    except Exception as e:
        print("Training Failed!")
        print(e)

    print("Closing environment.")
    env.close()

    # Save the file
    learn.save(model_save_path)

    # Determine results
    results = collector.results()
    print(",".join(str(x) for x in results))
    with open(carla_out_path + '/results.csv', 'w') as file:
        collector.save(file)

if __name__ == '__main__':
    main()
