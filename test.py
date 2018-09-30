import os
from carla_env.env import CarlaEnv
import carla_env.scenarios as scenarios
import carla_env.rewards as rewards
from example_collector import EpisodeCollector
from train import TRAIN_CONFIG, TRAIN_MODEL
import deepq_learner


MAX_EPISODES = 1000

class DoneError(BaseException):
    pass


# Update the environment for testing components
TEST_ENV = TRAIN_CONFIG.copy()
TEST_ENV.update({
    "server_map": "/Game/Maps/Town01",
    "reward_function": rewards.REWARD_LANE_KEEP,
    "scenarios": scenarios.TOWN1_LANE_KEEP,
})


def main():
    collector = EpisodeCollector()
    def on_step(py_measurements):
        collector.step(py_measurements)
    def on_next():
        collector.next()
        if collector.valid_episodes >= MAX_EPISODES:
            raise DoneError()

    env = CarlaEnv(TEST_ENV)
    env.on_step = on_step
    env.on_next = on_next

    carla_out_path = "/media/grant/FastData/carla"
    if not os.path.exists(carla_out_path):
        os.mkdir(carla_out_path)
    checkpoint_path = os.path.join(carla_out_path, "checkpoints")
    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)

    # Learn
    learn_config = deepq_learner.DEEPQ_CONFIG.copy()
    learn_config.update({
        "gpu_memory_fraction": 0.5,
        "lr": 1e-90,
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
    learn = deepq_learner.DeepqLearner(env=env, q_func=TRAIN_MODEL, config=learn_config)

    print("Running training....")
    try:
        learn.run()
    except DoneError:
        pass
    except Exception as e:
        print("Training Failed!")
        raise e
    finally:
        print("Closing environment.")
        env.close()

        # Determine results
        results = collector.results()
        print(",".join(str(x) for x in results))
        with open(carla_out_path + '/results.csv', 'w') as file:
            collector.save(file)

if __name__ == '__main__':
    main()
