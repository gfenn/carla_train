import os
from carla_env.env import CarlaEnv
import carla_env.scenarios as scenarios
import carla_env.rewards as rewards
from train import TRAIN_CONFIG, TRAIN_MODEL
import deepq_learner


MAX_EPISODES = 10

class DoneError(BaseException):
    pass


# Update the environment for testing components
RECORD_ENV = TRAIN_CONFIG.copy()
RECORD_ENV.update({
    "server_map": "/Game/Maps/Town01",
    "reward_function": rewards.REWARD_LANE_KEEP,
    "scenarios": scenarios.TOWN1_LANE_KEEP,
    "save_images_rgb": True,
    "save_images_class": True,
    "save_images_fusion": False,
    "save_image_frequency": 1,
    "convert_images_to_video": False,
    # "render_x_res": 632,
    # "render_y_res": 632,
    "quality": "Epic"
})


class FinishedCompleter:
    def __init__(self):
        self.episodes = 0

    def on_next(self):
        self.episodes += 1
        if self.episodes >= MAX_EPISODES:
            raise DoneError()


def main():
    completionCheck = FinishedCompleter()
    env = CarlaEnv(RECORD_ENV)
    env.on_next = completionCheck.on_next

    carla_out_path = "/media/grant/FastData/carla"
    if not os.path.exists(carla_out_path):
        os.mkdir(carla_out_path)
    checkpoint_path = os.path.join(carla_out_path, "checkpoints")
    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)

    # Learn
    learn_config = deepq_learner.DEEPQ_CONFIG.copy()
    learn_config.update({
        "gpu_memory_fraction": 0.7,
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

    print("Running Recording....")
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


if __name__ == '__main__':
    main()
