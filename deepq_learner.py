import os
import tempfile

import tensorflow as tf
import zipfile
import cloudpickle
import numpy as np

import baselines.common.tf_util as U
from baselines.common.tf_util import load_state, save_state
from baselines import logger
from baselines.common.schedules import LinearSchedule

from baselines import deepq
from baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from baselines.deepq.utils import ObservationInput
import time

class ActWrapper(object):
    def __init__(self, act, act_params):
        self._act = act
        self._act_params = act_params

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            model_data, act_params = cloudpickle.load(f)
        act = deepq.build_act(**act_params)
        sess = tf.Session()
        sess.__enter__()
        with tempfile.TemporaryDirectory() as td:
            arc_path = os.path.join(td, "packed.zip")
            with open(arc_path, "wb") as f:
                f.write(model_data)

            zipfile.ZipFile(arc_path, 'r', zipfile.ZIP_DEFLATED).extractall(td)
            load_state(os.path.join(td, "model"))

        return ActWrapper(act, act_params)

    def __call__(self, *args, **kwargs):
        return self._act(*args, **kwargs)

    def save(self, path=None):
        """Save model to a pickle located at `path`"""
        if path is None:
            path = os.path.join(logger.get_dir(), "model.pkl")

        with tempfile.TemporaryDirectory() as td:
            save_state(os.path.join(td, "model"))
            arc_name = os.path.join(td, "packed.zip")
            with zipfile.ZipFile(arc_name, 'w') as zipf:
                for root, dirs, files in os.walk(td):
                    for fname in files:
                        file_path = os.path.join(root, fname)
                        if file_path != arc_name:
                            zipf.write(file_path, os.path.relpath(file_path, td))
            with open(arc_name, "rb") as f:
                model_data = f.read()
        with open(path, "wb") as f:
            cloudpickle.dump((model_data, self._act_params), f)


def load(path):
    """Load act function that was returned by learn function.

    Parameters
    ----------
    path: str
        path to the act function pickle

    Returns
    -------
    act: ActWrapper
        function that takes a batch of observations
        and returns actions.
    """
    return ActWrapper.load(path)

MINUTE = 60.0
HOUR = MINUTE * 60.0
DAY = HOUR * 24.0


DEEPQ_CONFIG = {
    "gpu_memory_fraction": 0.7,
    "lr": 5e-4,
    "max_timesteps": int(1e6),
    "buffer_size": 50000,
    "exploration_fraction": 0.1,
    "exploration_final_eps": 0.02,
    "train_freq": 1,
    "batch_size": 32,
    "print_freq": 100,
    "checkpoint_freq": 10000,
    "checkpoint_path": None,
    "learning_starts": 1000,
    "gamma": 0.99,
    "target_network_update_freq": 500,
    "prioritized_replay": False,
    "prioritized_replay_alpha": 0.6,
    "prioritized_replay_beta0": 0.4,
    "prioritized_replay_beta_iters": None,
    "prioritized_replay_eps": 1e-6,
    "param_noise": False
}


class DeepqLearner:
    def __init__(self, env, q_func, config=DEEPQ_CONFIG, callback=None):
        self.env = env
        self.q_func = q_func
        self.config = config
        self.callback = callback

        # Create all the functions necessary to train the model
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=config["gpu_memory_fraction"])
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        sess.__enter__()

        # capture the shape outside the closure so that the env object is not serialized
        # by cloudpickle when serializing make_obs_ph

        def make_obs_ph(name):
            return ObservationInput(env.observation_space, name=name)

        act, self.train, self.update_target, self.debug = deepq.build_train(
            make_obs_ph=make_obs_ph,
            q_func=q_func,
            num_actions=env.action_space.n,
            optimizer=tf.train.AdamOptimizer(learning_rate=config["lr"]),
            gamma=config["gamma"],
            grad_norm_clipping=10,
            param_noise=config["param_noise"]
        )

        act_params = {
            # 'make_obs_ph': make_obs_ph,
            # 'q_func': q_func,
            'num_actions': env.action_space.n,
        }

        self.act = ActWrapper(act, act_params)

        # Create the replay buffer
        self.config = config
        self.replay_buffer = None
        self.beta_schedule = None
        self.make_replay_buffer()

        # Create the schedule for exploration starting from 1.
        self.exploration = LinearSchedule(schedule_timesteps=int(config["exploration_fraction"] * config["max_timesteps"]),
                                          initial_p=1.0,
                                          final_p=config["exploration_final_eps"])

        # Initialize the parameters and copy them to the target network.
        U.initialize()
        self.update_target()

        self.t = 0
        self.episode_rewards = [0.0]
        self.num_episodes = 1
        self.saved_mean_reward = None
        self.saved_episode_num = None
        self.episode_frames = 0
        self.model_file = None
        self.start_time = 0
        self.episode_start_time = 0

    def make_replay_buffer(self):
        if self.config["prioritized_replay"]:
            self.replay_buffer = PrioritizedReplayBuffer(self.config["buffer_size"], alpha=self.config["prioritized_replay_alpha"])
            if self.config["prioritized_replay_beta_iters"] is None:
                self.config["prioritized_replay_beta_iters"] = self.config["max_timesteps"]
            self.beta_schedule = LinearSchedule(self.config["prioritized_replay_beta_iters"],
                                                initial_p=self.config["prioritized_replay_beta0"],
                                                final_p=1.0)
        else:
            self.replay_buffer = ReplayBuffer(self.config["buffer_size"])
            self.beta_schedule = None

    def run(self):
        reset = True
        obs = self.env.reset()
        self.start_time = time.time()
        self.episode_start_time = time.time()

        with tempfile.TemporaryDirectory() as td:
            td = self.config["checkpoint_path"] or td

            self.model_file = os.path.join(td, "model")
            if tf.train.latest_checkpoint(td) is not None:
                load_state(self.model_file)
                logger.log('Loaded model from {}'.format(self.model_file))

            for self.t in range(self.config["max_timesteps"]):
                if self.callback is not None:
                    if self.callback(locals(), globals()):
                        break

                # Determine next action to take, then take that action and observe results
                action = self._action(obs, reset)
                env_action = action
                new_obs, rew, done, _ = self.env.step(env_action)
                self.replay_buffer.add(obs, action, rew, new_obs, float(done))
                obs = new_obs

                # Increment typical values
                reset = False
                self.episode_frames += 1
                self.episode_rewards[-1] += rew

                # See if done with episode
                if done:
                    obs = self._reset()
                    reset = True

                # Do training and deepq updating as needed
                if self.t > self.config["learning_starts"]:
                    if self.t % self.config["train_freq"] == 0:
                        self._train()
                    if self.t % self.config["target_network_update_freq"] == 0:
                        self.update_target()

    def _action(self, obs, reset):
        # Take action and update exploration to the newest value
        kwargs = {}
        if not self.config["param_noise"]:
            update_eps = self.exploration.value(self.t)
            # update_param_noise_threshold = 0.
        else:
            update_eps = 0.
            # Compute the threshold such that the KL divergence between perturbed and non-perturbed
            # policy is comparable to eps-greedy exploration with eps = exploration.value(t).
            # See Appendix C.1 in Parameter Space Noise for Exploration, Plappert et al., 2017
            # for detailed explanation.
            update_param_noise_threshold = -np.log(
                1. - self.exploration.value(self.t) + self.exploration.value(self.t) / float(self.env.action_space.n))
            kwargs['reset'] = reset
            kwargs['update_param_noise_threshold'] = update_param_noise_threshold
            kwargs['update_param_noise_scale'] = True
        return self.act(np.array(obs)[None], update_eps=update_eps, **kwargs)[0]

    def _train(self):
        try:
            # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
            if self.config["prioritized_replay"]:
                experience = self.replay_buffer.sample(self.config["batch_size"], beta=self.beta_schedule.value(self.t))
                (obses_t, actions, rewards, obses_tp1, dones, weights, batch_idxes) = experience
            else:
                obses_t, actions, rewards, obses_tp1, dones = self.replay_buffer.sample(self.config["batch_size"])
                weights, batch_idxes = np.ones_like(rewards), None

            # Determine errors
            td_errors = self.train(obses_t, actions, rewards, obses_tp1, dones, weights)
            if self.config["prioritized_replay"]:
                new_priorities = np.abs(td_errors) + self.config["prioritized_replay_eps"]
                self.replay_buffer.update_priorities(batch_idxes, new_priorities)
        except Exception as e:
            self.make_replay_buffer()
            print(e)

    def _reset(self):
        self.attempt_print()
        self.attempt_checkpoint()
        self.episode_rewards.append(0.0)
        self.num_episodes += 1
        self.episode_frames = 0
        self.episode_start_time = time.time()

        return self.env.reset()

    def calc_mean_100ep_reward(self):
        if self.num_episodes <= 1:
            return None
        return round(np.mean(self.episode_rewards[-101:-1]), 1)

    def attempt_print(self):
        p_freq = self.config["print_freq"]
        if p_freq is not None and self.num_episodes % p_freq == 0:
            logger.record_tabular("% time spent exploring", int(100 * self.exploration.value(self.t)))
            logger.record_tabular("reward - current", self.episode_rewards[-1])
            logger.record_tabular("reward - mean", self.calc_mean_100ep_reward())
            logger.record_tabular("reward - saved", self.saved_mean_reward)
            logger.record_tabular("episode # - current", self.num_episodes)
            logger.record_tabular("episode # - saved", self.saved_episode_num)
            logger.record_tabular("steps - total", self.t)
            logger.record_tabular("steps - episode", self.episode_frames)
            logger.record_tabular("time - ep duration", str(time.time() - self.episode_start_time) + "s")
            logger.record_tabular("time - remaining", self.estimate_time_remaining())
            logger.dump_tabular()


    def estimate_time_remaining(self):
        duration = time.time() - self.start_time
        if duration <= 0:
            return "Unknown"

        time_remaining = self.t / duration * (self.config["max_timesteps"] - self.t) / 60.0
        suffix = ""

        # Format based on time
        if time_remaining < MINUTE:
            suffix = " seconds"
        elif time_remaining < HOUR:
            suffix = " minutes"
            time_remaining = time_remaining / MINUTE
        elif time_remaining < DAY:
            suffix = " hours"
            time_remaining = time_remaining / HOUR
        else:
            suffix = " days"
            time_remaining = time_remaining / DAY

        # Round remaining time and return
        time_remaining = round(time_remaining * 100.0) / 100.0
        return str(time_remaining) + suffix


    def attempt_checkpoint(self):
        # Determine if we're going to checkpoint
        c_freq = self.config["checkpoint_freq"]
        if c_freq is not None \
                and self.num_episodes > 100 \
                and self.t > self.config["learning_starts"] \
                and self.num_episodes % c_freq == 0:

            # Determine if reward is growing
            mean_100ep_reward = self.calc_mean_100ep_reward()
            if self.saved_mean_reward is None or mean_100ep_reward > self.saved_mean_reward:
                if self.config["print_freq"] is not None:
                    logger.log("Saving model due to mean reward increase: {} -> {}".format(
                        self.saved_mean_reward, mean_100ep_reward))
                    self.saved_mean_reward = mean_100ep_reward
                    self.saved_episode_num = self.num_episodes
                    save_state(self.model_file)

    def save(self, save_path):
        print("Saving model to " + save_path)
        self.act.save(save_path)
