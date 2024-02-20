from tensordict import TensorDict
from stable_baselines3.common.callbacks import BaseCallback, ConvertCallback
from contextlib import nullcontext
from tqdm.rich import tqdm

from utils import TorchRLPolicy
from gym import TorchRLGymWrapper

import torch as th
import gymnasium as gym
import math

import utils


class Trainer:
    def __init__(self, policy: TorchRLPolicy, env, logger=None):
        self.policy = policy

        if isinstance(env, gym.Env):
            env = TorchRLGymWrapper(env)
        self.env = env

        self.replay_buffer = utils.create_replay_buffer()

        self.logger = logger

        self.num_timesteps = 0

    def learn(
        self,
        *,
        total_timesteps=1_000_000,
        progress_bar=False,
        reset_num_timesteps=False,
        learning_starts=1000,
        max_episode_steps=3000,
        batch_size=256,
        gradient_epochs=10,
    ):
        if reset_num_timesteps:
            self.num_timesteps = 0

        pbar = (
            tqdm(total=total_timesteps - self.num_timesteps) if progress_bar else None
        )

        with pbar if pbar else nullcontext():
            while self.num_timesteps < total_timesteps:
                td = self.collect_trajectory(max_episode_steps=max_episode_steps)
                self._log_trajectory(td)

                if self.num_timesteps > learning_starts:
                    _, losses, loss_td = self.update_policy(
                        max_episode_steps, batch_size, gradient_epochs, pbar
                    )
                    self._log_losses(losses, loss_td)

    def collect_trajectory(self, pbar=None, max_episode_steps=3000):
        sample_trajectory = []
        td = self.env.reset()

        episode_steps = 0
        gradient_steps = 0

        while True:
            if td.device != self.policy.device:
                td = td.to(self.model.device, non_blocking=True)

            td = self.policy(td)
            td = self.env.step(td)

            self.replay_buffer.add(td.detach().cpu)

            episode_steps += 1

            self.policy.num_timesteps += 1
            self.num_timesteps += 1
            if pbar:
                pbar.update(1)

            if max_episode_steps and episode_steps >= max_episode_steps:
                td["next", "truncated"][0] = True

            sample_trajectory.append(td.detach().cpu())

            truncated, terminated, done = (
                td["next", "truncated"][0],
                td["next", "terminated"][0],
                td["next", "done"][0],
            )

            if truncated or terminated or done:
                break

            td = td["next"].clone()

        return th.stack(sample_trajectory)

    def update_policy(
        self, max_episode_steps=3000, batch_size=256, epochs=10, pbar=None
    ):
        gradient_steps = math.ceil(max_episode_steps / batch_size) * epochs

        losses = TensorDict(
            {},
            batch_size=[
                gradient_steps,
            ],
        )

        self.policy.train()
        for i in range(gradient_steps):
            sample = self.replay_buffer.sample(batch_size)
            if sample.device != self.policy.device:
                sample = sample.to(self.policy.device, non_blocking=True)
            else:
                sample = sample.clone()

            loss, loss_td = self.policy.step(sample)

            losses[i] = loss_td.select("loss_actor", "loss_qvalue", "loss_alpha")

            if pbar:
                pbar.set_description(
                    f"Gradient steps: {i} / {gradient_steps}, Policy loss: {loss}"
                )
        if pbar:
            pbar.set_description(f"Policy loss: {loss}")

        return loss, losses, loss_td

    def _log_trajectory(self, tensordict):
        # Logging
        metrics_to_log = {}
        metrics_to_log["train/reward_mean"] = tensordict["next", "reward"].mean().item()
        metrics_to_log["train/reward_stdev"] = tensordict["next", "reward"].std().item()
        metrics_to_log["train/reward_min"] = tensordict["next", "reward"].min().item()
        metrics_to_log["train/reward_max"] = tensordict["next", "reward"].max().item()
        metrics_to_log["train/ep_len"] = tensordict.batch_size[0]

        if self.logger:
            for key, value in metrics_to_log.items():
                self.logger.log_scalar(key, value, step=self.num_timesteps)

        return metrics_to_log

    def _log_losses(self, losses, loss_td):
        # Logging
        metrics_to_log = {}
        metrics_to_log["train/actor_loss"] = losses["loss_actor"].mean().item()
        metrics_to_log["train/q_loss"] = losses["loss_qvalue"].mean().item()
        metrics_to_log["train/alpha_loss"] = losses["loss_alpha"].mean().item()
        metrics_to_log["train/alpha"] = loss_td["alpha"].item()
        metrics_to_log["train/entropy"] = loss_td["entropy"].item()

        if self.logger:
            for key, value in metrics_to_log.items():
                self.logger.log_scalar(key, value, step=self.num_timesteps)

        return metrics_to_log


class TrainerForSB3:  # for use with sb3 related functionality
    def __init__(self, model: TorchRLPolicy, env, logger=None):
        self.model = model

        if isinstance(env, gym.Env):
            env = TorchRLGymWrapper(env)
        self.env = env

        # replay buffer
        self.replay_buffer = self._create_replay_buffer()

        # logger
        self.logger = logger

        # attributes
        self.num_timesteps = 0

    def learn(
        self,
        *,
        total_timesteps=1_000_000,
        callback=None,  # sb3 callback integration
        progress_bar=False,
        reset_num_timesteps=False,
        learning_starts=1000,
        max_episode_steps=None,
        gradient_steps=None,
        log_grad_steps_interval=None,
    ):  # for support with SB3 callbacks
        if progress_bar:
            pbar = tqdm(total=total_timesteps)
        else:
            pbar = None

        if reset_num_timesteps:
            self.num_timesteps = 0

        # callback
        if not isinstance(callback, BaseCallback):
            callback = ConvertCallback(callback)
        self.model.num_timesteps = self.num_timesteps
        callback.init_callback(self.model)

        callback.on_training_start(locals(), globals())

        with pbar if pbar else nullcontext():  # replace with sb3 progress bar callback
            while self.num_timesteps < total_timesteps:
                tensordict = self.collect_trajectory(
                    pbar=pbar,
                    callback=callback,
                )

                self._log_trajectory(tensordict)

        callback.on_training_end()

    def save(self, path):
        th.save(self.model.state_dict, path)

    def collect_trajectory(self, pbar=None, max_episode_steps=3000, callback=None):
        sample_trajectory = []
        td = self.env.reset()

        callback.on_rollout_start()

        start = self.num_timesteps

        while True:
            # move to correct device
            if td.device != self.model.device:
                td = td.to(self.model.device, non_blocking=True)

            td = self.model(td)
            td = self.env.step(td)

            self.replay_buffer.add(td.detach().cpu())

            if pbar:
                pbar.update(1)
            self.num_timesteps += 1

            self.model.num_timesteps += 1  # for sb3 callback (not clean, mb change)
            callback.update_locals(locals())

            if (
                max_episode_steps and self.num_timesteps - start >= max_episode_steps
            ) or not callback.on_step():
                td["next", "truncated"][0] = True  # manual truncation

            sample_trajectory.append(td.detach().cpu())

            truncated, terminated, done = (
                td["next", "truncated"][0],
                td["next", "terminated"][0],
                td["next", "done"][0],
            )
            if terminated or truncated or done:
                break

            td = td["next"].clone()

            if self.num_timesteps >= 100:  # learning_starts:
                self.model.train()
                loss, losses, loss_td = self.update_model()
                self.model.eval()

                if self.num_timesteps % 100 == 0:  # redo
                    self._log_losses(losses, loss_td)

        callback.on_rollout_end()

        tensordict = th.stack(sample_trajectory)
        return tensordict

    def update_model(self, gradient_steps=1, pbar=None):  #
        """
        set log_grad_steps_interval to -1 for no logging (might want to change)
        """
        if not gradient_steps:
            gradient_steps = self.env._max_episode_steps

        losses = TensorDict(
            {},
            batch_size=[
                gradient_steps,
            ],
        )

        self.model.train()
        for i in range(math.ceil(gradient_steps / self.replay_buffer._batch_size)):
            sampled_tensordict = self.replay_buffer.sample()
            if sampled_tensordict.device != self.model.device:
                sampled_tensordict = sampled_tensordict.to(
                    self.model.device, non_blocking=True
                )
            else:
                sampled_tensordict = sampled_tensordict.clone()

            loss, loss_td = self.model.step(sampled_tensordict)

            losses[i] = loss_td.select("loss_actor", "loss_qvalue", "loss_alpha")

            if pbar:
                pbar.set_description(f"Gradient Steps: {i}, Policy loss: {loss}")
        if pbar:
            pbar.set_description(f"Policy loss: {loss}")

        return loss, losses, loss_td

    def _log_trajectory(self, tensordict):
        # Logging
        metrics_to_log = {}
        metrics_to_log["train/reward_mean"] = tensordict["next", "reward"].mean().item()
        metrics_to_log["train/reward_stdev"] = tensordict["next", "reward"].std().item()
        metrics_to_log["train/reward_min"] = tensordict["next", "reward"].min().item()
        metrics_to_log["train/reward_max"] = tensordict["next", "reward"].max().item()
        metrics_to_log["train/ep_len"] = tensordict.batch_size[0]

        if self.logger:
            for key, value in metrics_to_log.items():
                self.logger.log_scalar(key, value, step=self.num_timesteps)

        return metrics_to_log

    def _log_losses(self, losses, loss_td):
        # Logging
        metrics_to_log = {}
        metrics_to_log["train/actor_loss"] = losses["loss_actor"].mean().item()
        metrics_to_log["train/q_loss"] = losses["loss_qvalue"].mean().item()
        metrics_to_log["train/alpha_loss"] = losses["loss_alpha"].mean().item()
        metrics_to_log["train/alpha"] = loss_td["alpha"].item()
        metrics_to_log["train/entropy"] = loss_td["entropy"].item()

        if self.logger:
            for key, value in metrics_to_log.items():
                self.logger.log_scalar(key, value, step=self.num_timesteps)

        return metrics_to_log
