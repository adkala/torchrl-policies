from tensordict import TensorDict
from stable_baselines3.common.callbacks import BaseCallback, ConvertCallback
from contextlib import nullcontext
from tqdm.rich import tqdm

from .gym import TorchRLGymWrapper
from .policies.base_policy import TorchRLPolicy

import torch as th
import gymnasium as gym
import math

from . import utils


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
        callbacks=None,
    ):
        if reset_num_timesteps:
            self.num_timesteps = 0

        pbar = (
            tqdm(total=total_timesteps - self.num_timesteps) if progress_bar else None
        )

        with pbar if pbar else nullcontext():
            while self.num_timesteps < total_timesteps:
                td = self.collect_trajectory(
                    pbar=pbar, max_episode_steps=max_episode_steps, callbacks=callbacks
                )
                self._log_trajectory(td)

                if self.num_timesteps > learning_starts:
                    _, losses, loss_td = self.update_policy(
                        max_episode_steps, batch_size, gradient_epochs, pbar
                    )
                    self._log_losses(losses, loss_td)
                    self._log_gradients(self.policy.policy)

    def collect_trajectory(self, pbar=None, max_episode_steps=3000, callbacks=None):
        sample_trajectory = []
        td = self.env.reset()

        episode_steps = 0

        while True:
            if td.device != self.policy.device:
                td = td.to(self.policy.device, non_blocking=True)

            td = self.policy(td)
            td = self.env.step(td)

            self.replay_buffer.add(td.detach().cpu())

            episode_steps += 1

            self.policy.num_timesteps += 1
            self.num_timesteps += 1
            if pbar:
                pbar.update(1)

            if callbacks:
                for callback in callbacks:
                    callback.on_step()

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

            loss, loss_td = self.policy.grad(sample)

            losses[i] = loss_td.select("loss_actor", "loss_qvalue", "loss_alpha")

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

    def _log_gradients(self, policy: th.nn.Module):
        for tag, value in policy.named_parameters():
            if value.grad is not None:
                self.logger.log_histogram(
                    tag + "/grad", value.grad.cpu(), step=self.num_timesteps
                )

    def _log_video(self, video):
        raise NotImplementedError
