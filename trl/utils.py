from tensordict import TensorDict, TensorDictBase
from torchrl.data import TensorDictReplayBuffer, BoundedTensorSpec
from torchrl.data.replay_buffers.storages import LazyMemmapStorage, LazyTensorStorage
from torch import nn

from stable_baselines3.common.callbacks import BaseCallback, ConvertCallback

from abc import ABC, abstractmethod
from tqdm.rich import tqdm
from contextlib import nullcontext

import gymnasium as gym
import torch
import numpy as np
import math


class TRLModel(ABC):
    def __call__(self, td: TensorDictBase):
        return self.forward(td)

    @abstractmethod
    def forward(self, td):  # policy.forward
        pass

    @abstractmethod
    def step(self):  # run gradient step
        pass

    @abstractmethod
    def eval(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def state_dict(self):
        pass

    @abstractmethod
    def load_state_dict(self, state_dict):
        pass

    def save(self, path):
        return torch.save(self.state_dict(), path)


class TRLForSB3:  # for use with sb3 related functionality
    def __init__(self, model: TRLModel, env, logger=None):
        self.model = model

        if isinstance(env, gym.Env):
            env = TRLGymWrapper(env)
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
    ):
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
                    # pbar=pbar, max_episode_steps=max_episode_steps, callback=callback
                    pbar=pbar,
                    callback=callback,
                )

                self._log_trajectory(tensordict)

                # if self.num_timesteps >= learning_starts:
                #     loss, losses, loss_td = self.update_model(
                #         # gradient_steps=gradient_steps,
                #         pbar=pbar,
                #     )
                #     self._log_losses(losses, loss_td)

        callback.on_training_end()

    def save(self, path):
        torch.save(self.model.state_dict, path)

    def collect_trajectory(self, pbar=None, max_episode_steps=3000, callback=None):
        sample_trajectory = []
        td = self.env.reset()

        # remove observation change (REMOVE)
        # td["observation"] = td["observation"].permute((2, 0, 1))
        # end -- remove observation change (REMOVE)

        callback.on_rollout_start()

        start = self.num_timesteps

        self.model.eval()
        # with torch.no_grad():
        while True:
            # move to correct device
            if td.device != self.model.device:
                td = td.to(self.model.device, non_blocking=True)

            td = self.model(td)
            td = self.env.step(td)

            self.replay_buffer.add(td.detach().cpu())

            # remove observation change (REMOVE)
            # td["next"]["observation"] = td["next"]["observation"].permute((2, 0, 1))
            # end -- remove observation change (REMOVE)

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
                loss, losses, loss_td = self.update_model(
                    # gradient_steps=gradient_steps,
                    # pbar=pbar,
                )
                self.model.eval()

                if self.num_timesteps % 100 == 0:  # redo
                    self._log_losses(losses, loss_td)

        callback.on_rollout_end()

        tensordict = torch.stack(sample_trajectory)
        # self.replay_buffer.extend(tensordict)
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

    @staticmethod
    def _create_replay_buffer(
        batch_size=256, buffer_size=1_000_000, pin_memory=False, prefetch=3
    ):
        replay_buffer = TensorDictReplayBuffer(
            pin_memory=pin_memory,
            prefetch=prefetch,
            # storage=LazyMemmapStorage(
            storage=LazyTensorStorage(
                buffer_size,
                # scratch_dir=None,
            ),
            batch_size=batch_size,
        )
        return replay_buffer


class TRLGymWrapper:  # purpose: get TD for trajectories for step, reset functions
    def __init__(self, env: gym.Env):
        self.env = env
        # self._max_episode_steps = env._max_episode_steps
        self._max_episode_steps = 10000  # don't hardcode this

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)

        td = self._get_td(observation=observation)
        return td

    def step(self, td: TensorDictBase):
        action = td["action"]
        observation, reward, terminated, truncated, info = self.env.step(
            action.detach().cpu().numpy()
        )

        td.set(
            "next",
            self._get_td(
                observation=observation,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                done=(terminated or truncated),
            ),
        )

        return td

    @staticmethod
    def _get_td(
        observation=[0], reward=0, terminated=False, truncated=False, done=False
    ):
        td = TensorDict(
            {
                "done": torch.tensor([done], dtype=torch.bool),
                "reward": torch.tensor([reward], dtype=torch.float32),
                "observation": torch.tensor(observation, dtype=torch.float32),
                "terminated": torch.tensor([terminated], dtype=torch.bool),
                "truncated": torch.tensor([truncated], dtype=torch.bool),
            },
            batch_size=[],
        )
        return td


def create_action_space(low: np.ndarray, high: np.ndarray, dtype, device=None):
    return BoundedTensorSpec(low, high, device=device, dtype=dtype)
