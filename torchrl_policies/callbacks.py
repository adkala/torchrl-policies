from torchrl_policies.policies.base_policy import TorchRLPolicy

import abc
import torch as th
import os


class BaseCallback(abc.ABC):
    @abc.abstractmethod
    def on_step(self):
        pass


class CheckpointCallback:
    def __init__(
        self, save_freq, save_path, policy: TorchRLPolicy
    ):  # todo: use locals and globals for getting all state dicts
        self.save_freq = save_freq
        self.save_path = save_path

        self.policy = policy

        self.num_timesteps = 0

        if not os.path.exists(save_path):
            os.makedirs(save_path)

    def on_step(self):
        self.num_timesteps += 1
        if self.num_timesteps % self.save_freq == 0:
            print("saving")
            th.save(
                {"policy_state_dict": self.policy.state_dict},
                f"{self.save_path}/checkpoint_{self.num_timesteps}.pt",
            )
