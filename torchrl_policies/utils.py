from tensordict import TensorDictBase
from torchrl.data import TensorDictReplayBuffer, BoundedTensorSpec
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.data import TensorDictReplayBuffer
from abc import ABC, abstractmethod
from torch import nn

import torch
import numpy as np


class TorchRLPolicy(ABC):
    def __init__(self, obs_spec, action_spec):
        self.obs_spec = obs_spec
        self.action_spec = action_spec

    def __call__(self, td: TensorDictBase):
        return self.forward(td)

    @abstractmethod
    def forward(self, td):  # policy.forward
        pass

    @abstractmethod
    def grad(self):  # run gradient step
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


def create_action_space(low: np.ndarray, high: np.ndarray, dtype, device=None):
    return BoundedTensorSpec(low, high, device=device, dtype=dtype)


def create_replay_buffer(buffer_size=1_000_000, pin_memory=False):
    replay_buffer = TensorDictReplayBuffer(
        pin_memory=pin_memory,
        storage=LazyTensorStorage(
            buffer_size,
        ),
    )
    return replay_buffer
