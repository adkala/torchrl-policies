from tensordict import TensorDictBase, TensorDict
from abc import ABC, abstractmethod

import torch as th


class TorchRLPolicy(ABC):
    @abstractmethod
    def forward(self, td) -> TensorDict:  # policy.forward
        """
        Returned tensordict must contain "action" key
        """
        pass

    @abstractmethod
    def grad(self):
        """
        Take a gradient step
        """
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

    def __init__(self, *, device=None, **kwargs):
        """
        Attributes:
            device (torch.device): device to use for policy,
            num_timesteps (int): number of timesteps seen by policy,
            policy (th.nn.Module): policy network(s)
        """
        if device is None:
            device = "cuda" if th.cuda.is_available() else "cpu"
        if isinstance(device, str):
            device = th.device(device)

        self.device = device
        self.num_timesteps = 0
        self.policy = th.nn.Module()  # default policy is none

    def __call__(self, td: TensorDictBase):
        return self.forward(td)

    def save(self, path):
        return th.save(self.state_dict(), path)
