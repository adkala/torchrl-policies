from tensordict import TensorDict

import torch as th

from .base_policy import TorchRLPolicy


class RandomPolicy(TorchRLPolicy):
    def __init__(self, *args, observation_space, action_space, **kwargs):
        super().__init__(*args, **kwargs)
        self.observation_space = observation_space
        self.action_space = action_space

    def forward(self, td):  # policy.forward
        td.set("action", self.action_space.rand())
        return td

    def grad(self, sampled_tensordict):
        return 0, TensorDict(
            {
                "loss_actor": th.tensor(0, dtype=th.float32),
                "loss_qvalue": th.tensor(0, dtype=th.float32),
                "loss_alpha": th.tensor(0, dtype=th.float32),
                "alpha": th.tensor(0, dtype=th.float32),
                "entropy": th.tensor(0, dtype=th.float32),
            },
            batch_size=th.Size([]),
        )

    def train(self):
        pass

    def eval(self):
        pass

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass
