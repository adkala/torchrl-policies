from utils import TorchRLPolicy


class RandomPolicy(TorchRLPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, td):  # policy.forward
        return self.action_spec.rand()

    def grad(self, sampled_tensordict):
        pass

    def train(self):
        pass

    def eval(self):
        pass

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass
