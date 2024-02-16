from torch import nn

import torch
import gymnasium as gym


class MLPPolicy(
    nn.Module
):  # duplicate default MLPPolicy from SAC implementation in SB3 (2 layers, 256 units)
    def __init__(
        self, input_size, output_size, layers=2, num_units=256, squashed=True
    ):  # set squashed to False for default critic implementation
        super().__init__()
        modules = [
            nn.Linear(input_size, num_units),
            nn.ReLU(),
            *([nn.Linear(num_units, num_units), nn.ReLU()] * layers),
            nn.Linear(num_units, output_size),
        ]
        if squashed:
            modules += [nn.Tanh()]

        self.linear = nn.Sequential(*modules)

    def forward(self, observations: torch.Tensor) -> torch.Tensor():
        return self.linear(observations)


class NatureCNN(nn.Module):  # standard implementation for pixel-based input in SB3
    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 512,
        normalized_image: bool = False,
        wrappers=[],
    ) -> None:
        super().__init__()
        for f in wrappers:
            observation_space = f(observation_space)

        n_input_channels = observation_space.shape[-1]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(
                    observation_space.sample().transpose(2, 0, 1)[None]
                ).float()
            ).shape[
                -1
            ]  # transposed for channel first

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batched = True
        if observations.dim() == 3:
            observations = observations[None]
            batched = False

        out = self.linear(self.cnn(observations))

        if not batched:
            out = out[0]

        return out


class PixelQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.cnn = NatureCNN(env.observation_space)
        self.linear = MLPPolicy(515, 1, squashed=False)

    def forward(self, pixels, action):
        x = self.cnn(pixels)
        x = torch.cat((x, action), -1)
        x = self.linear(x)
        return x


class ObsActionNetwork(nn.Module):
    def __init__(self, network):
        super().__init__()
        self.network = network

    def forward(self, obs, action):
        x = torch.cat((obs, action), -1)
        x = self.network(x)
        return x
