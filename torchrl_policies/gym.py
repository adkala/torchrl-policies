from tensordict import TensorDict, TensorDictBase

import torch as th
import gymnasium as gym


class TorchRLGymWrapper:  # purpose: get TD for trajectories for step, reset functions
    def __init__(self, env: gym.Env, max_episode_steps=None):
        self.env = env
        self._max_episode_steps = (
            env._max_episode_steps if not max_episode_steps else None
        )
        # self._max_episode_steps = 10000  # don't hardcode this

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
                "done": th.tensor([done], dtype=th.bool),
                "reward": th.tensor([reward], dtype=th.float32),
                "observation": th.tensor(observation, dtype=th.float32),
                "terminated": th.tensor([terminated], dtype=th.bool),
                "truncated": th.tensor([truncated], dtype=th.bool),
            },
            batch_size=[],
        )
        return td
