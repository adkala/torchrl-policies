from torchrl.record.loggers import TensorboardLogger

import gymnasium as gym
import sys

from torchrl_policies.policies.random import RandomPolicy


POLICY_LIST = [
    0: lambda obs_spec, action_spec: RandomPolicy(obs_spec, action_spec)
]

ENV_LIST = [
    0: lambda: gym.make('HalfCheetah-v2')
]

LOGGER_LIST = [
    0: lambda: name, dir: TensorboardLogger(name, dir)
]

def main():
    policy = POLICY_LIST[sys.argv[1]]
    env = ENV_LIST[sys.argv[2]]
    logger = LOGGER_LIST[sys.argv[3]]

    
