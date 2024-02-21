from torchrl.record.loggers import TensorboardLogger, WandbLogger

import gymnasium as gym
import sys
import torch as th
import numpy as np

from torchrl_policies import utils
from torchrl_policies.trainer import Trainer
from torchrl_policies.policies.random import RandomPolicy
from torchrl_policies.callbacks import CheckpointCallback


POLICY_LIST = {
    "0": lambda obs, ac: RandomPolicy(observation_space=obs, action_space=ac)
}

ENV_LIST = {
    "0": lambda: (
        gym.make("HalfCheetah-v4", render_mode="human"),
        {
            "observation_space": utils.create_space_spec(
                -np.inf, np.inf, (17,), th.float32
            ),
            "action_space": utils.create_space_spec(-1, 1, (6,), th.float32),
        },
    )
}

LOGGER_LIST = {
    "0": lambda name: utils.PrintLogger(name),
    "1": lambda name: TensorboardLogger(name),
    "2": lambda name: WandbLogger(
        name, project="torchrl-policies-test", entity="adkala", sync_tensorboard=True
    ),
}


def main():
    print(sys.argv[1])
    env, spec = ENV_LIST[sys.argv[2]]()

    policy = POLICY_LIST[sys.argv[1]](spec["observation_space"], spec["action_space"])

    name = (
        policy.__class__.__name__
        + "_"
        + env.unwrapped.spec.id
        + "_"
        + utils.get_time_str()
    )
    logger = LOGGER_LIST[sys.argv[3]](name)

    callbacks = [
        CheckpointCallback(save_freq=1000, save_path=f"models/{name}", policy=policy)
    ]

    trainer = Trainer(policy, env, logger=logger)

    trainer.learn(
        progress_bar=True,
        max_episode_steps=10,
        learning_starts=1,
        callbacks=callbacks,
    )


if __name__ == "__main__":
    main()
