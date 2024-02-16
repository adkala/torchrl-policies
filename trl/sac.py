from tensordict import TensorDictBase
from tensordict.nn.distributions import NormalParamExtractor
from tensordict.nn import InteractionType, TensorDictModule
from torchrl.modules import MLP, ProbabilisticActor, ValueOperator
from torchrl.envs.libs.gym import GymWrapper
from torchrl.envs import TransformedEnv, Compose
from torchrl.modules.distributions import TanhNormal
from torchrl.data import TensorDictReplayBuffer
from torchrl.data.replay_buffers.storages import LazyMemmapStorage
from torchrl.objectives import SoftUpdate
from torchrl.objectives.sac import SACLoss
from torch import nn

from tqdm import tqdm

from utils import TRLModel

import numpy as np
import torch
import gymnasium as gym


class SAC:
    def __init__(
        self, policy, critic, action_spec, device=None
    ):  # used kwarg to better match sb3, but not defaulting critic bc lazy (sb3 defaults critic)
        # set device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device

        # actor critic
        self.actor = self._create_actor(policy, action_spec)
        self.critic = self._create_critic(critic)

        self.policy = nn.ModuleList([self.actor, self.critic])  # for logging
        self.policy.to(device)

        # loss``
        self.loss, self.target_net_updater = self._create_loss(self.actor, self.critic)

        # optimizers
        (
            self.optimizer_actor,
            self.optimizer_critic,
            self.optimizer_alpha,
        ) = self._create_optimizers(self.loss)

    def step(self, sampled_tensordict):
        loss_td = self.loss(sampled_tensordict)

        actor_loss = loss_td["loss_actor"]
        q_loss = loss_td["loss_qvalue"]
        alpha_loss = loss_td["loss_alpha"]

        # Update actor
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()

        # Update critic
        self.optimizer_critic.zero_grad()
        q_loss.backward()
        self.optimizer_critic.step()

        # Update alpha
        self.optimizer_alpha.zero_grad()
        alpha_loss.backward()
        self.optimizer_alpha.step()

        # Update qnet_target params
        self.target_net_updater.step()

        return actor_loss.item(), loss_td.detach().cpu()

    def _create_actor(self, policy, action_spec):
        in_keys = ["observation"]  # 'observation' for CarRacing-v2

        dist_class = TanhNormal
        dist_kwargs = {
            "min": action_spec.space.low,
            "max": action_spec.space.high,
            "tanh_loc": False,
        }

        actor_extractor = NormalParamExtractor()
        actor_net = nn.Sequential(policy, actor_extractor)
        actor_module = TensorDictModule(
            actor_net, in_keys=in_keys, out_keys=["loc", "scale"]
        )

        actor = ProbabilisticActor(
            spec=action_spec,
            in_keys=["loc", "scale"],
            module=actor_module,
            distribution_class=dist_class,
            distribution_kwargs=dist_kwargs,
            default_interaction_type=InteractionType.RANDOM,
            return_log_prob=False,
        )

        return actor

    def _create_critic(self, critic):
        # scuffed q network using custom tensordict module
        qvalue = ValueOperator(in_keys=["observation", "action"], module=critic)
        return qvalue

    def _create_loss(
        self, actor, critic, alpha=1.0, gamma=0.99, target_update_polyak=0.995
    ):
        loss_module = SACLoss(
            actor_network=actor,
            qvalue_network=critic,
            num_qvalue_nets=2,
            loss_function="l2",
            delay_actor=False,
            delay_qvalue=True,
            alpha_init=alpha,
        )
        loss_module.make_value_estimator(gamma=gamma)

        # Define Target Network Updater
        target_net_updater = SoftUpdate(loss_module, eps=target_update_polyak)
        return loss_module, target_net_updater

    def _create_optimizers(self, loss_module, lr=1e-5, eps=1.0e-8):
        critic_params = list(loss_module.qvalue_network_params.flatten_keys().values())
        actor_params = list(loss_module.actor_network_params.flatten_keys().values())

        optimizer_actor = torch.optim.Adam(
            actor_params,
            lr=lr,
            eps=eps,
        )
        optimizer_critic = torch.optim.Adam(
            critic_params,
            lr=lr,
            eps=eps,
        )
        optimizer_alpha = torch.optim.Adam(
            [loss_module.log_alpha],
            lr=3.0e-4,
        )
        return optimizer_actor, optimizer_critic, optimizer_alpha


class SACModel(SAC, TRLModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, td):  # policy.forward
        td = self.actor(td)
        return td

    def step(self, sampled_tensordict):
        return super().step(sampled_tensordict)

    def train(self):
        self.actor.train()
        self.critic.train()

    def eval(self):
        self.actor.eval()
        self.critic.eval()

    def state_dict(self):
        return {"policy_state_dict": self.policy.state_dict()}

    def load_state_dict(self, state_dict):
        raise NotImplementedError
