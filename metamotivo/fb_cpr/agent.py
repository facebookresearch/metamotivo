# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
from typing import Dict

import torch
import torch.nn.functional as F
from torch import autograd

from ..fb.agent import FBAgent
from ..fb.agent import TrainConfig as FBTrainConfig
from ..nn_models import _soft_update_params, eval_mode
from .model import Config as FBcprModelConfig
from .model import FBcprModel, config_from_dict


@dataclasses.dataclass
class TrainConfig(FBTrainConfig):
    lr_discriminator: float = 1e-4
    lr_critic: float = 1e-4
    critic_target_tau: float = 0.005
    critic_pessimism_penalty: float = 0.5
    reg_coeff: float = 1
    scale_reg: bool = True
    # the z distribution for rollouts (when agent.use_mix_rollout=1) and for the mini-batches used in the network updates is:
    # - a fraction of 'expert_asm_ratio' zs from expert trajectory encoding
    # - a fraction of 'train_goal_ratio' zs from goal encoding (goals sampled from the train buffer)
    # - the remaining fraction from the uniform distribution
    expert_asm_ratio: float = 0
    # a fraction of 'relabel_ratio' transitions in each mini-batch are relabeled with a z sampled from the above distribution
    relabel_ratio: float | None = 1
    grad_penalty_discriminator: float = 10.0
    weight_decay_discriminator: float = 0.0


@dataclasses.dataclass
class Config:
    model: FBcprModelConfig = dataclasses.field(default_factory=FBcprModelConfig)
    train: TrainConfig = dataclasses.field(default_factory=TrainConfig)
    cudagraphs: bool = False
    compile: bool = False


class FBcprAgent(FBAgent):
    def __init__(self, **kwargs):
        # make sure batch size is a multiple of seq_length
        seq_length = kwargs["model"]["seq_length"]
        batch_size = kwargs["train"]["batch_size"]
        kwargs["train"]["batch_size"] = int(torch.ceil(torch.tensor([batch_size / seq_length])) * seq_length)
        del seq_length, batch_size

        self.cfg = config_from_dict(kwargs, Config)
        self._model = FBcprModel(**dataclasses.asdict(self.cfg.model))
        self._model.to(self.cfg.model.device)
        self.setup_training()
        self.setup_compile()

    def setup_training(self) -> None:
        super().setup_training()

        # prepare parameter list
        self._critic_map_paramlist = tuple(x for x in self._model._critic.parameters())
        self._target_critic_map_paramlist = tuple(x for x in self._model._target_critic.parameters())

        self.critic_optimizer = torch.optim.Adam(
            self._model._critic.parameters(),
            lr=self.cfg.train.lr_critic,
            capturable=self.cfg.cudagraphs and not self.cfg.compile,
            weight_decay=self.cfg.train.weight_decay,
        )
        self.discriminator_optimizer = torch.optim.Adam(
            self._model._discriminator.parameters(),
            lr=self.cfg.train.lr_discriminator,
            capturable=self.cfg.cudagraphs and not self.cfg.compile,
            weight_decay=self.cfg.train.weight_decay_discriminator,
        )

    def setup_compile(self):
        super().setup_compile()
        if self.cfg.compile:
            mode = "reduce-overhead" if not self.cfg.cudagraphs else None
            self.update_critic = torch.compile(self.update_critic, mode=mode)
            self.update_discriminator = torch.compile(self.update_discriminator, mode=mode)
            self.encode_expert = torch.compile(self.encode_expert, mode=mode, fullgraph=True)

        if self.cfg.cudagraphs:
            from tensordict.nn import CudaGraphModule

            self.update_critic = CudaGraphModule(self.update_critic, warmup=5)
            self.update_discriminator = CudaGraphModule(self.update_discriminator, warmup=5)
            self.encode_expert = CudaGraphModule(self.encode_expert, warmup=5)

    @torch.no_grad()
    def sample_mixed_z(self, train_goal: torch.Tensor, expert_encodings: torch.Tensor, *args, **kwargs):
        z = self._model.sample_z(self.cfg.train.batch_size, device=self.device)
        p_goal = self.cfg.train.train_goal_ratio
        p_expert_asm = self.cfg.train.expert_asm_ratio
        prob = torch.tensor(
            [p_goal, p_expert_asm, 1 - p_goal - p_expert_asm],
            dtype=torch.float32,
            device=self.device,
        )
        mix_idxs = torch.multinomial(prob, num_samples=self.cfg.train.batch_size, replacement=True).reshape(-1, 1)

        # zs obtained by encoding train goals
        perm = torch.randperm(self.cfg.train.batch_size, device=self.device)
        goals = self._model._backward_map(train_goal[perm])
        goals = self._model.project_z(goals)
        z = torch.where(mix_idxs == 0, goals, z)

        # zs obtained by encoding expert trajectories
        perm = torch.randperm(self.cfg.train.batch_size, device=self.device)
        z = torch.where(mix_idxs == 1, expert_encodings[perm], z)

        return z

    @torch.no_grad()
    def encode_expert(self, next_obs: torch.Tensor):
        # encode expert trajectories through B
        B_expert = self._model._backward_map(next_obs).detach()  # batch x d
        B_expert = B_expert.view(
            self.cfg.train.batch_size // self.cfg.model.seq_length,
            self.cfg.model.seq_length,
            B_expert.shape[-1],
        )  # N x L x d
        z_expert = B_expert.mean(dim=1)  # N x d
        z_expert = self._model.project_z(z_expert)
        z_expert = torch.repeat_interleave(z_expert, self.cfg.model.seq_length, dim=0)  # batch x d
        return z_expert

    def update(self, replay_buffer, step: int) -> Dict[str, torch.Tensor]:
        expert_batch = replay_buffer["expert_slicer"].sample(self.cfg.train.batch_size)
        train_batch = replay_buffer["train"].sample(self.cfg.train.batch_size)

        train_obs, train_action, train_next_obs = (
            train_batch["observation"].to(self.device),
            train_batch["action"].to(self.device),
            train_batch["next"]["observation"].to(self.device),
        )
        discount = self.cfg.train.discount * ~train_batch["next"]["terminated"].to(self.device)
        expert_obs, expert_next_obs = (
            expert_batch["observation"].to(self.device),
            expert_batch["next"]["observation"].to(self.device),
        )

        self._model._obs_normalizer(train_obs)
        self._model._obs_normalizer(train_next_obs)

        with torch.no_grad(), eval_mode(self._model._obs_normalizer):
            train_obs, train_next_obs = (
                self._model._obs_normalizer(train_obs),
                self._model._obs_normalizer(train_next_obs),
            )
            expert_obs, expert_next_obs = (
                self._model._obs_normalizer(expert_obs),
                self._model._obs_normalizer(expert_next_obs),
            )

        torch.compiler.cudagraph_mark_step_begin()
        expert_z = self.encode_expert(next_obs=expert_next_obs)
        train_z = train_batch["z"].to(self.device)

        # train the discriminator
        grad_penalty = self.cfg.train.grad_penalty_discriminator if self.cfg.train.grad_penalty_discriminator > 0 else None
        metrics = self.update_discriminator(
            expert_obs=expert_obs,
            expert_z=expert_z,
            train_obs=train_obs,
            train_z=train_z,
            grad_penalty=grad_penalty,
        )

        z = self.sample_mixed_z(train_goal=train_next_obs, expert_encodings=expert_z).clone()
        self.z_buffer.add(z)

        if self.cfg.train.relabel_ratio is not None:
            mask = torch.rand((self.cfg.train.batch_size, 1), device=self.device) <= self.cfg.train.relabel_ratio
            train_z = torch.where(mask, z, train_z)

        q_loss_coef = self.cfg.train.q_loss_coef if self.cfg.train.q_loss_coef > 0 else None
        clip_grad_norm = self.cfg.train.clip_grad_norm if self.cfg.train.clip_grad_norm > 0 else None

        metrics.update(
            self.update_fb(
                obs=train_obs,
                action=train_action,
                discount=discount,
                next_obs=train_next_obs,
                goal=train_next_obs,
                z=train_z,
                q_loss_coef=q_loss_coef,
                clip_grad_norm=clip_grad_norm,
            )
        )
        metrics.update(
            self.update_critic(
                obs=train_obs,
                action=train_action,
                discount=discount,
                next_obs=train_next_obs,
                z=train_z,
            )
        )
        metrics.update(
            self.update_actor(
                obs=train_obs,
                action=train_action,
                z=train_z,
                clip_grad_norm=clip_grad_norm,
            )
        )

        with torch.no_grad():
            _soft_update_params(
                self._forward_map_paramlist,
                self._target_forward_map_paramlist,
                self.cfg.train.fb_target_tau,
            )
            _soft_update_params(
                self._backward_map_paramlist,
                self._target_backward_map_paramlist,
                self.cfg.train.fb_target_tau,
            )
            _soft_update_params(
                self._critic_map_paramlist,
                self._target_critic_map_paramlist,
                self.cfg.train.critic_target_tau,
            )

        return metrics

    @torch.compiler.disable
    def gradient_penalty_wgan(
        self,
        real_obs: torch.Tensor,
        real_z: torch.Tensor,
        fake_obs: torch.Tensor,
        fake_z: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = real_obs.shape[0]
        alpha = torch.rand(batch_size, 1, device=real_obs.device)
        interpolates = torch.cat(
            [
                (alpha * real_obs + (1 - alpha) * fake_obs).requires_grad_(True),
                (alpha * real_z + (1 - alpha) * fake_z).requires_grad_(True),
            ],
            dim=1,
        )
        d_interpolates = self._model._discriminator.compute_logits(
            interpolates[:, 0 : real_obs.shape[1]], interpolates[:, real_obs.shape[1] :]
        )
        gradients = autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def update_discriminator(
        self,
        expert_obs: torch.Tensor,
        expert_z: torch.Tensor,
        train_obs: torch.Tensor,
        train_z: torch.Tensor,
        grad_penalty: float | None,
    ) -> Dict[str, torch.Tensor]:
        expert_logits = self._model._discriminator.compute_logits(obs=expert_obs, z=expert_z)
        unlabeled_logits = self._model._discriminator.compute_logits(obs=train_obs, z=train_z)
        # these are equivalent to binary cross entropy
        expert_loss = -torch.nn.functional.logsigmoid(expert_logits)
        unlabeled_loss = torch.nn.functional.softplus(unlabeled_logits)
        loss = torch.mean(expert_loss + unlabeled_loss)

        if grad_penalty is not None:
            wgan_gp = self.gradient_penalty_wgan(expert_obs, expert_z, train_obs, train_z)
            loss += grad_penalty * wgan_gp

        self.discriminator_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.discriminator_optimizer.step()

        with torch.no_grad():
            output_metrics = {
                "disc_loss": loss.detach(),
                "disc_expert_loss": expert_loss.detach().mean().detach(),
                "disc_train_loss": unlabeled_loss.detach().mean().detach(),
            }
            if grad_penalty is not None:
                output_metrics["disc_wgan_gp_loss"] = wgan_gp.detach()
        return output_metrics

    def update_critic(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        discount: torch.Tensor,
        next_obs: torch.Tensor,
        z: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        num_parallel = self.cfg.model.archi.critic.num_parallel
        # compute target critic
        with torch.no_grad():
            reward = self._model._discriminator.compute_reward(obs=obs, z=z)
            dist = self._model._actor(next_obs, z, self._model.cfg.actor_std)
            next_action = dist.sample(clip=self.cfg.train.stddev_clip)
            next_Qs = self._model._target_critic(next_obs, z, next_action)  # num_parallel x batch x 1
            Q_mean, Q_unc, next_V = self.get_targets_uncertainty(next_Qs, self.cfg.train.critic_pessimism_penalty)
            target_Q = reward + discount * next_V
            expanded_targets = target_Q.expand(num_parallel, -1, -1)

        # compute critic loss
        Qs = self._model._critic(obs, z, action)  # num_parallel x batch x (1 or n_bins)
        critic_loss = 0.5 * num_parallel * F.mse_loss(Qs, expanded_targets)

        # optimize critic
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        with torch.no_grad():
            output_metrics = {
                "target_Q": target_Q.mean().detach(),
                "Q1": Qs.mean().detach(),
                "mean_next_Q": Q_mean.mean().detach(),
                "unc_Q": Q_unc.mean().detach(),
                "critic_loss": critic_loss.mean().detach(),
                "mean_disc_reward": reward.mean().detach(),
            }
        return output_metrics

    def update_actor(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        z: torch.Tensor,
        clip_grad_norm: float | None,
    ) -> Dict[str, torch.Tensor]:
        dist = self._model._actor(obs, z, self._model.cfg.actor_std)
        action = dist.sample(clip=self.cfg.train.stddev_clip)

        # compute discriminator reward loss
        Qs_discriminator = self._model._critic(obs, z, action)  # num_parallel x batch x (1 or n_bins)
        _, _, Q_discriminator = self.get_targets_uncertainty(Qs_discriminator, self.cfg.train.actor_pessimism_penalty)  # batch

        # compute fb reward loss
        Fs = self._model._forward_map(obs, z, action)  # num_parallel x batch x z_dim
        Qs_fb = (Fs * z).sum(-1)  # num_parallel x batch
        _, _, Q_fb = self.get_targets_uncertainty(Qs_fb, self.cfg.train.actor_pessimism_penalty)  # batch

        weight = Q_fb.abs().mean().detach() if self.cfg.train.scale_reg else 1.0
        actor_loss = -Q_discriminator.mean() * self.cfg.train.reg_coeff * weight - Q_fb.mean()

        # optimize actor
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self._model._actor.parameters(), clip_grad_norm)
        self.actor_optimizer.step()

        with torch.no_grad():
            output_metrics = {
                "actor_loss": actor_loss.detach(),
                "Q_discriminator": Q_discriminator.mean().detach(),
                "Q_fb": Q_fb.mean().detach(),
            }
        return output_metrics
