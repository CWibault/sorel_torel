from collections import namedtuple
from dataclasses import dataclass, asdict
from datetime import datetime
from functools import partial
import os
import warnings

import distrax
import flax.linen as nn
from flax.linen.initializers import constant, uniform
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import numpy as onp
import optax
import tyro
import wandb


r"""
     |\  __
     \| /_/
      \|
    ___|_____
    \       /
     \     /
      \___/     Preliminaries
"""

AgentTrainState = namedtuple("AgentTrainState", "actor vec_q vec_q_target alpha")


def sym(scale):
    def _init(*args, **kwargs):
        return uniform(2 * scale)(*args, **kwargs) - scale

    return _init


class SoftQNetwork(nn.Module):
    @nn.compact
    def __call__(self, obs, action):
        x = jnp.concatenate([obs, action], axis=-1)
        for _ in range(3):
            x = nn.Dense(256, bias_init=constant(0.1))(x)
            x = nn.relu(x)
        q = nn.Dense(1, kernel_init=sym(3e-3), bias_init=sym(3e-3))(x)
        return q.squeeze(-1)


class VectorQ(nn.Module):
    num_critics: int

    @nn.compact
    def __call__(self, obs, action):
        vmap_critic = nn.vmap(
            SoftQNetwork,
            variable_axes={"params": 0},  # Parameters not shared between critics
            split_rngs={"params": True, "dropout": True},  # Different initializations
            in_axes=None,
            out_axes=-1,
            axis_size=self.num_critics,
        )
        q_values = vmap_critic()(obs, action)
        return q_values


class TanhGaussianActor(nn.Module):
    num_actions: int
    log_std_max: float = 2.0
    log_std_min: float = -5.0

    @nn.compact
    def __call__(self, x):
        for _ in range(3):
            x = nn.Dense(256, bias_init=constant(0.1))(x)
            x = nn.relu(x)
        log_std = nn.Dense(
            self.num_actions, kernel_init=sym(1e-3), bias_init=sym(1e-3)
        )(x)
        std = jnp.exp(jnp.clip(log_std, self.log_std_min, self.log_std_max))
        mean = nn.Dense(self.num_actions, kernel_init=sym(1e-3), bias_init=sym(1e-3))(x)
        pi = distrax.Transformed(
            distrax.Normal(mean, std),
            distrax.Tanh(),
        )
        return pi


class EntropyCoef(nn.Module):
    ent_coef_init: float = 1.0

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        log_ent_coef = self.param(
            "log_ent_coef",
            init_fn=lambda key: jnp.full((), jnp.log(self.ent_coef_init)),
        )
        return log_ent_coef


def create_train_state(args, rng, network, dummy_input):
    return TrainState.create(apply_fn=network.apply,
                             params=network.init(rng, *dummy_input),
                             tx=optax.adam(args.lr, eps=1e-5))


def sample_from_buffer(buffer, batch_size, rng):
    """Sample a batch from the buffer."""
    idxs = jax.random.randint(rng, (batch_size,), 0, len(buffer.obs))
    return jax.tree_map(lambda x: x[idxs], buffer)


r"""
          __/)
       .-(__(=:
    |\ |    \)
    \ ||
     \||
      \|
    ___|_____
    \       /
     \     /
      \___/     Agent
"""


def make_train_step(args, actor_apply_fn, q_apply_fn, alpha_apply_fn, dataset, rollout_fn):
    """Make JIT-compatible agent train step with model-based rollouts."""

    def _train_step(runner_state, _):
        rng, agent_state, rollout_buffer = runner_state

        # --- Update model buffer ---
        params = agent_state.actor.params
        policy_fn = lambda obs, rng: actor_apply_fn(params, obs).sample(seed=rng)
        rng, rng_buffer = jax.random.split(rng)
        rollout_buffer = jax.lax.cond(
            agent_state.actor.step % args.rollout_interval == 0,
            lambda: rollout_fn(rng_buffer, policy_fn, rollout_buffer),
            lambda: rollout_buffer,
        )

        # --- Sample batch ---
        rng, rng_dataset, rng_rollout = jax.random.split(rng, 3)
        dataset_size = int(args.batch_size * args.dataset_sample_ratio)
        rollout_size = args.batch_size - dataset_size
        dataset_batch = sample_from_buffer(dataset, dataset_size, rng_dataset)
        rollout_batch = sample_from_buffer(rollout_buffer, rollout_size, rng_rollout)
        batch = jax.tree_map(
            lambda x, y: jnp.concatenate([x, y]), dataset_batch, rollout_batch
        )

        # --- Update alpha ---
        @jax.value_and_grad
        def _alpha_loss_fn(params, rng):
            def _compute_entropy(rng, transition):
                pi = actor_apply_fn(agent_state.actor.params, transition.obs)
                _, log_pi = pi.sample_and_log_prob(seed=rng)
                return -log_pi.sum()

            log_alpha = alpha_apply_fn(params)
            rng = jax.random.split(rng, args.batch_size)
            entropy = jax.vmap(_compute_entropy)(rng, batch).mean()
            target_entropy = -batch.action.shape[-1]
            return log_alpha * (entropy - target_entropy)

        rng, rng_alpha = jax.random.split(rng)
        alpha_loss, alpha_grad = _alpha_loss_fn(agent_state.alpha.params, rng_alpha)
        updated_alpha = agent_state.alpha.apply_gradients(grads=alpha_grad)
        agent_state = agent_state._replace(alpha=updated_alpha)
        alpha = jnp.exp(alpha_apply_fn(agent_state.alpha.params))

        # --- Update actor ---
        @partial(jax.value_and_grad, has_aux=True)
        def _actor_loss_function(params, rng):
            def _compute_loss(rng, transition):
                pi = actor_apply_fn(params, transition.obs)
                sampled_action, log_pi = pi.sample_and_log_prob(seed=rng)
                log_pi = log_pi.sum()
                q_values = q_apply_fn(
                    agent_state.vec_q.params, transition.obs, sampled_action
                )
                q_min = jnp.min(q_values)
                return -q_min + alpha * log_pi, -log_pi, q_min, q_values.std()

            rng = jax.random.split(rng, args.batch_size)
            loss, entropy, q_min, q_std = jax.vmap(_compute_loss)(rng, batch)
            return loss.mean(), (entropy.mean(), q_min.mean(), q_std.mean())

        rng, rng_actor = jax.random.split(rng)
        (actor_loss, (entropy, q_min, q_std)), actor_grad = _actor_loss_function(
            agent_state.actor.params, rng_actor
        )
        updated_actor = agent_state.actor.apply_gradients(grads=actor_grad)
        agent_state = agent_state._replace(actor=updated_actor)

        # --- Update Q target network ---
        updated_q_target_params = optax.incremental_update(
            agent_state.vec_q.params,
            agent_state.vec_q_target.params,
            args.polyak_step_size,
        )
        updated_q_target = agent_state.vec_q_target.replace(
            step=agent_state.vec_q_target.step + 1, params=updated_q_target_params
        )
        agent_state = agent_state._replace(vec_q_target=updated_q_target)

        # --- Compute targets ---
        def _sample_next_v(rng, transition):
            next_pi = actor_apply_fn(agent_state.actor.params, transition.next_obs)
            # Note: Important to use sample_and_log_prob here for numerical stability
            # See https://github.com/deepmind/distrax/issues/7
            next_action, log_next_pi = next_pi.sample_and_log_prob(seed=rng)
            # Minimum of the target Q-values
            next_q = q_apply_fn(
                agent_state.vec_q_target.params, transition.next_obs, next_action
            )
            return next_q.min(-1) - alpha * log_next_pi.sum(-1)

        rng, rng_next_v = jax.random.split(rng)
        rng_next_v = jax.random.split(rng_next_v, args.batch_size)
        next_v_target = jax.vmap(_sample_next_v)(rng_next_v, batch)
        target = batch.reward + args.gamma * (1 - batch.done) * next_v_target

        # --- Update critics ---
        @jax.value_and_grad
        def _q_loss_fn(params):
            q_pred = q_apply_fn(params, batch.obs, batch.action)
            return jnp.square((q_pred - jnp.expand_dims(target, -1))).sum(-1).mean()

        critic_loss, critic_grad = _q_loss_fn(agent_state.vec_q.params)
        updated_q = agent_state.vec_q.apply_gradients(grads=critic_grad)
        agent_state = agent_state._replace(vec_q=updated_q)

        num_done = jnp.sum(batch.done)
        loss = {
            "critic_loss": critic_loss,
            "actor_loss": actor_loss,
            "alpha_loss": alpha_loss,
            "entropy": entropy,
            "alpha": alpha,
            "q_min": q_min,
            "q_std": q_std,
            "terminations/num_done": num_done,
            "terminations/done_ratio": num_done / batch.done.shape[0],
        }
        return (rng, agent_state, rollout_buffer), loss

    return _train_step
