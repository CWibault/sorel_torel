from collections import namedtuple
from dataclasses import dataclass
from functools import partial
import os

import distrax
import flax.linen as nn
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
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

AgentTrainState = namedtuple("AgentTrainState", "actor dual_q dual_q_target value")
Transition = namedtuple("Transition", "obs action reward next_obs done")


class SoftQNetwork(nn.Module):
    obs_mean: jax.Array
    obs_std: jax.Array

    @nn.compact
    def __call__(self, obs, action):
        obs = (obs - self.obs_mean) / (self.obs_std + 1e-3)
        x = jnp.concatenate([obs, action], axis=-1)
        for _ in range(2):
            x = nn.Dense(256)(x)
            x = nn.relu(x)
        q = nn.Dense(1)(x)
        return q.squeeze(-1)


class DualQNetwork(nn.Module):
    obs_mean: jax.Array
    obs_std: jax.Array

    @nn.compact
    def __call__(self, obs, action):
        vmap_critic = nn.vmap(
            SoftQNetwork,
            variable_axes={"params": 0},  # Parameters not shared between critics
            split_rngs={"params": True, "dropout": True},  # Different initializations
            in_axes=None,
            out_axes=-1,
            axis_size=2,  # Two Q networks
        )
        q_values = vmap_critic(self.obs_mean, self.obs_std)(obs, action)
        return q_values


class StateValueFunction(nn.Module):
    obs_mean: jax.Array
    obs_std: jax.Array

    @nn.compact
    def __call__(self, x):
        x = (x - self.obs_mean) / (self.obs_std + 1e-3)
        for _ in range(2):
            x = nn.Dense(256)(x)
            x = nn.relu(x)
        v = nn.Dense(1)(x)
        return v.squeeze(-1)


class TanhGaussianActor(nn.Module):
    num_actions: int
    obs_mean: jax.Array
    obs_std: jax.Array
    log_std_max: float = 2.0
    log_std_min: float = -20.0

    @nn.compact
    def __call__(self, x, eval=False):
        x = (x - self.obs_mean) / (self.obs_std + 1e-3)
        for _ in range(2):
            x = nn.Dense(256)(x)
            x = nn.relu(x)
        x = nn.Dense(self.num_actions)(x)
        x = nn.tanh(x)
        if eval:
            return distrax.Deterministic(x)
        logstd = self.param(
            "logstd",
            init_fn=lambda key: jnp.zeros(self.num_actions, dtype=jnp.float32),
        )
        std = jnp.exp(jnp.clip(logstd, self.log_std_min, self.log_std_max))
        return distrax.Normal(x, std)


def create_train_state(args, rng, network, dummy_input):
    lr_schedule = optax.cosine_decay_schedule(args.lr, args.num_updates)
    return TrainState.create(
        apply_fn=network.apply,
        params=network.init(rng, *dummy_input),
        tx=optax.adam(lr_schedule, eps=1e-5),
    )


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


def make_train_step(args, actor_apply_fn, q_apply_fn, value_apply_fn, dataset):
    """Make JIT-compatible agent train step."""

    def _train_step(runner_state, _):
        rng, agent_state = runner_state

        # --- Sample batch ---
        rng, rng_batch = jax.random.split(rng)
        batch_indices = jax.random.randint(
            rng_batch, (args.batch_size,), 0, len(dataset.obs)
        )
        batch = jax.tree_util.tree_map(lambda x: x[batch_indices], dataset)

        # --- Update Q target network ---
        updated_q_target_params = optax.incremental_update(
            agent_state.dual_q.params,
            agent_state.dual_q_target.params,
            args.polyak_step_size,
        )
        updated_q_target = agent_state.dual_q_target.replace(
            step=agent_state.dual_q_target.step + 1, params=updated_q_target_params
        )
        agent_state = agent_state._replace(dual_q_target=updated_q_target)

        # --- Compute targets ---
        v_target = q_apply_fn(agent_state.dual_q_target.params, batch.obs, batch.action)
        v_target = v_target.min(-1)
        next_v_target = value_apply_fn(agent_state.value.params, batch.next_obs)
        q_targets = batch.reward + args.gamma * (1 - batch.done) * next_v_target

        # --- Update Q and value functions ---
        def _q_loss_fn(params):
            # Compute loss for both critics
            q_pred = q_apply_fn(params, batch.obs, batch.action)
            q_loss = jnp.square(q_pred - jnp.expand_dims(q_targets, axis=-1)).mean()
            return q_loss

        @partial(jax.value_and_grad, has_aux=True)
        def _value_loss_fn(params):
            adv = v_target - value_apply_fn(params, batch.obs)
            # Asymmetric L2 loss
            value_loss = jnp.abs(args.iql_tau - (adv < 0.0).astype(float)) * (adv**2)
            return jnp.mean(value_loss), adv

        q_loss, q_grad = jax.value_and_grad(_q_loss_fn)(agent_state.dual_q.params)
        (value_loss, adv), value_grad = _value_loss_fn(agent_state.value.params)
        agent_state = agent_state._replace(
            dual_q=agent_state.dual_q.apply_gradients(grads=q_grad),
            value=agent_state.value.apply_gradients(grads=value_grad),
        )

        # --- Update actor ---
        exp_adv = jnp.exp(adv * args.beta).clip(max=args.exp_adv_clip)

        @jax.value_and_grad
        def _actor_loss_function(params):
            def _compute_loss(transition, exp_adv):
                pi = actor_apply_fn(params, transition.obs)
                bc_loss = -pi.log_prob(transition.action)
                return exp_adv * bc_loss.sum()

            actor_loss = jax.vmap(_compute_loss)(batch, exp_adv)
            return actor_loss.mean()

        actor_loss, actor_grad = _actor_loss_function(agent_state.actor.params)
        updated_actor = agent_state.actor.apply_gradients(grads=actor_grad)
        agent_state = agent_state._replace(actor=updated_actor)

        loss = {
            "value_loss": value_loss,
            "q_loss": q_loss,
            "actor_loss": actor_loss,
        }
        return (rng, agent_state), loss

    return _train_step

