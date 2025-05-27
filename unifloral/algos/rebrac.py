from collections import namedtuple
from dataclasses import dataclass
import os

import distrax
import flax.linen as nn
from flax.linen.initializers import constant, uniform
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

AgentTrainState = namedtuple("AgentTrainState", "actor actor_target dual_q dual_q_target")
Transition = namedtuple("Transition", "obs action reward next_obs next_action done")


def sym(scale):
    def _init(*args, **kwargs):
        return uniform(2 * scale)(*args, **kwargs) - scale

    return _init


class SoftQNetwork(nn.Module):
    obs_mean: jax.Array
    obs_std: jax.Array
    use_ln: bool
    norm_obs: bool

    @nn.compact
    def __call__(self, obs, action):
        if self.norm_obs:
            obs = (obs - self.obs_mean) / (self.obs_std + 1e-3)
        x = jnp.concatenate([obs, action], axis=-1)
        for _ in range(3):
            x = nn.Dense(256, bias_init=constant(0.1))(x)
            x = nn.relu(x)
            x = nn.LayerNorm()(x) if self.use_ln else x
        q = nn.Dense(1, bias_init=sym(3e-3), kernel_init=sym(3e-3))(x)
        return q.squeeze(-1)


class DualQNetwork(nn.Module):
    obs_mean: jax.Array
    obs_std: jax.Array
    use_ln: bool
    norm_obs: bool

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
        q_fn = vmap_critic(self.obs_mean, self.obs_std, self.use_ln, self.norm_obs)
        q_values = q_fn(obs, action)
        return q_values


class DeterministicTanhActor(nn.Module):
    num_actions: int
    obs_mean: jax.Array
    obs_std: jax.Array
    use_ln: bool
    norm_obs: bool

    @nn.compact
    def __call__(self, x):
        if self.norm_obs:
            x = (x - self.obs_mean) / (self.obs_std + 1e-3)
        for _ in range(3):
            x = nn.Dense(256, bias_init=constant(0.1))(x)
            x = nn.relu(x)
            x = nn.LayerNorm()(x) if self.use_ln else x
        init_fn = sym(1e-3)
        action = nn.Dense(self.num_actions, bias_init=init_fn, kernel_init=init_fn)(x)
        pi = distrax.Transformed(
            distrax.Deterministic(action),
            distrax.Tanh())
        return pi


def create_train_state(args, rng, network, dummy_input):
    return TrainState.create(
        apply_fn=network.apply,
        params=network.init(rng, *dummy_input),
        tx=optax.adam(args.lr, eps=1e-5))


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


def make_train_step(args, actor_apply_fn, q_apply_fn, dataset):
    """Make JIT-compatible agent train step."""

    def _train_step(runner_state, _):
        rng, agent_state = runner_state

        # --- Sample batch ---
        rng, rng_batch = jax.random.split(rng)
        batch_indices = jax.random.randint(
            rng_batch, (args.batch_size,), 0, len(dataset.obs)
        )
        batch = jax.tree_util.tree_map(lambda x: x[batch_indices], dataset)

        # --- Update critics ---
        def _update_critics(runner_state, _):
            rng, agent_state = runner_state

            def _compute_target(rng, transition):
                next_obs = transition.next_obs

                # --- Sample noised action ---
                next_pi = actor_apply_fn(agent_state.actor_target.params, next_obs)
                rng, rng_action, rng_noise = jax.random.split(rng, 3)
                action = next_pi.sample(seed=rng_action)
                noise = jax.random.normal(rng_noise, shape=action.shape)
                noise *= args.policy_noise
                noise = jnp.clip(noise, -args.noise_clip, args.noise_clip)
                action = jnp.clip(action + noise, -1, 1)
                bc_loss = jnp.square(action - transition.next_action).sum()

                # --- Compute targets ---
                target_q = q_apply_fn(
                    agent_state.dual_q_target.params, next_obs, action
                )
                next_q_value = jnp.min(target_q) - args.critic_bc_coef * bc_loss
                next_q_value = (1.0 - transition.done) * next_q_value
                return transition.reward + args.gamma * next_q_value, bc_loss

            rng, rng_targets = jax.random.split(rng)
            rng_targets = jax.random.split(rng_targets, args.batch_size)
            target_fn = jax.vmap(_compute_target)
            targets, bc_loss = target_fn(rng_targets, batch)

            # --- Compute critic loss ---
            @jax.value_and_grad
            def _q_loss_fn(params):
                q_pred = q_apply_fn(params, batch.obs, batch.action)
                q_loss = jnp.square(q_pred - jnp.expand_dims(targets, axis=-1)).sum(-1)
                return q_loss.mean()

            q_loss, q_grad = _q_loss_fn(agent_state.dual_q.params)
            updated_q_state = agent_state.dual_q.apply_gradients(grads=q_grad)
            agent_state = agent_state._replace(dual_q=updated_q_state)
            return (rng, agent_state), (q_loss, bc_loss)

        # --- Iterate critic update ---
        (rng, agent_state), (q_loss, critic_bc_loss) = jax.lax.scan(
            _update_critics,
            (rng, agent_state),
            None,
            length=args.num_critic_updates_per_step,
        )

        # --- Update actor ---
        def _actor_loss_function(params):
            def _transition_loss(transition):
                pi = actor_apply_fn(params, transition.obs)
                pi_action = pi.sample(seed=None)
                q = q_apply_fn(agent_state.dual_q.params, transition.obs, pi_action)
                bc_loss = jnp.square(pi_action - transition.action).sum()
                return q.min(), bc_loss

            q, bc_loss = jax.vmap(_transition_loss)(batch)
            lambda_ = 1.0 / (jnp.abs(q).mean() + 1e-7)
            lambda_ = jax.lax.stop_gradient(lambda_)
            actor_loss = -lambda_ * q.mean() + args.actor_bc_coef * bc_loss.mean()
            return actor_loss.mean(), (q.mean(), lambda_.mean(), bc_loss.mean())

        loss_fn = jax.value_and_grad(_actor_loss_function, has_aux=True)
        (actor_loss, (q_mean, lambda_, bc_loss)), actor_grad = loss_fn(
            agent_state.actor.params
        )
        agent_state = agent_state._replace(
            actor=agent_state.actor.apply_gradients(grads=actor_grad)
        )

        # --- Update target networks ---
        def _update_target(state, target_state):
            new_target_params = optax.incremental_update(
                state.params, target_state.params, args.polyak_step_size
            )
            return target_state.replace(
                step=target_state.step + 1, params=new_target_params
            )

        agent_state = agent_state._replace(
            actor_target=_update_target(agent_state.actor, agent_state.actor_target),
            dual_q_target=_update_target(agent_state.dual_q, agent_state.dual_q_target),
        )

        loss = {
            "actor_loss": actor_loss,
            "q_loss": q_loss.mean(),
            "q_mean": q_mean,
            "lambda": lambda_,
            "bc_loss": bc_loss,
            "critic_bc_loss": critic_bc_loss.mean(),
        }
        return (rng, agent_state), loss

    return _train_step

