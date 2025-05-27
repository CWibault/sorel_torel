"""
Script containing class to wrap dynamics model into Gymnax environment. 
"""

import warnings
from copy import copy
from collections import namedtuple
from dataclasses import dataclass

import flax.linen as nn
from flax import struct
from gymnax.environments import spaces
from gymnax.environments.environment import Environment as GymnaxEnv
from jax import numpy as jnp
import jax


@struct.dataclass
class EnvParams:
    model_idx: int = -1
    stochasticity: int = 1 # 1 for stochastic, 0 for deterministic
    max_steps_in_episode: int = 1000


@struct.dataclass
class EnvState():
    obs: jnp.ndarray
    time: int


class DynamicsEnv(GymnaxEnv):

    def __init__(self, 
        elite_dynamics_model,
        elite_params,
        num_elites,
        reset_fn,
        termination_fn,
        action_dim,
        obs_dim,
        min_reward, 
        max_reward,
        action_type = "continuous", # "discrete" or "continuous"
        min_action = -1, # minimum action value (required for continuous)
        max_action = 1, # maximum action value (required for continuous)
        max_steps_in_episode = 1000): 

        self.elite_dynamics_model = elite_dynamics_model
        self.elite_params = elite_params
        self.num_elites = num_elites
        self.reset_fn = reset_fn
        self.termination_fn = termination_fn
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.min_reward = min_reward
        self.max_reward = max_reward
        self.action_type = action_type
        self.min_action = min_action
        self.max_action = max_action
        self.max_steps_in_episode = max_steps_in_episode
        # potential to add prior if dataset is not diverse
        self.step_penalty_coef = 0.0 # mopo 
        self.terminate_when_uncertain = False # morel 
        self.uncertainty_threshold = 1.0 # morel 
        self.uncertainty_penalty_offset = -200 # morel 

    @property
    def default_params(self):
        return EnvParams(max_steps_in_episode=self.max_steps_in_episode)

    def step_env(self, key, state, action, params):
        key, key_dynamics, key_noise = jax.random.split(key, 3)

        # --- ensemble predictions ---
        obs = state.obs
        obs = jnp.array(obs).reshape(1, -1)
        action = jnp.array(action).reshape(1, -1)
        obs_ac = jnp.concatenate([obs, action], axis = 1)
        ensemble_mean, ensemble_logvar = self.elite_dynamics_model.apply(
            self.elite_params, obs_ac)
        ensemble_std = jnp.exp(0.5 * ensemble_logvar)

        # --- sample from ensemble ---
        def random_model_idx(_):
            return jax.random.randint(key_dynamics, (), 0, self.num_elites)
        def use_param_model_idx(_):
            return params.model_idx
        model_idx = jax.lax.cond(
            params.model_idx == -1, 
            random_model_idx,
            use_param_model_idx,
            operand=None)
        sample_mean, sample_std = ensemble_mean[model_idx], ensemble_std[model_idx]

        # --- sample from distribution ---
        def stochastic_sample(_):
            return sample_mean + jax.random.normal(key=key_noise, shape=sample_mean.shape) * sample_std
        def deterministic_sample(_):
            return sample_mean
        samples = jax.lax.cond(
            params.stochasticity == 1, 
            stochastic_sample,
            deterministic_sample,
            operand=None)
        
        # --- calculate next obs, clip reward such that in distribution, and update state ---
        delta_obs, reward = samples[..., :-1], samples[..., -1:]
        obs += delta_obs
        obs = obs.squeeze()
        reward = jnp.clip(reward, self.min_reward, self.max_reward) 
        reward = reward.squeeze()
        state = EnvState(obs=obs, time=state.time + 1)
        done = self.is_terminal(state, params)

        # --- apply mopo or (amended version of) morel [amended to bypass computing discrepancy] ---
        step_penalty = jnp.max(jnp.linalg.norm(ensemble_std, axis=-1))
        reward -= self.step_penalty_coef * step_penalty
        if self.terminate_when_uncertain == True:
            dists = ensemble_mean[:, None, :] - ensemble_mean[None, :, :]
            dists = jnp.linalg.norm(dists, axis=-1)
            is_uncertain = jnp.max(dists) > self.uncertainty_threshold
            reward = jnp.where(is_uncertain, self.min_reward + self.uncertainty_penalty_offset, reward)
            done |= is_uncertain

        info = {}
        return obs, state, reward, done, info

    def reset_env(self, key, params):
        obs = self.reset_fn(key)
        state = EnvState(obs=obs, time=0)
        return obs, state

    def get_obs(self, state):
        return state.obs

    def is_terminal(self, state, params):
        done = self.termination_fn(state.obs)
        done |= state.time >= params.max_steps_in_episode
        return done

    @property
    def name(self):
        return "DynamicsEnv"

    def action_space(self, params):
        if self.action_type == "continuous":
            return spaces.Box(
                low=self.min_action, 
                high=self.max_action, 
                shape=(self.num_actions,))
        elif self.action_type == "discrete":
            return spaces.Discrete(
                self.num_actions)

    def observation_space(self, params):
        return spaces.Box(low=-jnp.inf, high=jnp.inf, shape=(self.num_observations,))

    @property
    def num_actions(self) -> int:
        return self.action_dim

    @property
    def num_observations(self) -> int:
        return self.obs_dim