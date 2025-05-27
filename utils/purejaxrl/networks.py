"""
Continuous and Discrete Actor-Critic Networks. 

Continuous and Discrete Recurrent Actor-Critic Networks. 
"""

import functools
import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
import distrax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence


# --- ppo networks ---
class ActorCriticDiscrete(nn.Module):
    action_dim: Sequence[int]
    layer_size: int = 64
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        elif self.activation == "tanh":
            activation = nn.tanh
        else:
            raise ValueError(f"Invalid activation function: {self.activation}")
        actor_mean = nn.Dense(self.layer_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(self.layer_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(self.layer_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        critic = activation(critic)
        critic = nn.Dense(self.layer_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)

        return pi, jnp.squeeze(critic, axis=-1)


class ActorCriticContinuous(nn.Module):
    action_dim: Sequence[int]
    layer_size: int = 256
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        elif self.activation == "tanh":
            activation = nn.tanh
        else:
            raise ValueError(f"Invalid activation function: {self.activation}")
        actor_mean = nn.Dense(self.layer_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(self.layer_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean)
        actor_logtstd = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))

        critic = nn.Dense(self.layer_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        critic = activation(critic)
        critic = nn.Dense(self.layer_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)

        return pi, jnp.squeeze(critic, axis=-1)


# --- recurrent ppo networks ---

class ActorCriticRNNDiscrete(nn.Module):
    action_dim: Sequence[int]
    rnn_size: int = 256
    layer_size: int = 256
    activation: str = "relu"

    @nn.compact
    def __call__(self, hidden, x):
        if self.activation == "relu":
            activation = nn.relu
        elif self.activation == "tanh":
            activation = nn.tanh
        else:
            raise ValueError(f"Invalid activation function: {self.activation}")
        obs, dones = x
        embedding = nn.Dense(self.rnn_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(obs)
        embedding = activation(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        actor_mean = nn.Dense(self.layer_size, kernel_init=orthogonal(2), bias_init=constant(0.0))(embedding)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean)

        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(self.layer_size, kernel_init=orthogonal(2), bias_init=constant(0.0))(embedding)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)

        return hidden, pi, jnp.squeeze(critic, axis=-1)


class ActorCriticRNNContinuous(nn.Module):
    action_dim: Sequence[int]
    rnn_size: int = 256
    layer_size: int = 256
    activation: str = "relu"

    @nn.compact
    def __call__(self, hidden, x):
        if self.activation == "relu":
            activation = nn.relu
        elif self.activation == "tanh":
            activation = nn.tanh
        else:
            raise ValueError(f"Invalid activation function: {self.activation}")
        obs, dones = x
        embedding = nn.Dense(self.rnn_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(obs)
        embedding = activation(embedding)

        rnn_in = (embedding, dones)

        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        actor_mean = nn.Dense(self.layer_size, kernel_init=orthogonal(2), bias_init=constant(0.0))(embedding)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean)

        actor_logtstd = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))

        critic = nn.Dense(self.layer_size, kernel_init=orthogonal(2), bias_init=constant(0.0))(embedding)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)

        return hidden, pi, jnp.squeeze(critic, axis=-1)
    

class ScannedRNN(nn.Module):
    
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False})

    @nn.compact
    def __call__(self, carry, x):
        rnn_state = carry
        ins, resets = x
        rnn_state = jnp.where(resets[:, jnp.newaxis], self.initialize_carry(ins.shape[0], rnn_state.shape[-1]), rnn_state)
        new_rnn_state, y = nn.GRUCell(rnn_state.shape[-1])(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        return jnp.zeros((batch_size, hidden_size), dtype=jnp.float32)