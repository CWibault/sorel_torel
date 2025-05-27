"""
Hard-coded reset functions for world model - in future, could be learned. 
"""

import jax.numpy as jnp
import jax


def reset_fn_cartpole(key):
    obs = jax.random.uniform(key, minval=-0.05, maxval=0.05, shape=(4,))
    return obs


def reset_fn_halfcheetah(key):
    reset_noise_scale=0.1
    noise_low = -reset_noise_scale
    noise_high = reset_noise_scale
    obs = jax.random.uniform(key, (17,), minval=noise_low, maxval=noise_high)
    return obs


def reset_fn_hopper(key):
    reset_noise_scale=0.005
    noise_low = -reset_noise_scale
    noise_high = reset_noise_scale
    obs = jax.random.uniform(key, (11,), minval=noise_low, maxval=noise_high)
    offset = jnp.array((1.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    obs = obs + offset
    return obs


def reset_fn_pendulum(key):
    high = jnp.array([jnp.pi, 1])
    state = jax.random.uniform(key, shape=(2,), minval=-high, maxval=high)
    obs = jnp.array([jnp.cos(state[0]), jnp.sin(state[0]), state[1]])
    return obs


def reset_fn_walker2d(key):
    reset_noise_scale=0.005
    noise_low = -reset_noise_scale
    noise_high = reset_noise_scale
    obs = jax.random.uniform(key, (17,), minval=noise_low, maxval=noise_high)
    offset = jnp.array((1.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    obs = obs + offset
    return obs


def get_reset_fn(task):
    if "cartpole" in task:
        return reset_fn_cartpole
    elif "halfcheetah" in task: 
        return reset_fn_halfcheetah
    elif "hopper" in task:
        return reset_fn_hopper
    elif "pendulum" in task:
        return reset_fn_pendulum
    elif "walker2d" in task:
        return reset_fn_walker2d
    else:
        raise ValueError(f"Task {task} not supported")