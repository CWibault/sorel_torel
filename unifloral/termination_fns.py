"""
Hard-coded termination functions for world model - in future, could be learned. 
This code is adapted from https://github.com/yihaosun1124/OfflineRL-Kit/blob/6e578d13568fa934096baa2ca96e38e1fa44a233/offlinerlkit/utils/termination_fns.py#L123 - thank you to the authors!
"""

import jax.numpy as jnp


def termination_fn_halfcheetah(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 1
    not_done = jnp.logical_and(jnp.all(next_obs > -100), jnp.all(next_obs < 100))
    done = ~not_done
    return done


def termination_fn_hopper(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 1
    height = next_obs[0]
    angle = next_obs[1]
    not_done = (jnp.isfinite(next_obs).all()
                * jnp.abs(next_obs[1:] < 100).all()
                * (height > 0.7)
                * (jnp.abs(angle) < 0.2))
    done = ~not_done
    return done


def termination_fn_walker2d(obs, act, next_obs):
    assert len(obs.shape) == len(next_obs.shape) == len(act.shape) == 1
    height = next_obs[0]
    angle = next_obs[1]
    not_done = (jnp.logical_and(jnp.all(next_obs > -100), jnp.all(next_obs < 100))
                * (height > 0.8)
                * (height < 2.0)
                * (angle > -1.0)
                * (angle < 1.0))
    done = ~not_done
    return done


def get_termination_fn(task):
    if "halfcheetah" in task: 
        return termination_fn_halfcheetah
    elif "hopper" in task:
        return termination_fn_hopper
    elif "walker2d" in task:
        return termination_fn_walker2d
    else:
        raise ValueError(f"Unknown task: {task}")