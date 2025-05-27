"""
Hard-coded termination functions for world model - in future, could be learned. 
This code is adapted from https://github.com/yihaosun1124/OfflineRL-Kit/blob/6e578d13568fa934096baa2ca96e38e1fa44a233/offlinerlkit/utils/termination_fns.py#L123 - thank you to the authors!
"""

import jax.numpy as jnp


def termination_fn_cartpole(obs):
    assert len(obs.shape) == 1
    x_threshold = 2.4
    theta_threshold_radians = 12 * 2 * jnp.pi / 360
    done1 = jnp.logical_or(obs[0] < -x_threshold,
                           obs[0] > x_threshold)
    done2 = jnp.logical_or(obs[2] < -theta_threshold_radians,
                           obs[2] > theta_threshold_radians)
    done = jnp.logical_or(done1, done2)
    return done


def termination_fn_halfcheetah(obs):
    assert len(obs.shape) == 1
    not_done = jnp.logical_and(jnp.all(obs > -100), jnp.all(obs < 100))
    done = ~not_done
    return done


def termination_fn_hopper(obs):
    assert len(obs.shape) == 1
    height = obs[0]
    angle = obs[1]
    not_done = (jnp.isfinite(obs).all()
                * jnp.abs(obs[1:] < 100).all()
                * (height > 0.7)
                * (jnp.abs(angle) < 0.2))
    done = ~not_done
    return done


def termination_fn_pendulum(obs):
    assert len(obs.shape) == 1
    done = jnp.array(False)
    return done


def termination_fn_walker2d(obs):
    assert len(obs.shape) == 1
    height = obs[0]
    angle = obs[1]
    not_done = (jnp.logical_and(jnp.all(obs > -100), jnp.all(obs < 100))
                * (height > 0.8)
                * (height < 2.0)
                * (angle > -1.0)
                * (angle < 1.0))
    done = ~not_done
    return done


def get_termination_fn(task):
    if "cartpole" in task:
        return termination_fn_cartpole
    elif "halfcheetah" in task: 
        return termination_fn_halfcheetah
    elif "hopper" in task:
        return termination_fn_hopper
    elif "pendulum" in task:
        return termination_fn_pendulum
    elif "walker2d" in task:
        return termination_fn_walker2d
    else:
        raise ValueError(f"Unknown task: {task}")