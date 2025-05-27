import warnings

import jax
import jax.numpy as jnp
import numpy as np

def eval_policy_d4rl(rng, 
                    env,
                    env_params,
                    actor,
                    discount_factor,
                    num_eval_workers,
                    max_steps):
    """gym environment (D4RL compatible) evaluation function."""

    num_eval_workers = int(num_eval_workers)

    # --- Reset environment ---
    step = 0
    returned = np.zeros(num_eval_workers).astype(bool)
    cum_reward = np.zeros(num_eval_workers)
    discounted_return = np.zeros(num_eval_workers)
    episode_length = np.zeros(num_eval_workers)
    obs = env.reset()

    while step < max_steps and not returned.all():

        # --- select action ---
        rng, action_rng = jax.random.split(rng)
        action_rng = jax.random.split(action_rng, num_eval_workers)
        action = jax.vmap(actor)(jnp.array(obs), action_rng)
        
        # --- Take step in environment ---
        step += 1
        obs, reward, done, info = env.step(np.array(action))

        # --- Track cumulative reward ---
        cum_reward += reward * ~returned
        discounted_return += reward * (discount_factor ** episode_length) * ~returned
        episode_length += 1 * ~returned
        returned |= done

    if step >= max_steps and not returned.all():
        warnings.warn("Maximum steps reached before all episodes terminated")
    return cum_reward, discounted_return, episode_length