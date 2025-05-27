import jax.numpy as jnp


def scale_regret(expected_regret, 
                 discount_factor, 
                 min_reward, 
                 max_reward):
    """Scale the regret to lie between 0 and 1."""
    min_return = min_reward / (1-discount_factor)
    max_return = max_reward / (1-discount_factor)
    max_regret = max_return - min_return
    scaled_regret = expected_regret / max_regret
    scaled_regret = jnp.maximum(scaled_regret, 0)
    scaled_regret = jnp.minimum(scaled_regret, 1)
    return scaled_regret


def get_regret(infinite_horizon_discounted_return, 
               discount_factor, 
               min_reward, 
               max_reward):
    """Obtain normalised regret from infinite horizon discounted return."""
    min_return = min_reward / (1-discount_factor)
    max_return = max_reward / (1-discount_factor)
    max_regret = max_return - min_return
    norm_regret = (max_return - infinite_horizon_discounted_return) / max_regret
    norm_regret = jnp.maximum(norm_regret, 0)
    norm_regret = jnp.minimum(norm_regret, 1)
    return norm_regret


def infinite_horizon_discounted_return(max_steps_in_episode, 
                                        discount_factor, 
                                        discounted_return):
    """Convert finite horizon discounted return to infinite horizon discounted return."""
    average_undiscounted_return = discounted_return * (1 - discount_factor) / (1 - discount_factor ** max_steps_in_episode)
    infinite_horizon_discounted_returns = discounted_return + average_undiscounted_return * (discount_factor ** max_steps_in_episode) / (1 - discount_factor)
    return infinite_horizon_discounted_returns