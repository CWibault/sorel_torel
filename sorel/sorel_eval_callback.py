"""
Callback function to evaluate the approximate regret.  

Callback function also determines the true regret (used in the SOReL paper to validate the approximate regret as a proxy for the true regret). In practice, only the optimal hyper-parameter policy would be deployed. 
"""

import jax
import jax.numpy as jnp

from utils.evaluate.eval_recurrent_policy import eval_recurrent_policy
from utils.regret_utils import infinite_horizon_discounted_return, get_regret, scale_regret
from utils.logging import wandb_log_info


def make_sorel_eval_callback(args,
                             model_env, 
                             model_env_params, 
                             true_env, 
                             true_env_params, 
                             rnn_size,
                             num_elites,
                             num_eval_workers, 
                             discount_factor, 
                             min_reward, 
                             max_reward):
    
    num_eval_workers = int(num_eval_workers)
    max_steps = model_env.max_steps_in_episode
    elites_batched_params = model_env_params.replace(model_idx=jnp.arange(num_elites),
                                                    stochasticity=jnp.array([0] * num_elites), # deterministic 
                                                    max_steps_in_episode=jnp.array([max_steps] * num_elites))

    def get_expected_regret(actor, rng):
        # --- calculate variance of means ---
        _, discounted_returns, _ = jax.vmap(eval_recurrent_policy, in_axes=(None, None, 0, None, None, None, None, None))(rng, model_env, elites_batched_params, actor, rnn_size, discount_factor, num_elites, max_steps)
        infinite_discounted_returns = infinite_horizon_discounted_return(max_steps, discount_factor, discounted_returns)
        variance_of_means = infinite_discounted_returns.var(ddof=1)
        mean_of_means = infinite_discounted_returns.mean()

        def variance_single_model(model_idx):
            eval_workers_batched_params = model_env_params.replace(model_idx=jnp.full((num_eval_workers,), model_idx),
                                                                   stochasticity=jnp.array([1] * num_eval_workers),
                                                                   max_steps_in_episode=jnp.array([max_steps] * num_eval_workers))
            returns, discounted_returns, episode_lengths = eval_recurrent_policy(rng, 
                                                            model_env, 
                                                            eval_workers_batched_params, 
                                                            actor, 
                                                            rnn_size, 
                                                            discount_factor, 
                                                            num_eval_workers, 
                                                            max_steps)
            infinite_discounted_returns = infinite_horizon_discounted_return(max_steps, discount_factor, discounted_returns)
            return  infinite_discounted_returns.var(ddof=1), infinite_discounted_returns.mean(), returns.mean(), episode_lengths.mean()
        
        # --- calculate mean of variances --- 
        variances_single_model, means_single_model, mean_returns_single_model, mean_episode_lengths_single_model = jax.vmap(variance_single_model, in_axes=(0,))(jnp.arange(num_elites))
        norm_single_model_regrets = get_regret(means_single_model, discount_factor, min_reward, max_reward)
        mean_of_variances = variances_single_model.mean()
        mean_of_means = (means_single_model.mean() + mean_of_means) / 2

        expected_regret_based_on_variance = 2 * jnp.sqrt(variance_of_means + mean_of_variances)
        norm_expected_regret_based_on_variance = scale_regret(expected_regret_based_on_variance, discount_factor, min_reward, max_reward)
        norm_expected_regret_based_on_median = get_regret(jnp.median(means_single_model), discount_factor, min_reward, max_reward)
        norm_expected_regret_based_on_max = jnp.max(norm_single_model_regrets)
        norm_expected_regret_based_on_min = jnp.min(norm_single_model_regrets)
        norm_expected_regret_based_on_mean = get_regret(mean_of_means, discount_factor, min_reward, max_reward)

        norm_expected_regret = jnp.maximum(norm_expected_regret_based_on_variance, norm_expected_regret_based_on_median)
        mean_returns = mean_returns_single_model.mean()
        mean_episode_lengths = mean_episode_lengths_single_model.mean()
        return norm_expected_regret, norm_expected_regret_based_on_variance, norm_expected_regret_based_on_median, norm_expected_regret_based_on_max, norm_expected_regret_based_on_min, norm_expected_regret_based_on_mean, mean_returns, mean_episode_lengths
    
    def get_true_regret(actor, rng):
        returns, discounted_returns, episode_lengths = eval_recurrent_policy(rng, true_env, true_env_params, actor, rnn_size, discount_factor, num_eval_workers, max_steps)
        infinite_discounted_returns = infinite_horizon_discounted_return(max_steps, discount_factor, discounted_returns)
        norm_true_regret = get_regret(infinite_discounted_returns.mean(), discount_factor, min_reward, max_reward)
        return norm_true_regret, returns.mean(), episode_lengths.mean()

    def sorel_eval_callback(train_state, recurrent_actor, rng):
        norm_expected_regret, norm_expected_regret_based_on_variance, norm_expected_regret_based_on_median, norm_expected_regret_based_on_max, norm_expected_regret_based_on_min, norm_expected_regret_based_on_mean, offline_mean_returns, offline_mean_episode_lengths = get_expected_regret(recurrent_actor, rng)
        norm_true_regret, true_mean_returns, true_mean_episode_lengths = get_true_regret(recurrent_actor, rng)

        jax.debug.print("Norm Expected Regret: {}", norm_expected_regret)
        jax.debug.print("Norm True Regret: {}", norm_true_regret)

        log_dict = {"norm_expected_regret": norm_expected_regret,
                        "norm_expected_regret_based_on_variance": norm_expected_regret_based_on_variance,
                        "norm_expected_regret_based_on_median": norm_expected_regret_based_on_median,
                        "norm_expected_regret_based_on_max": norm_expected_regret_based_on_max,
                        "norm_expected_regret_based_on_min": norm_expected_regret_based_on_min,
                        "norm_expected_regret_based_on_mean": norm_expected_regret_based_on_mean,
                        "norm_true_regret": norm_true_regret, 
                        "offline_returns": offline_mean_returns,
                        "true_returns": true_mean_returns,
                        "true_episode_lengths": true_mean_episode_lengths}
        if args.log:
            wandb_log_info(log_dict, args.task + "_" + args.algo)
        return log_dict
        
    return sorel_eval_callback
