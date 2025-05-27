import argparse
import os
from dataclasses import dataclass

import distrax
import flax.linen as nn
from flax.linen.initializers import constant, uniform
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import optax
import tyro
import wandb

# --- additional imports ---
import numpy as np
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# --- required for loading actors ---
from unifloral.algos.mopo import SoftQNetwork, VectorQ, EntropyCoef, TanhGaussianActor
from unifloral.algos.morel import SoftQNetwork, VectorQ, EntropyCoef, TanhGaussianActor
from unifloral.algos.rebrac import SoftQNetwork, DualQNetwork, DeterministicTanhActor
from unifloral.algos.iql import SoftQNetwork, DualQNetwork, StateValueFunction, TanhGaussianActor
from unifloral.actor_wrapper import ActorWrapper 
from unifloral.evaluation import bootstrap_bandit_trials

from utils.logging import load_pkl
from utils.regret_utils import infinite_horizon_discounted_return, get_regret

try: 
    from utils.envs.make_env import make_env
    from utils.evaluate.eval_policy import eval_policy_batched
except:
    pass

try: 
    import d4rl
    import gym 
    from utils.evaluate.eval_policy_d4rl import eval_policy_d4rl
except:
    pass


@dataclass
class args:
    # --- run identification ---
    algo: str = "rebrac"
    seed: int = 0

    # --- environment and offline dataset ---
    task: str = "brax-halfcheetah-full-replay"
    n_value: int = 200000
    discount_factor: float = 0.998
    min_reward: float = -0.5
    max_reward: float = 3.5
    max_episode_steps: int = 1000

    # --- final evaluation ---
    eval_final_episodes: int = 1000


def unifloral_online_tuning(args):

    print(f"========== Evaluating {args.task} ==========")

    # --- for d4rl envs, loop over actors and evaluate --- 
    if "d4rl" in args.task: 
        true_env = gym.vector.make(args.task[5:], num_envs=args.eval_final_episodes)
        true_env_params = None
        directory = f"torel/runs/{args.algo}/{args.task}/{args.seed}"
        num_policies = 0
        episode_returns_list = []
        discounted_episode_returns_list = []
        episode_lengths_list = []
        norm_true_regrets_list = []
        for filename in os.listdir(directory):
            if filename.endswith(".pkl"):
                filepath = os.path.join(directory, filename)
                actor = load_pkl(filepath)
                rng = jax.random.PRNGKey(args.seed)
                episode_returns, discounted_episode_returns, episode_lengths = eval_policy_d4rl(rng, 
                                                                                                true_env,
                                                                                                true_env_params,
                                                                                                actor, 
                                                                                                args.discount_factor, 
                                                                                                args.eval_final_episodes, 
                                                                                                args.max_episode_steps)
                episode_returns_list.append(episode_returns)
                discounted_episode_returns_list.append(discounted_episode_returns)
                episode_lengths_list.append(episode_lengths)
                num_policies += 1
        episode_returns = np.array(episode_returns_list)
        discounted_episode_returns = np.array(discounted_episode_returns_list)
        episode_lengths = np.array(episode_lengths_list)
    # --- for jax envs, batch actor params and vmap evaluation over policies --- 
    else: 
        true_env, true_env_params = make_env(args) 
        directory = f"torel/runs/{args.algo}/{args.task}/{args.seed}/{args.n_value}"
        actor_params_list = []
        actor_net = None
        num_policies = 0
        for filename in os.listdir(directory):
            if filename.endswith(".pkl"):
                filepath = os.path.join(directory, filename)
                actor_wrapper = load_pkl(filepath)
                actor_params_list.append(actor_wrapper.actor_params)
                if actor_net is None:
                    actor_net = actor_wrapper.actor_net
                num_policies += 1
        batched_actor_params = jax.tree_util.tree_map(lambda *args: jnp.stack(args), *actor_params_list)
        rng = jax.random.PRNGKey(args.seed)
        eval_rng = jax.random.split(rng, num_policies)
        episode_returns, discounted_episode_returns, episode_lengths = jax.vmap(eval_policy_batched, in_axes=(0, None, None, None, 0, None, None, None))(eval_rng, true_env, true_env_params, actor_net, batched_actor_params, args.discount_factor, args.eval_final_episodes, args.max_episode_steps)

    # --- create results dictionary ---
    infinite_discounted_returns = infinite_horizon_discounted_return(args.max_episode_steps, args.discount_factor, discounted_episode_returns)
    norm_true_regrets = get_regret(infinite_discounted_returns, args.discount_factor, args.min_reward, args.max_reward)
    assert norm_true_regrets.shape == (num_policies, args.eval_final_episodes)
    online_tuning = {"norm_true_regrets": norm_true_regrets, 
                    "episode_returns": episode_returns, 
                    "discounted_episode_returns": discounted_episode_returns, 
                    "episode_lengths": episode_lengths, 
                    "min_reward": args.min_reward, 
                    "max_reward": args.max_reward, 
                    "discount_factor": args.discount_factor, 
                    "eval_final_episodes": args.eval_final_episodes, 
                    "max_episode_steps": args.max_episode_steps}

    # --- bootstrap trials - use unifloral evaluation procedure to select hyperparameters based on regret ---
    scores_array = 100 - (100 * norm_true_regrets) # rescale regret to be optimised (rather than minimised), and lie between 0 and 100 (prevent unifloral exploration term from dominating)
    bandit_trial_dict = bootstrap_bandit_trials(scores_array) 
    online_tuning["samples"] = bandit_trial_dict["pulls"] * args.max_episode_steps
    online_tuning["estimated_bests_mean"] = (100 - bandit_trial_dict["estimated_bests_mean"]) / 100
    online_tuning["estimated_bests_ci_low"] = (100 - bandit_trial_dict["estimated_bests_ci_low"]) / 100
    online_tuning["estimated_bests_ci_high"] = (100 - bandit_trial_dict["estimated_bests_ci_high"]) / 100

    np.savez(directory + "/unifloral_online_tuning.npz", **online_tuning)
    print('Saved results to: ', f"{directory}/unifloral_online_tuning.npz")


if __name__ == "__main__":
    args = tyro.cli(args)
    unifloral_online_tuning(args)