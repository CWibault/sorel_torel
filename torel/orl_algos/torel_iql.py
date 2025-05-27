"""
IQL with final evaluation to determine TOReL regret metric for that hyperparameter combination.

Requires a saved dynamics environment (run fit_posterior.py) (for TOReL). 
""" 

from collections import namedtuple
from dataclasses import dataclass
from functools import partial
import os

import distrax
import flax.linen as nn
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

from pil.ensemble_dynamics import DynamicsEnv # required for loading dynamics environment

from torel.torel_eval_callback import make_torel_eval_callback

from unifloral.algos.dynamics import Transition, EnsembleDynamics, EnsembleDynamicsModel # required for loading dynamics model
from unifloral.algos.iql import AgentTrainState, Transition, SoftQNetwork, DualQNetwork, StateValueFunction, TanhGaussianActor, create_train_state, make_train_step
from unifloral.actor_wrapper import ActorWrapper

from utils.dataset_utils import load_npz_as_dict, remove_done_states
from utils.evaluate.eval_policy import eval_policy
from utils.evaluate.eval_policy_d4rl import eval_policy_d4rl
from utils.logging import wandb_log_info, save_pkl, load_pkl
from utils.envs.env_wrappers import LogWrapper, ClipAction
from utils.regret_utils import infinite_horizon_discounted_return, get_regret

try: 
    from utils.envs.make_env import make_env
except:
    pass

try: 
    import d4rl
    import gym 
except:
    pass

os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=True"


@dataclass
class Args:
    # --- logging ---
    log: bool = False
    wandb_project: str = "sorel-torel-test"
    wandb_team: str = "team"
    wandb_group: str = "group"

    # --- run identification ---
    seed: int = 0
    algo: str = "iql"
    num_updates: int = 1_000_000

    # --- environment and offline dataset ---
    task: str = "brax-halfcheetah-full-replay"
    n_value: int = 200000 
    discount_factor: float = 0.998
    min_reward: float = 0
    max_reward: float = 3.5

    # --- iql ---
    lr: float = 3e-4
    batch_size: int = 256
    gamma: float = 0.99
    polyak_step_size: float = 0.005
    beta: float = 3.0
    iql_tau: float = 0.7
    exp_adv_clip: float = 100.0

    # --- evaluation ---
    num_eval_workers: int = 20


if __name__ == "__main__":

    # --- parse arguments ---
    args = tyro.cli(Args)

    # --- initialise logger ---
    if args.log:
        wandb.init(config=args,
                    project=args.wandb_project,
                    entity=args.wandb_team,
                    group=args.wandb_group,
                    job_type="train_agent")

    # --- initialise dataset ---
    rng = jax.random.PRNGKey(args.seed)
    if "d4rl" in args.task:
        sampled_dataset = d4rl.qlearning_dataset(gym.make(args.task[5:]))
        sampled_dataset = Transition(
            obs=jnp.array(sampled_dataset["observations"]),
            action=jnp.array(sampled_dataset["actions"]),
            reward=jnp.array(sampled_dataset["rewards"]),
            next_obs=jnp.array(sampled_dataset["next_observations"]),
            done=jnp.array(sampled_dataset["terminals"]))
    else: 
        dataset = load_npz_as_dict(f'datasets/{args.task}.npz')
        if "hopper" not in args.task and "walker2d" not in args.task: 
            print("Removing done states from dataset")
            dataset = remove_done_states(dataset)
        dataset["next_action"] = np.roll(dataset["action"], -1, axis=0)
        dataset = Transition(obs=np.array(dataset["obs"]),
                                action=np.array(dataset["action"]),
                                reward=np.array(dataset["reward"]),
                                next_obs=np.array(dataset["next_obs"]),
                                done=np.array(dataset["done"]))
        dataset_indices = np.load(f"torel/runs/torel/{args.task}/{args.seed}/{args.n_value}/dataset_indices.npy")
        sampled_dataset = Transition(obs=jnp.array(dataset.obs[dataset_indices]),
                                    action=jnp.array(dataset.action[dataset_indices]),
                                    reward=jnp.array(dataset.reward[dataset_indices]),
                                    next_obs=jnp.array(dataset.next_obs[dataset_indices]),
                                    done=jnp.array(dataset.done[dataset_indices]))

    # --- initialise agent and value networks ---
    num_actions = sampled_dataset.action.shape[1]
    obs_mean = sampled_dataset.obs.mean(axis=0)
    obs_std = jnp.nan_to_num(sampled_dataset.obs.std(axis=0), nan=1.0)
    dummy_obs = jnp.zeros(sampled_dataset.obs.shape[1])
    dummy_action = jnp.zeros(num_actions)
    actor_net = TanhGaussianActor(num_actions, obs_mean, obs_std)
    q_net = DualQNetwork(obs_mean, obs_std)
    value_net = StateValueFunction(obs_mean, obs_std)

    # --- initialise target networks ---
    rng, rng_actor, rng_q, rng_value = jax.random.split(rng, 4)
    agent_state = AgentTrainState(
        actor=create_train_state(args, rng_actor, actor_net, [dummy_obs]),
        dual_q=create_train_state(args, rng_q, q_net, [dummy_obs, dummy_action]),
        dual_q_target=create_train_state(args, rng_q, q_net, [dummy_obs, dummy_action]),
        value=create_train_state(args, rng_value, value_net, [dummy_obs]))

    # --- make and execute train step ---
    _agent_train_step_fn = make_train_step(args, actor_net.apply, q_net.apply, value_net.apply, sampled_dataset)
    (rng, agent_state), loss = jax.lax.scan(_agent_train_step_fn, (rng, agent_state), None, args.num_updates)

    # --- wrap and evaluate policy --- 
    actor_wrapper = ActorWrapper(actor_net, agent_state.actor.params)
    if "d4rl" in args.task:
        model_env = load_pkl(f"torel/runs/torel/{args.task}/{args.seed}/dynamics_env.pkl")
        save_pkl(actor_wrapper, f"torel/runs/{args.algo}/{args.task}/{args.seed}", f"beta_{args.beta}_iql_tau_{args.iql_tau}_actor.pkl")
        true_env = gym.vector.make(args.task[5:], num_envs=args.num_eval_workers)
        true_env_params = None
        true_eval_policy = eval_policy_d4rl
    else:
        model_env = load_pkl(f"torel/runs/torel/{args.task}/{args.seed}/{args.n_value}/dynamics_env.pkl")
        save_pkl(actor_wrapper, f"torel/runs/{args.algo}/{args.task}/{args.seed}/{args.n_value}", f"beta_{args.beta}_iql_tau_{args.iql_tau}_actor.pkl")
        true_env, true_env_params = make_env(args)
        true_eval_policy = eval_policy
    if model_env.action_type == "continuous":
        print('Action space is continuous. Clipping actions.')
        model_env = ClipAction(model_env)
    model_env = LogWrapper(model_env)
    model_eval_policy = eval_policy
    eval_callback = make_torel_eval_callback(args,
                                             model_env, 
                                             model_env.default_params,
                                             model_eval_policy,
                                             true_env,
                                             true_env_params,
                                             true_eval_policy,
                                             model_env.num_elites, 
                                             args.num_eval_workers, 
                                             args.discount_factor, 
                                             args.min_reward, 
                                             args.max_reward)
    rng = jax.random.PRNGKey(args.seed) # reinitialise for evaluation
    log_dict = eval_callback(agent_state, actor_wrapper, rng)

    if args.log:
        wandb.finish()