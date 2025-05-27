"""
Script to fit world model and approximate inference to offline dataset. 

Differs to SOReL in that it also precomputes termination statistics for MOReL, and handles D4RL datasets. 
"""

from collections import namedtuple
from dataclasses import dataclass
import os
import pickle
from typing import Optional

from flax.core import frozen_dict
import flax.linen as nn
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import json
import optax
import tyro
import wandb

# --- additional imports ---
import numpy as np
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pil.ensemble_dynamics import DynamicsEnv
from pil.reset_fns import get_reset_fn as dynamics_env_get_reset_fn
from pil.termination_fns import get_termination_fn as dynamics_env_get_termination_fn
from pil.train_ensemble_dynamics import create_train_state, train_dynamics_model

from unifloral.termination_fns import get_termination_fn as unifloral_get_termination_fn
from unifloral.algos.dynamics import Transition, EnsembleDynamics, EnsembleDynamicsModel, compute_model_discrepancy

from utils.dataset_utils import remove_done_states, load_npz_as_dict
from utils.logging import save_args, wandb_log_info, save_pkl

try: 
    import d4rl
    import gym 
except:
    pass

os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=True"


@dataclass
class Args:
    # --- logging ---
    log: bool = True
    wandb_project: str = "sorel-torel-test"
    wandb_team: str = "team"
    wandb_group: str = "group"

    # --- run identification ---
    algo: str = "posterior"
    seed: int = 0

    # --- task and offline dataset ---
    task: str = "brax-halfcheetah-full-replay" # [brax-halfcheetah-full-replay, brax-hopper-full-replay, brax-walker2d-full-replay, d4rl-halfcheetah-medium-expert-v2, d4rl-hopper-medium-v2, d4rl-walker2d-medium-replay-v2]
    n_value: int = 200000
    action_type: str = "continuous"
    num_actions: int = -1 # only required for discrete action spaces
    min_action: float = -1
    max_action: float = 1
    max_episode_steps: int = 1000
    discount_factor: float = 0.998

    # --- world model and approximate inference hyperparameters --- 
    # model 
    n_layers: int = 3
    layer_size: int = 200
    num_ensemble: int = 7
    num_elites: int = 5
    # optimiser 
    num_epochs: int = 400
    lr: float = 0.001
    lr_schedule: str = "cosine"
    final_lr_pct: float = 0.1
    batch_size: int = 256
    logvar_diff_coef: float = 0.01
    weight_decay: float = 2.5e-5
    validation_split: float = 0.1
    precompute_term_stats: bool = True # required for MOReL

    # --- evaluation ---
    eval_interval: int = 10_000


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
        dataset = d4rl.qlearning_dataset(gym.make(args.task[5:]))
        sampled_dataset = Transition(
            obs=jnp.array(dataset["observations"]),
            action=jnp.array(dataset["actions"]),
            reward=jnp.array(dataset["rewards"]),
            next_obs=jnp.array(dataset["next_observations"]),
            next_action=jnp.roll(dataset["actions"], -1, axis=0),
            done=jnp.array(dataset["terminals"]))
        args.n_value = sampled_dataset.obs.shape[0]
        model_directory = f"torel/runs/torel/{args.task}/{args.seed}"
    else:
        dataset = load_npz_as_dict(f'datasets/{args.task}.npz')
        dataset = remove_done_states(dataset) # always remove, since next_obs is the reset state
        dataset["next_action"] = np.roll(dataset["action"], -1, axis=0)
        dataset = Transition(obs=np.array(dataset["obs"]),
                            action=np.array(dataset["action"]),
                            reward=np.array(dataset["reward"]),
                            next_obs=np.array(dataset["next_obs"]),
                            next_action=np.array(dataset["next_action"]),
                            done=np.array(dataset["done"]))
        dataset_indices = jax.random.choice(rng, dataset.obs.shape[0], shape=(args.n_value,), replace=False)
        model_directory = f"torel/runs/torel/{args.task}/{args.seed}/{args.n_value}" # save indices for training model later
        os.makedirs(model_directory, exist_ok=True)
        np.save(os.path.join(model_directory, "dataset_indices.npy"), dataset_indices)
        sampled_dataset = Transition(obs=jnp.array(dataset.obs[dataset_indices]),
                                    action=jnp.array(dataset.action[dataset_indices]),
                                    reward=jnp.array(dataset.reward[dataset_indices]),
                                    next_obs=jnp.array(dataset.next_obs[dataset_indices]),
                                    next_action=jnp.array(dataset.next_action[dataset_indices]),
                                    done=jnp.array(dataset.done[dataset_indices]))

    # --- initialise dynamics model ---
    num_actions = sampled_dataset.action.shape[1]
    dummy_delta_obs_action = jnp.zeros(sampled_dataset.obs.shape[1] + num_actions)
    dynamics_net = EnsembleDynamicsModel(obs_dim=sampled_dataset.obs.shape[1],
                                         action_dim=num_actions,
                                         num_ensemble=args.num_ensemble,
                                         n_layers=args.n_layers,
                                         layer_size=args.layer_size)
    rng, dynamics_rng = jax.random.split(rng)
    train_state = create_train_state(args, dynamics_rng, dynamics_net, [dummy_delta_obs_action])

    # --- train dynamics model ---
    print("Training dynamics model...")
    rng, trn_rng = jax.random.split(rng)
    train_state, elite_idxs, val_info = train_dynamics_model(train_state, args, sampled_dataset, trn_rng)

    # --- write final posterior information loss to file ---
    agg_fn = lambda x, k: {f"final_{k}": float(x)}
    info = agg_fn(val_info["regret_ub"], "regret_ub") | agg_fn(val_info["posterior_information_loss"], "posterior_information_loss") | agg_fn(val_info["expectation_term"], "expectation_term") | agg_fn(val_info["variance_of_means"], "variance_of_means") | agg_fn(val_info["means_of_variances"], "means_of_variances") | agg_fn(val_info["variance_term"], "variance_term")
    os.makedirs(model_directory, exist_ok=True)
    with open(os.path.join(model_directory, "dynamics_final_info.json"), "w") as f:
        json.dump(info, f, indent=4)

    # --- extract elite model parameters and create elite dynamics network ---
    params = frozen_dict.unfreeze(train_state.params)
    ensemble_params = params["params"]["ensemble"]
    ensemble_params = jax.tree_map(lambda p: p[elite_idxs], ensemble_params)
    params["params"]["ensemble"] = ensemble_params
    elite_params = frozen_dict.freeze(params)
    elite_dynamics_net=EnsembleDynamicsModel(obs_dim=sampled_dataset.obs.shape[1],
                                            action_dim=sampled_dataset.action.shape[1],
                                            num_ensemble=args.num_elites,
                                            n_layers=args.n_layers,
                                            layer_size=args.layer_size)

    # --- precompute statistics for termination penalties (required for MOReL) ---
    discrepancy, min_r = None, None
    if args.precompute_term_stats:
        print("Precomputing dataset statistics for termination penalties...")
        rng, rng_discrepancy = jax.random.split(rng)
        discrepancy = compute_model_discrepancy(elite_dynamics_net, elite_params, sampled_dataset, rng_discrepancy)
        min_r = float(jnp.min(sampled_dataset.reward))
        print(f"Model discrepancy: {discrepancy}, Minimum dataset reward: {min_r}")

    # --- save dynamics model ---
    termination_fn = unifloral_get_termination_fn(args.task)
    dynamics_model = EnsembleDynamics(elite_dynamics_net, elite_params, args.num_elites, termination_fn, discrepancy, min_r)
    print("Saving dynamics model...")
    save_pkl(dynamics_model, model_directory, "dynamics_model.pkl")

    # --- create and save dynamics environment ---
    if args.action_type == "continuous":
        action_dim = sampled_dataset.action.shape[1]
    else:
        print('Action space is discrete. Using number of actions: ', args.num_actions)
        action_dim = args.num_actions
    termination_fn = dynamics_env_get_termination_fn(args.task)
    reset_fn = dynamics_env_get_reset_fn(args.task)
    dynamics_env=DynamicsEnv(elite_dynamics_net, 
                            elite_params, 
                            args.num_elites,
                            reset_fn,
                            termination_fn,
                            action_dim,
                            sampled_dataset.obs.shape[1],
                            np.percentile(sampled_dataset.reward, 0.0),
                            np.percentile(sampled_dataset.reward, 97.5),
                            args.action_type,
                            args.min_action,
                            args.max_action,
                            args.max_episode_steps)
    save_pkl(dynamics_env, model_directory, "dynamics_env.pkl")

    # --- save args ---
    save_args(args, model_directory, f"{args.algo}_args.json")
    print("Saved posterior model")

    if args.log:
        wandb.finish()