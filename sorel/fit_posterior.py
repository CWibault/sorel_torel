"""
Script to fit world model and approximate inference to offline dataset. 

Optimising the hyperparameters in this script to minimise the Posterior Information Loss (PIL) (and ensure that variance_term ~ MSE term) is equivalent to optimising \phi_I and \phi_II in the SOReL paper. 
"""

import os
import pickle
import functools
import json
from dataclasses import dataclass

from flax.core import frozen_dict
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro
import wandb

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=True"

from pil.ensemble_dynamics import DynamicsEnv
from pil.reset_fns import get_reset_fn
from pil.termination_fns import get_termination_fn
from pil.train_ensemble_dynamics import create_train_state, train_dynamics_model

from unifloral.algos.dynamics import Transition, EnsembleDynamicsModel

from utils.dataset_utils import remove_done_states, load_npz_as_dict
from utils.logging import save_args, save_pkl


@dataclass
class args:
    # --- logging ---
    log: bool = True
    wandb_project: str = "sorel_torel_test"
    wandb_team: str = "team"
    wandb_group: str = "group"

    # --- run identification ---
    algo: str = "posterior"
    seed: int = 0

    # --- environment and offline dataset ---
    task: str = "brax-halfcheetah-full-replay"
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
    weight_decay: float = 2.5e-05
    validation_split: float = 0.1

    # --- evaluation ---
    eval_interval: int = 2500


if __name__ == "__main__":
    
    # --- parse general arguments ---
    args = tyro.cli(args)

    # --- parse task specific arguments ---
    save_args(args, f"sorel/runs/{args.task}/{args.seed}/{args.n_value}", f"{args.algo}_args.json")
    rng = jax.random.PRNGKey(args.seed)

    # --- initialise logger ---
    if args.log:
        wandb.init(config=args,
                    project=args.wandb_project,
                    entity=args.wandb_team,
                    group=args.wandb_group,
                    job_type=f"train_dynamics_{args.task}_{args.seed}_{args.n_value}")

    # --- initialise dataset ---
    assert "d4rl" not in args.task, "D4RL datasets are unsuitable for SOReL: no prior on the model - dataset should be diverse"
    dataset = load_npz_as_dict(f'datasets/{args.task}.npz')
    dataset = remove_done_states(dataset)
    dataset["next_action"] = np.roll(dataset["action"], -1, axis=0)
    dataset = Transition(obs=np.array(dataset["obs"]),
                        action=np.array(dataset["action"]),
                        reward=np.array(dataset["reward"]),
                        next_obs=np.array(dataset["next_obs"]),
                        next_action=np.array(dataset["next_action"]),
                        done=np.array(dataset["done"]))
    dataset_indices = jax.random.choice(rng, dataset.obs.shape[0], shape=(args.n_value,), replace=False)
    os.makedirs(f"sorel/runs/{args.task}/{args.seed}/{args.n_value}", exist_ok=True)
    np.save(f"sorel/runs/{args.task}/{args.seed}/{args.n_value}/dataset_indices", dataset_indices)
    sampled_dataset = Transition(obs=jnp.array(dataset.obs[dataset_indices]),
                                action=jnp.array(dataset.action[dataset_indices]),
                                reward=jnp.array(dataset.reward[dataset_indices]),
                                next_obs=jnp.array(dataset.next_obs[dataset_indices]),
                                next_action=jnp.array(dataset.next_action[dataset_indices]),
                                done=jnp.array(dataset.done[dataset_indices]))

    # --- initialise model ---
    dummy_delta_obs_action = jnp.zeros(sampled_dataset.obs.shape[1] + sampled_dataset.action.shape[1])
    dynamics_net = EnsembleDynamicsModel(obs_dim=sampled_dataset.obs.shape[1],
                                        action_dim=sampled_dataset.action.shape[1],
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
    os.makedirs(f"sorel/runs/{args.task}/{args.seed}/{args.n_value}", exist_ok=True)
    with open(os.path.join(f"sorel/runs/{args.task}/{args.seed}/{args.n_value}/posterior_final_info.json"), "w") as f:
        json.dump(info, f, indent=4)

    # --- extract only elite parameters ---
    elite_dynamics_net=EnsembleDynamicsModel(obs_dim=sampled_dataset.obs.shape[1],
                                             action_dim=sampled_dataset.action.shape[1],
                                             num_ensemble=args.num_elites,
                                             n_layers=args.n_layers,
                                             layer_size=args.layer_size)
    params = frozen_dict.unfreeze(train_state.params)
    ensemble_params = params["params"]["ensemble"]
    ensemble_params = jax.tree_map(lambda p: p[elite_idxs], ensemble_params)
    params["params"]["ensemble"] = ensemble_params
    elite_params = frozen_dict.freeze(params)

    # --- create environment ---
    if args.action_type == "continuous":
        action_dim = sampled_dataset.action.shape[1]
    else:
        print('Action space is discrete. Using number of actions: ', args.num_actions)
        action_dim = args.num_actions
    termination_fn = get_termination_fn(args.task)
    reset_fn = get_reset_fn(args.task)
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

    # --- save dynamics environment ---
    save_pkl(dynamics_env, f"sorel/runs/{args.task}/{args.seed}/{args.n_value}", "dynamics_env.pkl")

    if args.log:
        wandb.finish()