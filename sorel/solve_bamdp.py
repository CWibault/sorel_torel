"""
Script to solve the BAMDP using RNN_PPO. Requires a saved ensemble dynamics model (run fit_posterior.py). 

Optimising the hyperparameters to minimise the approximate regret is equivalent to optimising \phi_III in the SOReL paper. 

sorel_eval_callback function evaluates the true regret to validate the approximate regret. In practice, only the final policy (learned using the hyperparameters that minimise the approximate regret) should be deployed. 
"""

from dataclasses import dataclass
import json
import os
import tyro

import jax
import wandb

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=True"

from pil.ensemble_dynamics import DynamicsEnv # required for loading dynamics model

from sorel.sorel_eval_callback import make_sorel_eval_callback

from unifloral.algos.dynamics import Transition, EnsembleDynamicsModel # required for loading dynamics model

from utils.envs.env_wrappers import LogWrapper, ClipAction
from utils.envs.make_env import make_env
from utils.evaluate.eval_recurrent_policy import eval_recurrent_policy
from utils.logging import save_args, wandb_log_info, save_pkl, load_pkl
from utils.purejaxrl.actor_wrappers import RNNActorCriticWrapper
from utils.purejaxrl.ppo_rnn import make_train_ppo_rnn
from utils.regret_utils import infinite_horizon_discounted_return, get_regret


@dataclass
class args:
    # --- logging ---
    debug: bool = True
    log: bool = True
    collect_dataset: bool = False
    save_actor: bool = True
    wandb_project: str = "sorel-torel-test"
    wandb_team: str = "team"
    wandb_group: str = "group"

    # --- run identification ---
    seed: int = 0
    algo: str = "bamdp_solver"

    # --- environment and offline dataset ---
    task: str = "brax-halfcheetah-full-replay"
    n_value: int = 200000
    discount_factor: float = 0.998
    min_reward: float = -0.5
    max_reward: float = 3.5

    # --- ppo-rnn ---
    rnn_size: int = 256
    layer_size: int = 256
    activation: str = "tanh"
    num_envs: int = 512
    num_steps: int = 64
    total_timesteps: int = 50000000 
    update_epochs: int = 8
    num_minibatches: int = 16
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.003
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    lr: float = 0.0003
    anneal_lr: bool = True
    burn_in_pct: float = 0.25

    # --- evaluation ---
    eval_frequency: int = 100
    num_eval_workers: int = 10


if __name__ == "__main__":

    # --- parse general arguments ---
    args = tyro.cli(args)

    # --- initialise logger ---
    if args.log:
        wandb.init(config=args,
                    project=args.wandb_project,
                    entity=args.wandb_team,
                    group=args.wandb_group,
                    job_type=f"train_actor_{args.task}_{args.seed}_{args.n_value}")

    # --- load dynamics model ---
    env = load_pkl(f"sorel/runs/{args.task}/{args.seed}/{args.n_value}/dynamics_env.pkl")
    if env.action_type == "continuous":
        env = ClipAction(env)
    env = LogWrapper(env)

    # --- load true environment (for tracking) ---
    true_env, true_env_params = make_env(args)

    # --- make eval callback function to track the expected regret with training --- 
    eval_callback = make_sorel_eval_callback(args, 
                                            env, 
                                            env.default_params, 
                                            true_env, 
                                            true_env_params, 
                                            args.rnn_size, 
                                            env.num_elites, 
                                            args.num_eval_workers, 
                                            args.discount_factor, 
                                            args.min_reward, 
                                            args.max_reward)

    # --- train actor ---
    train_offline_ppo_rnn, actor_critic_net, _, _ = make_train_ppo_rnn(args, env, env.default_params, eval_callback)
    rng = jax.random.PRNGKey(args.seed) 
    train_state, train_log_dict = train_offline_ppo_rnn(rng)

    # --- save final parameters ---
    if args.save_actor:
        actor_wrapper = RNNActorCriticWrapper(actor_critic_net, train_state.params, args.rnn_size)
        save_pkl(actor_wrapper, f"sorel/runs/{args.task}/{args.seed}/{args.n_value}", "actor.pkl")

        # --- save final log dict ---
        final_log_dict = jax.tree_map(lambda x: x[-1], train_log_dict)
        agg_fn = lambda x, k: {f"final_{k}": float(x)}
        info = agg_fn(final_log_dict["offline_returns"], "offline_returns") | agg_fn(final_log_dict["true_returns"], "true_returns") | agg_fn(final_log_dict["true_episode_lengths"], "true_episode_lengths") | agg_fn(final_log_dict["norm_expected_regret"], "norm_expected_regret") | agg_fn(final_log_dict["norm_true_regret"], "norm_true_regret") | agg_fn(final_log_dict["norm_expected_regret_based_on_variance"], "norm_expected_regret_based_on_variance") | agg_fn(final_log_dict["norm_expected_regret_based_on_median"], "norm_expected_regret_based_on_median") | agg_fn(final_log_dict["norm_expected_regret_based_on_max"], "norm_expected_regret_based_on_max") | agg_fn(final_log_dict["norm_expected_regret_based_on_min"], "norm_expected_regret_based_on_min") | agg_fn(final_log_dict["norm_expected_regret_based_on_mean"], "norm_expected_regret_based_on_mean")
        os.makedirs(f"sorel/runs/{args.task}/{args.seed}/{args.n_value}", exist_ok=True)
        with open(os.path.join(f"sorel/runs/{args.task}/{args.seed}/{args.n_value}/dynamics_final_info.json"), "w") as f:
            json.dump(info, f, indent=4)
    
    if args.log:
        wandb.finish()
    