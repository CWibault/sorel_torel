"""
Generic evaluation callback for purejaxrl policy training scripts. 
"""

import jax

from utils.evaluate.eval_policy import eval_policy
from utils.logging import wandb_log_info

def make_eval_callback(args, env, env_params):
    def eval_callback(train_state, actor, rng):
        episode_return, _, episode_length = eval_policy(rng,
                                                        env,
                                                        env_params,
                                                        actor,
                                                        args.discount_factor,
                                                        args.num_eval_workers,
                                                        env_params.max_steps_in_episode)
        log_dict = {"step_count": train_state.step_count,
                    "episode_return": episode_return.mean(),
                    "episode_length": episode_length.mean()}
        jax.debug.print("Step: {}", train_state.step_count)
        jax.debug.print("Mean Episode return: {}", episode_return.mean())
        jax.debug.print("Mean Episode Length: {}", episode_length.mean())
        if args.log == True:
            wandb_log_info(log_dict, args.task + "_" + args.algo)
        return log_dict
    return eval_callback

