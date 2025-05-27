import argparse
from dataclasses import dataclass
import os
from typing import NamedTuple

from flax.training.train_state import TrainState as BaseTrainState
import jax
import jax.numpy as jnp
import numpy as np
import optax

os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=True"

from utils.envs.make_env import make_env
from utils.logging import wandb_log_info
from utils.purejaxrl.actor_wrappers import RNNActorCriticWrapper
from utils.purejaxrl.collect_datasets import CollectDataset
from utils.purejaxrl.networks import ActorCriticRNNDiscrete, ActorCriticRNNContinuous, ScannedRNN


@dataclass
class args:
    # --- logging ---
    debug: bool = True
    log: bool = True
    collect_dataset: bool = False
    wandb_project: str = "sorel-torel-test"
    wandb_team: str = "team"
    wandb_group: str = "group"

    # --- environment and offline dataset ---
    task: str = "walker2d"
    discount_factor: float = 0.998
    min_reward: float = 0
    max_reward: float = 3.5

    # --- ppo-rnn hyperparameters ---
    algo: str = "offline_ppo_rnn"
    seed: int = 0
    n_value: int = 500000
    rnn_size: int = 256
    layer_size: int = 256
    activation: str = "tanh"
    num_envs: int = 512
    num_steps: int = 64
    total_timesteps: int = 50000000 
    update_epochs: int = 8
    num_minibatches: int = 16
    gamma: float = 0.998
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.001
    vf_coef: float = 0.5
    lr: float = 0.0001
    anneal_lr: bool = True
    max_grad_norm: float = 0.5
    burn_in_pct: float = 0.25

    # --- evaluation ---
    eval_frequency: int = 100
    num_eval_workers: int = 10


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray 
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


class TrainState(BaseTrainState):
    step_count: int


def make_train_ppo_rnn(args, env, env_params, eval_callback = None):
    
    args.num_updates = args.total_timesteps // args.num_steps // args.num_envs
    args.minibatch_size = args.num_envs * args.num_steps // args.num_minibatches
    args.num_burn_in_steps = int(args.burn_in_pct * args.num_steps)

    assert args.num_updates // args.eval_frequency >= 1, "num_updates must be greater than eval_frequency"

    # --- initialise collect_dataset and log_dict objects, but these are only filled if args.collect_dataset and args.log are True ---
    collect_dataset = CollectDataset()
    log_dict = {'episode_return': [], 'episode_length': [], 'timestep': []}

    def linear_schedule(count):
        frac = 1.0 - (count // (args.num_minibatches * args.update_epochs)) / args.num_updates
        return args.lr * frac

    # --- make actor_critic_net ---
    try:
        actor_critic_net = ActorCriticRNNDiscrete(env.action_space(env_params).n, args.rnn_size, args.layer_size, args.activation)
        print("Discrete Recurrent Actor Critic")
    except:
        actor_critic_net = ActorCriticRNNContinuous(env.action_space(env_params).shape[0], args.rnn_size, args.layer_size, args.activation)
        print("Continuous Recurrent Actor Critic")

    @jax.jit
    def train_ppo_rnn(rng):

        # --- initialise actor_critic : require time dimension ---
        rng, _rng = jax.random.split(rng)
        init_obs = jnp.zeros((1, args.num_envs, *env.observation_space(env_params).shape)) # [time, batch, features]
        init_done = jnp.zeros((1, args.num_envs), dtype=bool) # [time, batch]
        print(init_obs.shape, init_done.shape)
        init_x = (init_obs, init_done)
        init_hstate = ScannedRNN.initialize_carry(args.num_envs, args.rnn_size) # [batch, rnn_size]
        actor_critic_params = actor_critic_net.init(_rng, init_hstate, init_x)
        if args.anneal_lr:
            tx = optax.chain(optax.clip_by_global_norm(args.max_grad_norm), optax.adam(learning_rate=linear_schedule, eps=1e-5))
        else:
            tx = optax.chain(optax.clip_by_global_norm(args.max_grad_norm), optax.adam(args.lr, eps=1e-5))
        train_state = TrainState.create(apply_fn=actor_critic_net.apply, params=actor_critic_params, tx=tx, step_count=0)

        # --- initialise environment: no time dimension ---
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, args.num_envs)
        obs, env_state = jax.vmap(env.reset, in_axes=(0))(reset_rng) # obs shape is [batch, features]
        done = jnp.zeros((args.num_envs), dtype=bool) # done shape is [batch]

        # --- initialise train runner state ---
        rng, _rng = jax.random.split(rng)
        train_runner_state = (train_state, env_state, obs, done, init_hstate, _rng)

        # --- train_loop ---
        def _train_eval_step(train_runner_state, unused):

            # --- update network ---
            def _update_step(runner_state, unused):

                # --- collect trajectories ---
                def _env_step(runner_state, unused):
                    train_state, env_state, last_obs, last_done, hstate, rng = runner_state
                    rng, _rng = jax.random.split(rng)

                    # --- select action ---
                    ac_in = (last_obs[np.newaxis, :], last_done[np.newaxis, :]) # [time, batch, features]
                    hstate, pi, value = actor_critic_net.apply(train_state.params, hstate, ac_in)
                    action = pi.sample(seed=_rng)
                    log_prob = pi.log_prob(action)
                    value, action, log_prob = value.squeeze(0), action.squeeze(0), log_prob.squeeze(0)

                    # --- step environment ---
                    rng, _rng = jax.random.split(rng)
                    rng_step = jax.random.split(_rng, args.num_envs)
                    obs, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(rng_step, env_state, action, env.default_params)
                    transition = Transition(last_done, action, value, reward, log_prob, last_obs, info)
                    if args.collect_dataset:
                        log_dict = {"obs": last_obs, "action": action, "next_obs": obs, "reward": reward, "done": done}
                        jax.debug.callback(collect_dataset, log_dict)
                    train_state = train_state.replace(step_count=train_state.step_count + args.num_envs)
                    runner_state = (train_state, env_state, obs, done, hstate, rng)
                    return runner_state, transition

                initial_hstate = runner_state[-2]
                runner_state, traj_batch = jax.lax.scan(_env_step, runner_state, None, args.num_steps)

                # --- calculate advantage ---
                train_state, env_state, last_obs, last_done, hstate, rng = runner_state
                ac_in = (last_obs[np.newaxis, :], last_done[np.newaxis, :])
                _, _, last_val = actor_critic_net.apply(train_state.params, hstate, ac_in)
                last_val = last_val.squeeze(0)
                
                def _calculate_gae(traj_batch, last_val, last_done):
                    def _get_advantages(carry, transition):
                        gae, next_value, next_done = carry
                        done, value, reward = transition.done, transition.value, transition.reward 
                        delta = reward + args.gamma * next_value * (1 - next_done) - value
                        gae = delta + args.gamma * args.gae_lambda * (1 - next_done) * gae
                        return (gae, value, done), gae
                    _, advantages = jax.lax.scan(_get_advantages, (jnp.zeros_like(last_val), last_val, last_done), traj_batch, reverse=True, unroll=16)
                    return advantages, advantages + traj_batch.value
                advantages, targets = _calculate_gae(traj_batch, last_val, last_done)

                # --- update actor_critic_ ---
                def _update_epoch(update_state, unused):
                    def _update_minibatch(train_state, batch_info):
                        init_hstate, traj_batch, advantages, targets = batch_info

                        def _loss_fn(params, init_hstate, traj_batch, gae, targets):
                            # --- rerun network ---
                            _, pi, value = actor_critic_net.apply(params, init_hstate[0], (traj_batch.obs, traj_batch.done))
                            log_prob = pi.log_prob(traj_batch.action)

                            # --- calculate value loss, applying burn-in if necessary ---
                            value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(-args.clip_eps, args.clip_eps)
                            value_losses = jnp.square(value - targets)
                            value_losses_clipped = jnp.square(value_pred_clipped - targets)
                            value_loss = 0.5 * jax.lax.dynamic_slice(
                                jnp.maximum(value_losses, value_losses_clipped), 
                                (args.num_burn_in_steps, 0), 
                                (args.num_steps - args.num_burn_in_steps, value_losses.shape[1])).mean()

                            # --- calculate actor loss and entropy, applying burn-in if necessary ---
                            ratio = jnp.exp(log_prob - traj_batch.log_prob)
                            gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                            loss_actor1 = ratio * gae
                            loss_actor2 = jnp.clip(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps) * gae
                            loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                            loss_actor = jax.lax.dynamic_slice(
                                loss_actor, 
                                (args.num_burn_in_steps, 0), 
                                (args.num_steps - args.num_burn_in_steps, loss_actor.shape[1])).mean()
                            entropy = jax.lax.dynamic_slice(
                                pi.entropy(), 
                                (args.num_burn_in_steps, 0), 
                                (args.num_steps - args.num_burn_in_steps, pi.entropy().shape[1])).mean()

                            total_loss = loss_actor + args.vf_coef * value_loss - args.ent_coef * entropy
                            return total_loss, (value_loss, loss_actor, entropy)

                        total_loss, grads = jax.value_and_grad(_loss_fn, has_aux=True)(train_state.params, init_hstate, traj_batch, advantages, targets)
                        train_state = train_state.apply_gradients(grads=grads)
                        return train_state, total_loss

                    train_state, init_hstate, traj_batch, advantages, targets, rng = update_state

                    # --- shuffle batches across environments, handling time dimension such that full trajectories remain intact--- 
                    rng, _rng = jax.random.split(rng)
                    permutation = jax.random.permutation(_rng, args.num_envs)
                    batch = (init_hstate, traj_batch, advantages, targets)
                    shuffled_batch = jax.tree_util.tree_map(lambda x: jnp.take(x, permutation, axis=1), batch)

                    # --- split shuffled batch into minibatches ---
                    minibatches = jax.tree_util.tree_map(lambda x: jnp.swapaxes(jnp.reshape(x, [x.shape[0], args.num_minibatches, -1] + list(x.shape[2:])), 1, 0), shuffled_batch)

                    # --- update network with minibatches ---
                    train_state, total_loss = jax.lax.scan(_update_minibatch, train_state, minibatches)
                    update_state = (train_state, init_hstate, traj_batch, advantages, targets, rng)
                    return update_state, total_loss

                init_hstate = initial_hstate[None, :]  # augment initial hstate with "time" dimension to use jax.tree_util.tree_map when shuffling batches (shuffling across first dimension i.e. environments)
                update_state = (train_state, init_hstate, traj_batch, advantages, targets, rng)
                update_state, loss_info = jax.lax.scan(_update_epoch, update_state, None, args.update_epochs)
                train_state = update_state[0]
                metric = traj_batch.info
                rng = update_state[-1]
                runner_state = (train_state, env_state, last_obs, last_done, hstate, rng)

                if args.log:
                    def log_callback(metric):
                        dones = metric["returned_episode"]
                        returns = metric["returned_episode_returns"] * dones
                        lengths = metric["returned_episode_lengths"] * dones
                        timesteps = metric["timestep"] * args.num_envs * dones 
                        num_completed = dones.sum()
                        mean_return = jnp.nan_to_num(jnp.sum(returns)/num_completed)
                        mean_length = jnp.nan_to_num(jnp.sum(lengths)/num_completed)
                        time_step = jnp.nan_to_num(jnp.sum(timesteps)/num_completed)
                        if num_completed > 0:
                            log_dict["episode_return"].append(mean_return)
                            log_dict["episode_length"].append(mean_length)
                            log_dict["timestep"].append(time_step)
                            print("Episode Return: ", mean_return, "Episode Length: ", mean_length, "Timestep: ", time_step)
                            wandb_log_info({"episode_return": mean_return, "episode_length": mean_length, "timestep": time_step}, args.task + "_" + args.algo)
                        return  
                    jax.debug.callback(log_callback, metric)

                return runner_state, None
            
            # --- train policy ---
            train_runner_state, _ = jax.lax.scan(_update_step, train_runner_state, None, args.eval_frequency)
            rng = train_runner_state[-1]
            
            # --- evaluate policy ---
            if eval_callback is not None:
                recurrent_actor = RNNActorCriticWrapper(actor_critic_net, train_runner_state[0].params, args.rnn_size)
                eval_callback_dict = eval_callback(train_runner_state[0], recurrent_actor, rng)
            else: 
                eval_callback_dict = {}
            return train_runner_state, eval_callback_dict

        # --- train eval policy ---
        train_runner_state, eval_callback_dict = jax.lax.scan(_train_eval_step, train_runner_state, None, args.num_updates//args.eval_frequency)
        train_state = train_runner_state[0]
        return train_state, eval_callback_dict

    return train_ppo_rnn, actor_critic_net, collect_dataset, log_dict


if __name__ == "__main__":
    args = args()
    train_ppo_rnn, actor_critic_net, collect_dataset, log_dict = make_train_ppo_rnn(args, make_env(args.task), args.task)


