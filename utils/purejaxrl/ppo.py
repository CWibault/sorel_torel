from dataclasses import dataclass
from typing import NamedTuple

from flax.training.train_state import TrainState as BaseTrainState
from gymnax.environments import spaces
import jax
import jax.numpy as jnp
import optax

from utils.logging import wandb_log_info
from utils.envs.make_env import make_env
from utils.purejaxrl.actor_wrappers import ActorCriticWrapper
from utils.purejaxrl.collect_datasets import CollectDataset
from utils.purejaxrl.networks import ActorCriticDiscrete, ActorCriticContinuous


@dataclass
class args:
    # --- logging ---
    log: bool = True
    collect_dataset: bool = True
    wandb_project: str = "sorel-torel-test"
    wandb_team: str = "team"
    wandb_group: str = "group"

    # --- task and offline dataset ---
    task: str = "halfcheetah"
    num_envs_to_save: int = 8
    discount_factor: float = 0.998

    # --- ppo hyperparameters ---
    algo: str = "ppo"
    layer_size: int = 256
    seed: int = 0
    lr: float = 3.0e-4
    num_envs: int = 8
    num_steps: int = 256
    total_timesteps: int = 1000000
    update_epochs: int = 8
    num_minibatches: int = 32
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    activation: str = "tanh"
    anneal_lr: bool = True

    # --- evaluation ---
    eval: bool = True
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


def make_train_ppo(args, env, env_params, eval_callback = None):

    args.num_updates = (args.total_timesteps // args.num_steps // args.num_envs)
    args.minibatch_size = args.num_envs * args.num_steps // args.num_minibatches

    if eval_callback is not None:
        assert args.num_updates // args.eval_frequency >= 1, "num_updates must be greater than eval_frequency"
    else: 
        args.eval_frequency = args.num_updates

    # --- initialise collect_dataset and log_dict objects, but these are only filled if args.collect_dataset and args.log are True ---
    collect_dataset = CollectDataset()
    log_dict = {'episode_return': [], 'episode_length': [], 'timestep': []}

    def linear_schedule(count):
        frac = 1.0 - (count // (args.num_minibatches * args.update_epochs)) / args.num_updates
        return args.lr * frac
    
    # --- make actor_critic_net ---
    try:
        actor_critic_net = ActorCriticDiscrete(env.action_space(env_params).n, args.layer_size, args.activation)
        print("Discrete Actor Critic")
    except:
        actor_critic_net = ActorCriticContinuous(env.action_space(env_params).shape[0], args.layer_size, args.activation)
        print("Continuous Actor Critic")

    @jax.jit
    def train_ppo(rng):

        # --- initialise actor critic ---
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros(env.observation_space(env_params).shape)
        actor_critic_params = actor_critic_net.init(_rng, init_x)
        if args.anneal_lr:
            tx = optax.chain(optax.clip_by_global_norm(args.max_grad_norm), optax.adam(learning_rate=linear_schedule, eps=1e-5))
        else:
            tx = optax.chain(optax.clip_by_global_norm(args.max_grad_norm), optax.adam(args.lr, eps=1e-5))
        train_state = TrainState.create(apply_fn=actor_critic_net.apply, params=actor_critic_params, tx=tx, step_count=0)

        # --- initalise environment ---
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, args.num_envs)
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)

        # --- initialise train runner state ---
        rng, _rng = jax.random.split(rng)
        train_runner_state = (train_state, env_state, obsv, _rng)

        # --- train_loop ---
        def _train_eval_step(train_runner_state, unused):

            # --- update network ---
            def _update_step(runner_state, unused):

                # --- collect trajectories ---
                def _env_step(runner_state, unused):
                    train_state, env_state, last_obs, rng = runner_state

                    # --- select action ---
                    rng, _rng = jax.random.split(rng)
                    pi, value = actor_critic_net.apply(train_state.params, last_obs)
                    unclipped_action = pi.sample(seed=_rng)
                    log_prob = pi.log_prob(unclipped_action)

                    # --- clip action if continuous ---
                    if isinstance(env.action_space(env_params), spaces.Discrete):
                        action = unclipped_action
                    else:
                        low = env.action_space(env_params).low
                        high = env.action_space(env_params).high
                        action = jnp.clip(unclipped_action, low, high)

                    # --- step environment ---
                    rng, _rng = jax.random.split(rng)
                    rng_step = jax.random.split(_rng, args.num_envs)
                    obs, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(rng_step, env_state, action, env_params)
                    transition = Transition(done, unclipped_action, value, reward, log_prob, last_obs, info)
                    if args.collect_dataset:
                        log_dict = {"obs": last_obs, "action": action, "next_obs": obs, "reward": reward, "done": done}
                        jax.debug.callback(collect_dataset, log_dict)
                    train_state = train_state.replace(step_count=train_state.step_count + args.num_envs)
                    runner_state = (train_state, env_state, obs, rng)
                    return runner_state, transition

                runner_state, traj_batch = jax.lax.scan(_env_step, runner_state, None, args.num_steps)

                # --- calculate advantage ---
                train_state, env_state, last_obs, rng = runner_state
                _, last_val = actor_critic_net.apply(train_state.params, last_obs)

                def _calculate_gae(traj_batch, last_val):
                    def _get_advantages(gae_and_next_value, transition):
                        gae, next_value = gae_and_next_value
                        done, value, reward = transition.done, transition.value, transition.reward
                        delta = reward + args.gamma * next_value * (1 - done) - value
                        gae = delta + args.gamma * args.gae_lambda * (1 - done) * gae
                        return (gae, value), gae

                    _, advantages = jax.lax.scan(_get_advantages, (jnp.zeros_like(last_val), last_val), traj_batch, reverse=True, unroll=16)
                    return advantages, advantages + traj_batch.value
                advantages, targets = _calculate_gae(traj_batch, last_val)

                # --- update network ---
                def _update_epoch(update_state, unused):
                    def _update_minibatch(train_state, batch_info):
                        traj_batch, advantages, targets = batch_info

                        def _loss_fn(params, traj_batch, gae, targets):
                            # --- rerun network ---
                            pi, value = actor_critic_net.apply(params, traj_batch.obs)
                            log_prob = pi.log_prob(traj_batch.action)

                            # --- calculate value loss, applying burn-in if necessary ---
                            value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(-args.clip_eps, args.clip_eps)
                            value_losses = jnp.square(value - targets)
                            value_losses_clipped = jnp.square(value_pred_clipped - targets)
                            value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

                            # --- calculate actor loss and entropy ---
                            ratio = jnp.exp(log_prob - traj_batch.log_prob)
                            gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                            loss_actor1 = ratio * gae
                            loss_actor2 = (jnp.clip(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps) * gae)
                            loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                            loss_actor = loss_actor.mean()
                            entropy = pi.entropy().mean()

                            total_loss = (loss_actor + args.vf_coef * value_loss - args.ent_coef * entropy)
                            return total_loss, (value_loss, loss_actor, entropy)

                        total_loss, grads = jax.value_and_grad(_loss_fn, has_aux=True)(train_state.params, traj_batch, advantages, targets)
                        train_state = train_state.apply_gradients(grads=grads)
                        return train_state, total_loss

                    train_state, traj_batch, advantages, targets, rng = update_state

                    # --- permute across trajectory steps and environments ---
                    rng, _rng = jax.random.split(rng)
                    batch_size = args.minibatch_size * args.num_minibatches
                    assert (batch_size == args.num_steps * args.num_envs), "batch size must be equal to number of steps * number of envs"
                    permutation = jax.random.permutation(_rng, batch_size)
                    batch = (traj_batch, advantages, targets)
                    batch = jax.tree_util.tree_map(lambda x: x.reshape((batch_size,) + x.shape[2:]), batch)
                    shuffled_batch = jax.tree_util.tree_map(lambda x: jnp.take(x, permutation, axis=0), batch)
                    
                    # --- split shuffled batch into minibatches ---
                    minibatches = jax.tree_util.tree_map(lambda x: jnp.reshape(x, [args.num_minibatches, -1] + list(x.shape[1:])), shuffled_batch)

                    # --- update network with minibatches ---
                    train_state, total_loss = jax.lax.scan(_update_minibatch, train_state, minibatches)
                    update_state = (train_state, traj_batch, advantages, targets, rng)
                    return update_state, total_loss

                update_state = (train_state, traj_batch, advantages, targets, rng)
                update_state, total_loss = jax.lax.scan(_update_epoch, update_state, None, args.update_epochs)
                train_state = update_state[0]
                metric = traj_batch.info
                rng = update_state[-1]
                runner_state = (train_state, env_state, last_obs, rng)
                
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
                actor = ActorCriticWrapper(actor_critic_net, train_runner_state[0].params)
                eval_callback_dict = eval_callback(train_runner_state[0], actor, rng)
            else: 
                eval_callback_dict = {}
            return train_runner_state, eval_callback_dict

        # --- train eval policy ---
        train_runner_state, eval_callback_dict = jax.lax.scan(_train_eval_step, train_runner_state, None, args.num_updates//args.eval_frequency)
        train_state = train_runner_state[0]
        return train_state, eval_callback_dict
    
    return train_ppo, actor_critic_net, collect_dataset, log_dict

if __name__ == "__main__":
    args = args()
    train_ppo, actor_critic_net, collect_dataset, log_dict = make_train_ppo(args, make_env(args.task), args.task)


