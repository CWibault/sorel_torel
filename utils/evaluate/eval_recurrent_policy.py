import jax
import jax.numpy as jnp

from utils.purejaxrl.networks import ScannedRNN

def eval_recurrent_policy(rng,
                          env,
                          env_params,
                          actor,
                          rnn_size,
                          discount_factor,
                          num_eval_workers,
                          max_steps):
    """jax environment evaluation function."""

    num_eval_workers = int(num_eval_workers)

    # --- detect if env_params is batched ---
    def is_batched(params, batch_size):
        if isinstance(params, dict):
            if not params:  
                return False
            example_key = next(iter(params))
            example_value = params[example_key]
        else: # assume dataclass or structured array
            try:
                fields = [f for f in dir(params) 
                        if not f.startswith('_') and not callable(getattr(params, f))]
                if not fields:  
                    return False
                example_value = getattr(params, fields[0])
            except:
                return False
        return (hasattr(example_value, 'shape') and 
                len(getattr(example_value, 'shape')) > 0 and 
                example_value.shape[0] == batch_size)
    batched = is_batched(env_params, num_eval_workers)
    
    def _policy_and_env_step(runner_state, _):
        env_state, last_obs, last_done, hstate, rng, cum_reward, discounted_return, episode_length = runner_state

        # --- select action ---
        action_rng = jax.random.split(rng, num_eval_workers)
        action, hstate = jax.vmap(actor)(last_obs, last_done, hstate, action_rng) # action shape: [num_envs, action_dim]

        # --- step env ---
        rng, step_rng = jax.random.split(rng)
        step_rngs = jax.random.split(step_rng, num_eval_workers)
        if batched:
            obs, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, 0))(step_rngs, env_state, action, env_params)
        else:
            obs, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0, None))(step_rngs, env_state, action, env_params)
        
        episode_length = episode_length + 1 * (~last_done)
        done = jnp.logical_or(done, last_done)
        cum_reward += reward * (~last_done)
        discounted_return += reward * (discount_factor ** episode_length) * (~last_done)
        new_state = (env_state, obs, done, hstate, rng, cum_reward, discounted_return, episode_length)
        return new_state, None

    # --- set initial runner state ---
    rng, _rng = jax.random.split(rng)
    reset_rngs = jax.random.split(_rng, num_eval_workers)
    if batched:
        init_obs, init_env_state = jax.vmap(env.reset, in_axes=(0, 0))(reset_rngs, env_params)
    else:
        init_obs, init_env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rngs, env_params) # [num_envs, features]
    init_done = jnp.zeros(num_eval_workers, dtype=bool) # [num_envs]
    init_hstate = ScannedRNN.initialize_carry(num_eval_workers, rnn_size) # [num_envs, rnn_size]
    init_cum_reward = jnp.zeros(num_eval_workers) 
    init_discounted_return = jnp.zeros(num_eval_workers)
    init_episode_length = jnp.zeros(num_eval_workers) 
    runner_state = (init_env_state, init_obs, init_done, init_hstate, rng, init_cum_reward, init_discounted_return, init_episode_length)

    # --- run policy ---
    final_state, _ = jax.lax.scan(_policy_and_env_step, runner_state, None, int(max_steps))
    return final_state[5], final_state[6], final_state[7]