"""
Functions to train ensemble dynamics model while tracking the validation Posterior Information Loss (PIL) [= (normalised) predictive variance loss + (normalised) MSE loss]. 
Also tracks the regret upper bound for SOReL. 

Code based on unifloral's implementation, with added cosine learning rate schedule on the Adam optimiser. 
"""
import functools

from flax.core import frozen_dict
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro

from utils.dataset_utils import create_dataset_iter
from utils.logging import wandb_log_info


def calculate_regret_upper_bound(discount_factor, posterior_information_loss):
    """Calculate the regret upper bound for SOReL."""
    regret_ub = jnp.minimum(1.0, 2 * jnp.sqrt(1 - jnp.exp(-(posterior_information_loss)/(1-discount_factor))))
    return regret_ub


def create_train_state(args, rng, network, dummy_input):
    """Create a train state for the dynamics model."""
    steps_per_epoch = max(1, args.n_value // args.batch_size)
    total_steps = args.num_epochs * steps_per_epoch
    if args.lr_schedule == "constant":
        lr_schedule = args.lr
    elif args.lr_schedule == "cosine":
        lr_schedule = optax.cosine_decay_schedule(
            init_value=args.lr,
            decay_steps=total_steps,
            alpha=args.final_lr_pct)
    elif args.lr_schedule == "exponential":
        lr_schedule = optax.exponential_decay(
            init_value=args.lr,
            transition_steps=total_steps // 10,
            decay_rate=0.96)
    elif args.lr_schedule == "warmup_cosine":
        warmup_steps = int(0.1 * total_steps)
        lr_schedule = optax.join_schedules(
            schedules=[optax.linear_schedule(
                    init_value=0.0,
                    end_value=args.lr,
                    transition_steps=warmup_steps),
                optax.cosine_decay_schedule(
                    init_value=args.lr,
                    decay_steps=total_steps - warmup_steps,
                    alpha=args.final_lr_pct)],
            boundaries=[warmup_steps])
    else: 
        lr_schedule = args.lr
        print(f"Warning: Unrecognised schedule '{args.lr_schedule}', using constant learning rate.")
    optimizer = optax.chain(optax.clip_by_global_norm(1.0),
                            optax.adamw(learning_rate=lr_schedule,
                            eps=1e-5, 
                            weight_decay=args.weight_decay))
    return TrainState.create(apply_fn=network.apply,
                             params=network.init(rng, dummy_input),
                             tx=optimizer)


def train_dynamics_model(train_state, args, dataset, rng):
    """Train dynamics model."""
    pure_log_info = functools.partial(wandb_log_info, algo=args.algo)

    # --- prepare dataset for training ---
    inputs = jnp.concatenate((dataset.obs, dataset.action), axis=-1)
    delta_obs = dataset.next_obs - dataset.obs
    targets = jnp.concatenate((delta_obs, dataset.reward.reshape(-1, 1)), axis=-1)
    targets_std = jnp.std(targets, axis=0)
    targets_std = jnp.where(targets_std == 0, 1, targets_std) # avoid division by zero
    print('targets_std: ', targets_std)

    rng, split_rng = jax.random.split(rng)
    train_size = int((1 - args.validation_split) * inputs.shape[0])
    shuffled_idxs = jax.random.permutation(split_rng, inputs.shape[0])
    train_idxs, val_idxs = shuffled_idxs[:train_size], shuffled_idxs[train_size:]
    train_inputs, train_targets = inputs[train_idxs], targets[train_idxs]
    val_inputs, val_targets = inputs[val_idxs], targets[val_idxs]
    print(f"Dataset size: {inputs.shape[0]}, train partition: {train_size}")

    # --- make train functions ---
    def _train_step(train_state, batch):
        """Train dynamics model for one step."""
        inputs, targets = batch

        def _loss_fn(params):
            mean, logvar = train_state.apply_fn(params, inputs)
            mse_loss = ((mean - targets) ** 2) * jnp.exp(-logvar)
            mse_loss = mse_loss.sum(0).mean()
            var_loss = logvar.sum(0).mean()
            max_logvar = params["params"]["max_logvar"]
            min_logvar = params["params"]["min_logvar"]
            logvar_diff = (max_logvar - min_logvar).sum()
            loss = mse_loss + var_loss + args.logvar_diff_coef * logvar_diff
            info = {}
            return loss, info

        grads, info = jax.grad(_loss_fn, has_aux=True)(train_state.params)
        train_state = train_state.apply_gradients(grads=grads)
        if args.log:
            do_log = train_state.step % args.eval_interval == 1
            jax.lax.cond(do_log, pure_log_info, lambda x: None, info)
        return train_state, None

    def _eval_step(train_state, batch):
        """Evaluate the dynamics model on the holdout set."""
        inputs, targets = batch

        def _loss_fn(params):
            """Calculate evaluation loss as unweighted prediction MSE."""
            ensemble_mean, ensemble_logvar = train_state.apply_fn(params, inputs)
            loss = jnp.mean(((ensemble_mean - targets) ** 2), axis=(1, 2))

            # --- calculate posterior information loss  ---
            ensemble_var = jnp.exp(ensemble_logvar)
            norm_mse = (((ensemble_mean - targets) ** 2) / (2 * targets_std ** 2)).sum(-1) # sum over dimensions
            expectation_term = norm_mse.mean()
            variance_of_means = (ensemble_mean.var(0, ddof=1)/(2 * targets_std ** 2)).sum(-1)
            variance_of_means_term = variance_of_means.mean()
            means_of_variances = (ensemble_var.mean(0) / (2 * targets_std ** 2)).sum(-1)
            means_of_variances_term = means_of_variances.mean()
            variance_term = variance_of_means_term + means_of_variances_term
            posterior_information_loss = expectation_term + variance_term
            regret_ub = calculate_regret_upper_bound(args.discount_factor, posterior_information_loss)
            info = {"step": train_state.step,
                    "variance_term": variance_term,
                    "variance_of_means": variance_of_means.mean(),
                    "means_of_variances": means_of_variances.mean(),
                    "expectation_term": expectation_term,
                    "posterior_information_loss": posterior_information_loss,
                    "regret_ub": regret_ub}
            return loss, info

        loss, info = _loss_fn(train_state.params)
        return train_state, (loss, info)

    def train_epoch(_, carry):
        """Train dynamics model for one epoch."""
        rng, train_state, elite_idxs = carry
        rng, rng_train, rng_val = jax.random.split(rng, 3)
        size = args.batch_size
        train_iter = create_dataset_iter(rng_train, train_inputs, train_targets, size)
        train_state = jax.lax.scan(_train_step, train_state, train_iter)[0]
        val_iter = create_dataset_iter(rng_val, val_inputs, val_targets, size)
        val_loss, val_info = jax.lax.scan(_eval_step, train_state, val_iter)[1]
        elite_idxs = val_loss.mean(axis=0).argsort()[: args.num_elites] # mean over all validation batches
        val_info = jax.tree_map(lambda x: jnp.mean(x, axis=0), val_info)
        if args.log:
            pure_log_info({**val_info, "elite_idxs": elite_idxs})
        return rng, train_state, elite_idxs

    # --- Train model for N epochs ---
    dummy_elite_idxs = jnp.zeros((args.num_elites,), jnp.int32)
    init_carry = (rng, train_state, dummy_elite_idxs)
    _, train_state, elite_idxs = jax.lax.fori_loop(0, args.num_epochs, train_epoch, init_carry)

    # --- final eval to get posterior information loss for final model ---
    rng, rng_val = jax.random.split(rng)
    val_iter = create_dataset_iter(rng_val, val_inputs, val_targets, args.batch_size)
    _, (_, val_info) = jax.lax.scan(_eval_step, train_state, val_iter)
    val_info = jax.tree_map(lambda x: jnp.mean(x, axis=0), val_info)
    return train_state, elite_idxs, val_info
