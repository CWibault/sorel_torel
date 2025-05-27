import jax
import numpy as np


def load_npz_as_dict(file_path):
    """Function to load npz file as dictionary."""
    npz = np.load(file_path)
    return {key: npz[key] for key in npz.files}


def remove_done_states(dataset): 
    """Function to remove the done states from the dataset, since next_obs is the reset state."""
    if dataset["action"].ndim == 1:
        dataset["action"] = np.expand_dims(dataset["action"], axis=1)
    if dataset["reward"].ndim == 1:
        dataset["reward"] = np.squeeze(dataset["reward"])
    for key in dataset.keys():
        dataset[key] = np.delete(dataset[key], np.where(dataset["done"] == 1), axis=0)
    return dataset
    

def create_dataset_iter(rng, inputs, targets, batch_size):
    """Function to create a batched dataset iterator."""
    perm = jax.random.permutation(rng, inputs.shape[0])
    shuffled_inputs, shuffled_targets = inputs[perm], targets[perm]
    num_batches = inputs.shape[0] // batch_size
    iter_size = num_batches * batch_size
    dataset_iter = jax.tree_map(
        lambda x: x[:iter_size].reshape(num_batches, batch_size, *x.shape[1:]),
        (shuffled_inputs, shuffled_targets))
    return dataset_iter
