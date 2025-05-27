from dataclasses import asdict
import json
import os
import pickle

import jax
import wandb


def save_args(args, directory, filename):
    """Save arguments to JSON file."""
    os.makedirs(directory, exist_ok=True)
    with open(directory + "/" + filename, "w") as f:
        json.dump(asdict(args), f, indent=4)


def wandb_log_info(info, algo):
    """Log metrics to wandb."""
    info = {f"{algo}/" + k: v for k, v in info.items()}
    jax.experimental.io_callback(wandb.log, None, info)


def save_pkl(obj, directory, filename):
    """Save object to pickle file."""
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, filename), "wb") as f:
        pickle.dump(obj, f)


def load_pkl(path):
    """Load object from pickle file."""
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj