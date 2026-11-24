
# pip install gzip

import torch
import numpy as np
import os
from os import path
import gzip
import pickle
from gendata import main as gen

from model import S2ConvNet_original as symmetric
from standard_model import StandardCNN as baseline

d_models = {
    "symmetric": symmetric,
    "baseline": baseline
}

d_model_paths = {
    "symmetric": path.join(path.dirname(path.abspath(__file__)), "best_models", "spherical_best_model.pt"),
    "baseline": path.join(path.dirname(path.abspath(__file__)), "best_models", "standard_best_model.pt")
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model(kind):   # kind is "symmetric" or "baseline"
    """Load the trained weights installed with the package onto the GPU if one is available, otherwise the CPU, and return the model."""

    # attempt to identify
    try:
        model_class = d_models[kind]
        model_save_path = d_model_paths[kind]
    except KeyError:
        raise ValueError(f"Model kind must be symmetric or baseline, not {kind}")

    # load model
    model = model_class()
    state_dict = torch.load(model_save_path)
    model.load_state_dict(state_dict["model_state_dict"])
    model = model.to(DEVICE)

    return model

def sample_batch(size=8):
    """Return a few test examples installed with the package, on the same device as the model."""

    DATA_PATH = path.join(path.dirname(path.abspath(__file__)), "s2_mnist.gz")

    with gzip.open(DATA_PATH, 'rb') as f:
        dataset = pickle.load(f)

    data = torch.from_numpy(
        dataset["test"]["images"][:, None, :, :].astype(np.float32))[:size]

    mean = data.mean()
    stdv = data.std()

    data -= mean
    data /= stdv

    data = data.to(DEVICE)

    return data

