
# pip install gzip

import torch
import numpy as np
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
    "symmetric": path.join("best_models", "spherical_best_model.pt"),
    "baseline": path.join("best_models", "standard_best_model.pt")
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
    model.load_state_dict(state_dict)
    model = model.to(DEVICE)

    return model

def sample_batch(size=128):
    """Return a few test examples installed with the package, on the same device as the model."""

    # generate the dataset
    gen()

    with gzip.open("s2_mnist.gz", 'rb') as f:
        dataset = pickle.load(f)

    data = torch.from_numpy(
        dataset["train"]["images"][:, None, :, :].astype(np.float32))[:size]

    mean = data.mean()
    stdv = data.std()

    data -= mean
    data /= stdv

    t_data = torch.tensor(data).to(DEVICE)

    return t_data

