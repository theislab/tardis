#!/usr/bin/env python3

import os
import sys
import warnings
from contextlib import contextmanager


@contextmanager
def suppress_stdout():
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    try:
        yield
    finally:
        sys.stdout.close()
        sys.stdout = original_stdout
        

@contextmanager
def ignore_predetermined_warnings():
    with warnings.catch_warnings():
        # scvi
        warnings.filterwarnings("ignore", ".*jax.tree_util.register_keypaths is deprecated.*")
        warnings.filterwarnings("ignore", ".*Since v1.0.0, scvi-tools no longer uses a random seed by default.*")
        warnings.filterwarnings("ignore", ".*Setting `dl_pin_memory_gpu_training` is deprecated in v1.0 and.*")

        # sparse
        warnings.filterwarnings("ignore", ".*SparseDataset is deprecated and will be removed in late 2024.*")

        # umap
        warnings.filterwarnings("ignore", ".*The 'nopython' keyword argument was not supplied to the 'numba.jit'.*")

        # scanpy
        warnings.filterwarnings(
            "ignore", ".*No data for colormapping provided via 'c'. Parameters 'cmap' will be ignored.*"
        )

        # vae.train
        warnings.filterwarnings("ignore", ".*The `srun` command is available on your system but is not used.*")
        warnings.filterwarnings("ignore", ".*The value argument must be within the support of the distribution.*")
        warnings.filterwarnings("ignore", ".*MPS available but not used. Set `accelerator` and `devices` using.*")
        warnings.filterwarnings(
            "ignore", r".*lr scheduler dict contains the key\(s\) \['monitor'\], but the keys will be.*"
        )
        warnings.filterwarnings("ignore", ".*`use_gpu` is deprecated in v1.0 and will be removed in v1.1.*")

        # plotting
        warnings.filterwarnings(
            "ignore",
            ".*Generic family 'sans-serif' not found because none of the following families were found: Times.*",
        )

        # wandb
        warnings.filterwarnings(
            "ignore",
            ".*`resume` will be ignored since W&B syncing is set to `offline`..*",
        )

        # raise error
        warnings.filterwarnings(
            "error", ".*set to `mps`. Please note that not all PyTorch operations are supported with.*"
        )
        yield


@contextmanager
def suppress_stdout():
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, "w")
    try:
        yield
    finally:
        sys.stdout.close()
        sys.stdout = original_stdout


@contextmanager
def dumb_contextmanager():
    yield


@contextmanager
def scanpy_temporary_verbosity(level=0):
    import scanpy as sc

    # Save the original verbosity level
    original_verbosity = sc.settings.verbosity
    try:
        # Set the verbosity to the desired level
        sc.settings.verbosity = level
        yield
    finally:
        # Revert the verbosity to the original level
        sc.settings.verbosity = original_verbosity
