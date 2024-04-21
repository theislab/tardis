import os
import sys
import gc
import warnings
import anndata as ad
import scanpy as sc
import torch
from pathlib import Path

sys.path.append(str(Path(os.getcwd()).resolve().parents[0]))

import tardis

local_run = True
if local_run:
    tardis.config = tardis.config_local
else:
    tardis.config = tardis.config_server
print(f"CUDA used: {torch.cuda.is_available()}")


adata_file_path = os.path.join(
    tardis.config.io_directories["processed"], "dataset_subset_sex_2.h5ad"
)
assert os.path.isfile(adata_file_path), f"File not already exist: `{adata_file_path}`"
adata = ad.read_h5ad(adata_file_path)
sc.pp.filter_cells(adata, min_genes=10, inplace=True)


warmup_epoch_range = [10, 30]
# _, n_epochs_kl_warmup = warmup_epoch_range
n_epochs_kl_warmup = 400

counteractive_minibatch_settings = dict(
    method="categorical_random",
    method_kwargs=dict(
        within_labels=False,
        within_batch=False,
        within_categorical_covs=None,
        seed="forward",
    ),
)

disentanglement_targets_configurations = [
    dict(
        obs_key="age",
        n_reserved_latent=8,
        counteractive_minibatch_settings=counteractive_minibatch_settings,
        losses=[
            dict(
                progress_bar=True,
                weight=100,
                method="mse",
                target_type="pseudo_categorical",
                latent_group="reserved",
                transformation="inverse",
                is_minimized=True,
                warmup_period=warmup_epoch_range,
                method_kwargs={},
                coefficient=1.0,
            ),
            dict(
                progress_bar=False,
                weight=10,
                method="mse",
                target_type="categorical",
                latent_group="reserved",
                transformation="none",
                is_minimized=False,
                warmup_period=warmup_epoch_range,
                method_kwargs={},
                coefficient="none",
            ),
        ],
    ),
]

model_params = dict(
    n_hidden=512,
    n_layers=3,
    n_latent=30,
    gene_likelihood="nb",
    use_batch_norm="none",
    use_layer_norm="both",
    dropout_rate=0.1,
    include_auxillary_loss=True,
)

train_params = dict(
    max_epochs=1000,
    train_size=0.8,
    batch_size=512,
    check_val_every_n_epoch=10,
    learning_rate_monitor=True,
    # early stopping:
    early_stopping=True,
    early_stopping_patience=150,
    early_stopping_monitor="elbo_train",
    plan_kwargs=dict(
        n_epochs_kl_warmup=n_epochs_kl_warmup,
        lr=1e-3,
        weight_decay=1e-6,
        # optimizer="AdamW"
        # lr-scheduler:
        reduce_lr_on_plateau=True,
        lr_patience=100,
        lr_scheduler_metric="elbo_train",
    ),
)

dataset_params = dict(
    layer=None,
    labels_key="cell_type",
    batch_key="concatenated_integration_covariates",
    categorical_covariate_keys=None,
    disentanglement_targets_configurations=disentanglement_targets_configurations,
)

tardis.MyModel.setup_anndata(adata, **dataset_params)

# tardis.MyModel.setup_wandb(
#     wandb_configurations=tardis.config.wandb,
#     hyperparams=dict(
#         model_params=model_params,
#         train_params=train_params,
#         dataset_params=dataset_params,
#     )
# )

vae = tardis.MyModel(adata, **model_params)
vae.train(**train_params)
