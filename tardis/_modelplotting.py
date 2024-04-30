#!/usr/bin/env python3

import copy
import math
import warnings
from typing import List, Optional, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scvi import settings

from ._utils.warnings import ignore_predetermined_warnings


class ModelPlotting:

    def plot_training_history(
        self,
        ignore_first: int,
        metrics_name: List[str],
        metrics_title: Optional[List[str]] = None,
        extra_triplets: Optional[List[Tuple[str, str, str]]] = None,
        n_col: int = 3,
        unit_size: float = 3.0,
    ):
        def _ignore_first(history_key, n):
            df = self.history[history_key]
            return df[(df.index >= n)]

        if not (
            isinstance(metrics_name, list) and len(metrics_name) > 0 and all([isinstance(i, str) for i in metrics_name])
        ):
            raise ValueError

        if metrics_title is None:
            metrics_title = copy.deepcopy(metrics_name)

        if not (
            isinstance(metrics_name, list)
            and len(metrics_title) == len(metrics_name)
            and all([isinstance(i, str) for i in metrics_title])
        ):
            raise ValueError

        params = [(f"{i}_train", f"{i}_validation", j) for i, j in zip(metrics_name, metrics_title)]
        if extra_triplets is not None:
            if not isinstance(extra_triplets, list) or not all(
                [isinstance(i, tuple) and len(i) == 3 for i in extra_triplets]
            ):
                raise ValueError
            params.extend(extra_triplets)
        params = tuple(params)

        validatation_calculated = "total_loss_validation" in self.history
        if not validatation_calculated:
            warnings.warn(
                message="Validation is not calculated during training.",
                category=UserWarning,
                stacklevel=settings.warnings_stacklevel,
            )

        sanity_check = [j[i] for j in params for i in range(2) if j[i] not in self.history]
        sanity_check = [sc for sc in sanity_check if not (sc.endswith("_validation") and validatation_calculated)]
        if len(sanity_check) != 0:
            ValueError(f"Following metrics are not in model.history `{sanity_check}`.")

        with ignore_predetermined_warnings():

            train_kwargs = {"label": "Train", "color": "darkgray"}
            valid_kwargs = {"label": "Validation", "color": "gray"}

            params_iter = iter(params)
            n_params = len(params)
            n_ceil = math.ceil(n_params / n_col)
            fig, axs = plt.subplots(nrows=n_ceil, ncols=n_col, figsize=(unit_size * 1 * n_col, n_ceil * unit_size))
            for i in range(n_ceil):
                for j in range(n_col):
                    ax = axs[i, j] if n_ceil != 1 else axs[j]
                    try:
                        train_data, valid_data, title_str = next(params_iter)
                    except StopIteration:
                        fig.delaxes(ax)
                        continue
                    ax.plot(_ignore_first(train_data, ignore_first), **train_kwargs)
                    try:
                        if validatation_calculated:
                            ax.plot(_ignore_first(valid_data, ignore_first), **valid_kwargs)
                            ax.legend(fontsize=8)
                    except KeyError:
                        warnings.warn(
                            message=f"Validation is not calculated during training for `{title_str}`.",
                            category=UserWarning,
                            stacklevel=settings.warnings_stacklevel,
                        )
                    sns.despine(ax=ax, top=True, right=True, left=False, bottom=False, offset=None, trim=False)
                    ax.set_title(title_str, fontsize=10)

            plt.tight_layout()
            plt.show()

    @staticmethod
    def generate_gray_tones(num_variable, alpha=1.0):
        # Define the range of lightness values to avoid very dark or very light grays
        lightness_min = 0.2  # Avoid near-black colors
        lightness_max = 0.8  # Avoid near-white colors

        # Calculate the step to evenly distribute the gray tones within the range
        step = (lightness_max - lightness_min) / max(1, num_variable - 1)  # Avoid division by zero

        # Generate the gray tones with alpha values
        gray_tones_with_alpha = [
            mcolors.to_hex(
                (lightness_min + i * step, lightness_min + i * step, lightness_min + i * step, alpha), keep_alpha=True
            )
            for i in range(num_variable)
        ]

        return gray_tones_with_alpha

    def plot_latent_kde(
        self,
        adata_obs: pd.DataFrame,
        latent_representation: np.ndarray,
        target_obs_key: str | None,
        latent_dim_of_interest: int | None,
    ):

        if target_obs_key is not None:
            latent_representations_with_metadata = np.append(
                latent_representation, adata_obs[target_obs_key].values.reshape(-1, 1), axis=1
            )
            columns = [f"Latent {i}" for i in range(self.module.n_latent)] + [target_obs_key]
        else:
            latent_representations_with_metadata = latent_representation
            columns = [f"Latent {i}" for i in range(self.module.n_latent)]

        df = pd.DataFrame(latent_representations_with_metadata, columns=columns)

        if target_obs_key is not None:
            n_element = len(adata_obs[target_obs_key].unique())
            kwargs = {"palette": ModelPlotting.generate_gray_tones(n_element)}
        else:
            kwargs = {"color": "darkgray"}

        individual_plot_size = 3

        if latent_dim_of_interest is not None:

            with ignore_predetermined_warnings():
                plt.figure(figsize=(individual_plot_size * 1.5, individual_plot_size * 1.5))
                sns.kdeplot(df, x=f"Latent {latent_dim_of_interest}", hue=target_obs_key, fill=True, **kwargs)
                sns.despine(top=True, right=True, left=False, bottom=False)
                if target_obs_key is not None:
                    plt.legend(loc="upper right")
                plt.show()

        else:
            # Determine the optimal subplot grid size
            total_plots = self.module.n_latent
            cols = math.ceil(total_plots**0.5)
            rows = math.ceil(total_plots / cols)

            fig, axs = plt.subplots(rows, cols, figsize=(cols * individual_plot_size, rows * individual_plot_size))

            # Iterate over all latent dimensions to create individual plots
            for i in range(self.module.n_latent):
                row, col = divmod(i, cols)
                if rows > 1 and cols > 1:
                    ax = axs[row, col]
                elif rows > 1:
                    ax = axs[row]
                else:
                    ax = axs[col]

                # Only show the legend for the upper right plot
                plot_legend = (row == 0) and (col == cols - 1 if cols > 1 else 0)

                g = sns.kdeplot(
                    data=df, x=f"Latent {i}", hue=target_obs_key, fill=True, ax=ax, legend=plot_legend, **kwargs
                )
                if plot_legend and target_obs_key is not None:
                    sns.move_legend(g, "upper right")
                sns.despine(ax=ax, top=True, right=True, left=False, bottom=False)

            # Remove empty subplots
            for i in range(self.module.n_latent, rows * cols):
                fig.delaxes(axs.flatten()[i])

            plt.tight_layout()
            plt.show()
