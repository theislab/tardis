#!/usr/bin/env python3

import numpy as np
import torch
from scipy.sparse import spmatrix

from ._cachedpossiblegroupdefinitionindices import (
    CachedPossibleGroupDefinitionIndices,
    DatapointDefinitionsKeyGenerator,
)
from ._disentenglementtargetmanager import DisentenglementTargetManager
from ._myconstants import NEGATIVE_EXAMPLE_KEY, POSITIVE_EXAMPLE_KEY
from ._trainingsteplogger import TrainingStepLogger


class CounteractiveMinibatchGenerator:

    _cached_group_definition_possible_indices_dict: dict = dict()

    @classmethod
    def main(cls, target_obs_key_ind: int, **kwargs):
        """
        Generates a counteractive minibatch based on the specified method.

        Parameters:
        - method: Name of the method to use for generating the minibatch.
        - **kwargs: Keyword arguments passed to the method, including method-specific kwargs and common parameters.
        """
        method = DisentenglementTargetManager.configurations.get_by_index(
            target_obs_key_ind
        ).counteractive_minibatch_settings.method
        if (
            not hasattr(CounteractiveMinibatchGenerator, method)
            or method == "main"
            or not callable(getattr(cls, method))
        ):
            raise AttributeError(f"{cls.__name__} does not have a callable attribute '{method}'.")
        else:
            class_function = getattr(cls, method)
            return class_function(target_obs_key_ind=target_obs_key_ind, **kwargs)

    @staticmethod
    def configuration_random_seed(config_seed):
        if isinstance(config_seed, int):
            return config_seed
        else:
            return getattr(TrainingStepLogger, config_seed)

    @classmethod
    def random(
        cls,
        target_obs_key_ind: int,
        minibatch_tensors: dict[str, torch.Tensor],
        dataset_tensors: dict[str, np.ndarray | spmatrix],
        splitter_index: np.ndarray,
        data_split_identifier: str,
        minibatch_relative_index: list[int],
    ) -> list[int]:

        config = DisentenglementTargetManager.configurations.items[target_obs_key_ind].counteractive_minibatch_settings
        possible_indices = CachedPossibleGroupDefinitionIndices.get(
            dataset_tensors, target_obs_key_ind, data_split_identifier, splitter_index, config
        )
        minibatch_definitions = (
            DatapointDefinitionsKeyGenerator.create_definitions(minibatch_tensors, target_obs_key_ind, config)
            .clone()  # prevent misbehavior at below randomization operation. testing needed for method training speed.
            .numpy()  # as it returns a tensor originally.
        )

        selection = {POSITIVE_EXAMPLE_KEY: minibatch_definitions.copy(), NEGATIVE_EXAMPLE_KEY: None}

        rng = np.random.default_rng(  # Seeded RNG for consistency
            seed=CounteractiveMinibatchGenerator.configuration_random_seed(config.method_kwargs["seed"])
        )
        n_cat = DisentenglementTargetManager.anndata_manager_state_registry["disentenglement_target"]["n_cats_per_key"][
            target_obs_key_ind
        ]
        indice_group_definitions = 0
        random_ints = rng.integers(0, n_cat, size=minibatch_definitions.shape[0])
        minibatch_definitions[:, indice_group_definitions] = np.where(
            random_ints == minibatch_definitions[:, indice_group_definitions],
            (random_ints + 1) % n_cat,  # if random integer is the same as the original
            random_ints,  # use random integers
        )
        selection[NEGATIVE_EXAMPLE_KEY] = minibatch_definitions

        selected_elements = dict()
        for selection_key, selection_minibatch_definitions in selection.items():
            _selected_elements = []
            for datapoint in selection_minibatch_definitions:
                try:
                    _selected_elements.append(rng.choice(possible_indices[tuple(datapoint)]))
                except KeyError as e:
                    to_report = {i: len(possible_indices[i]) for i in possible_indices}
                    raise KeyError(
                        f"The minibatch definition `{tuple(datapoint)}` is not found in possible cached indice "
                        f"dictionary.The keys of this dict is `{to_report}` (given as key and number of elements "
                        "within). It happens when the `within` statements are so strict, giving rise to there is no "
                        "corresponding element with such a configuration in the original dataset.In general, "
                        "please note that `within_batch` is the most frequent problem as it is actually "
                        "correlated with many possible metadatas directly."
                    ) from e
            selected_elements[selection_key] = _selected_elements

        return selected_elements
