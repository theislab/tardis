#!/usr/bin/env python3

from ._disentenglementtargetmanager import DisentenglementTargetManager
from ._cachedpossiblegroupdefinitionindices import CachedPossibleGroupDefinitionIndices, DatapointDefinitionsKeyGenerator
from ._trainingsteplogger import TrainingStepLogger

import numpy as np
import copy
import torch
from scipy.sparse import spmatrix


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
        method = DisentenglementTargetManager.configurations.get_by_index(target_obs_key_ind).counteractive_minibatch_settings.method
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
        minibatch_definitions = DatapointDefinitionsKeyGenerator.create_definitions(minibatch_tensors, target_obs_key_ind, config)
        
        # Seeded RNG for consistency
        rng = np.random.default_rng(seed=CounteractiveMinibatchGenerator.configuration_random_seed(config.method_kwargs["seed"])) 
        selected_elements = []
        for datapoint in minibatch_definitions:
            tuple_key = tuple(datapoint.tolist())
            selected_element = rng.choice(possible_indices[tuple_key])
            selected_elements.append(selected_element)
        
        return selected_elements