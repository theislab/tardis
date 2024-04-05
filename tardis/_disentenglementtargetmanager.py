#!/usr/bin/env python3

from scipy.sparse import spmatrix
import numpy as np

from ._disentenglementtargetconfigurations import DisentenglementTargetConfigurations
from ._mytrainingplan import TrainingStepLogger
from ._myconstants import REGISTRY_KEY_DISENTENGLEMENT_TARGETS


class DisentenglementTargetManager:

    configurations: "DisentenglementTargetConfigurations"
    anndata_manager_state_registry: dict

    @classmethod
    def set_configurations(cls, value):
        cls.configurations = value

    @classmethod
    def set_anndata_manager_state_registry(cls, value):
        cls.anndata_manager_state_registry = value


class CounteractiveMinibatchGenerator:

    _cached_group_definition_possible_indices_dict: dict = dict()

    @classmethod
    def main(cls, method: str, **kwargs):
        """
        Generates a counteractive minibatch based on the specified method.

        Parameters:
        - method: Name of the method to use for generating the minibatch.
        - **kwargs: Keyword arguments passed to the method, including method-specific kwargs and common parameters.
        """
        if not hasattr(CounteractiveMinibatchGenerator, method) or method == "main" or not callable(getattr(cls, method)):
            raise AttributeError(f"{cls.__name__} does not have a callable attribute '{method}'.")
        else:
            static_function = getattr(cls, method)
            return static_function(**kwargs)

    @classmethod
    def _cached_group_definition_possible_indices_dict_creator(
        cls, 
        obs_group_definitions,
        cell_dataset_tensors
        
    ):
        pass
        # np.unique(np.vstack([a,a,c]).T, axis=0)
            
        
        

    @classmethod
    def random(
        cls,
        dataset_tensors: dict[np.ndarray | spmatrix],
        target_obs_key: str,
        splitter_index: np.ndarray,
        minibatch_relative_index: list[int],
        # method_kwargs: Do not use preset values to prevent program misbehaviors.
        seed: int | str,
        exclude_itself: bool,
        exclude_group: bool,
        group_size_aware: bool,
        within_label: bool,
        within_batch: bool
    ) -> list[int]:
        
        # print("####")
        group_definitions = dataset_tensors[REGISTRY_KEY_DISENTENGLEMENT_TARGETS][target_obs_key].values
        # print("group_definitions:", group_definitions)
        
        # a=np.vstack([dataset_tensors["batch"].flatten(), dataset_tensors["labels"].flatten(), group_definitions])
        # print(a.shape)
        
        
        minibatch_global_index = splitter_index[minibatch_relative_index]
        minibatch_group_definitions = group_definitions[minibatch_global_index]
        
        
        
        
        if target_obs_key in cls._cached_group_definition_possible_indices_dict:
            possible_indices = cls._cached_group_definition_possible_indices_dict[target_obs_key]
        else:
            possible_indices = np.ones_like(group_definitions).astype(bool)
            
            if exclude_itself:
                possible_indices[group_definitions] = 0

        
        DisentenglementTargetManager.anndata_manager_state_registry
        
            
        
        # print("group_definitions:", dataset_tensors[REGISTRY_KEY_DISENTENGLEMENT_TARGETS][target_obs_key].values)
        
        
        # print(len(splitter_index), type(splitter_index), max(splitter_index))
        # print(len(minibatch_relative_index), type(minibatch_relative_index), max(minibatch_relative_index))
        
        # use DisentenglementTargetManager.anndata_manager_state_registry method to get indexes
        # of each target in `REGISTRY_KEY_DISENTENGLEMENT_TARGETS` jointobsfield.
        return {i: "__tensor__" for i in dataset_tensors}
