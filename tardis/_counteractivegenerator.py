#!/usr/bin/env python3

import copy
import warnings
from typing import Dict

import numpy as np
import torch
from scipy.sparse import spmatrix
from scvi import REGISTRY_KEYS, settings

from ._disentanglementmanager import DisentanglementManager
from ._myconstants import (
    NEGATIVE_EXAMPLE_KEY,
    POSITIVE_EXAMPLE_KEY,
    REGISTRY_KEY_DISENTANGLEMENT_TARGETS,
    REGISTRY_KEY_METRICS_COVARIATES_HELPER,
)
from ._mymonitor import TrainingStepLogger


class DatapointDefinitionsKeyGenerator:

    @classmethod
    def create_definitions(cls, dict_items, target_obs_key_ind, config):

        operation_mode_torch = True if torch.is_tensor(dict_items[REGISTRY_KEYS.BATCH_KEY]) else False

        batch_definitions = dict_items[REGISTRY_KEYS.BATCH_KEY]
        if config.method_kwargs["within_batch"]:
            pass
        elif operation_mode_torch:
            batch_definitions = torch.zeros_like(batch_definitions)
        else:
            batch_definitions = np.zeros_like(batch_definitions)

        if config.method_kwargs["within_labels"]:
            labels_definitions = dict_items[REGISTRY_KEYS.LABELS_KEY]
        elif operation_mode_torch:
            labels_definitions = torch.zeros_like(batch_definitions)
        else:
            labels_definitions = np.zeros_like(batch_definitions)

        if operation_mode_torch:
            group_definitions = dict_items[REGISTRY_KEY_DISENTANGLEMENT_TARGETS][:, target_obs_key_ind].view(-1, 1)
        else:
            group_definitions = (
                dict_items[REGISTRY_KEY_DISENTANGLEMENT_TARGETS].iloc[:, target_obs_key_ind].values.reshape(-1, 1)
            )

        catcovs_definitions = []
        if REGISTRY_KEYS.CAT_COVS_KEY in dict_items:
            cmk = config.method_kwargs["within_categorical_covs"]
            for include_cotcov_ind, include_cotcov in enumerate([] if cmk is None else cmk):
                if include_cotcov and operation_mode_torch:
                    catcovs_definitions.append(
                        dict_items[REGISTRY_KEYS.CAT_COVS_KEY][:, include_cotcov_ind].view(-1, 1)
                    )
                elif include_cotcov:
                    catcovs_definitions.append(
                        dict_items[REGISTRY_KEYS.CAT_COVS_KEY].iloc[:, include_cotcov_ind].values.reshape(-1, 1)
                    )
                elif operation_mode_torch:
                    catcovs_definitions.append(torch.zeros_like(batch_definitions))
                else:
                    catcovs_definitions.append(np.zeros_like(batch_definitions))

        definitions = [group_definitions, batch_definitions, labels_definitions] + catcovs_definitions
        if operation_mode_torch:
            return torch.cat(definitions, dim=1)
        else:
            return np.hstack(definitions)


class CachedPossibleGroupDefinitionIndices:

    _items: Dict[str, Dict[str, Dict[tuple, np.ndarray]]]

    @staticmethod
    def _initialize_verify_input(dataset_tensors):

        dataset_tensors_keys = set(dataset_tensors.keys())
        known_keys = dict(
            must_have_keys=[
                REGISTRY_KEY_DISENTANGLEMENT_TARGETS,
                REGISTRY_KEYS.X_KEY,
                REGISTRY_KEYS.BATCH_KEY,
                REGISTRY_KEYS.LABELS_KEY,
            ],
            warning_keys=[REGISTRY_KEYS.CONT_COVS_KEY, REGISTRY_KEYS.SIZE_FACTOR_KEY],
        )
        possible_keys = {j for i in known_keys for j in known_keys[i]}
        possible_keys.add(REGISTRY_KEYS.CAT_COVS_KEY)  # will be used if provided but not must-have or warning-raising.
        possible_keys.add(REGISTRY_KEY_METRICS_COVARIATES_HELPER)

        for must_have_key in known_keys["must_have_keys"]:
            if must_have_key not in dataset_tensors_keys:
                raise ValueError(f"Registry `{must_have_key}` not found in `tensors`: {dataset_tensors.keys()}")

        for warning_key in known_keys["warning_keys"]:
            if warning_key in dataset_tensors_keys:
                warnings.warn(
                    message=f"Registry `{warning_key}` will not be used for counteractive minibatch generation.",
                    category=UserWarning,
                    stacklevel=settings.warnings_stacklevel,
                )

        unexpected_keys = dataset_tensors_keys - possible_keys
        if len(unexpected_keys) > 0:
            raise ValueError(f"Unexpected registry keys: {unexpected_keys}")

    @staticmethod
    def _initialize_verify_config(config):
        _required_kwargs = {"within_batch", "within_labels", "within_categorical_covs"}
        if len(_required_kwargs - set(config.method_kwargs.keys())) > 0:
            raise ValueError(
                "Required method kwarg is missing for possible indice caching step "
                f"of counteractive minibatch generation.\n{config}"
            )

        ecc = copy.deepcopy(DisentanglementManager.anndata_manager_state_registry[REGISTRY_KEYS.CAT_COVS_KEY])
        n_ecc = len(ecc["field_keys"]) if "field_keys" in ecc else 0
        cmk = copy.deepcopy(config.method_kwargs["within_categorical_covs"])
        n_cmk = 0 if cmk is None else len(cmk)
        if n_cmk != n_ecc:
            raise ValueError(
                "Number of categorical covariate should be the same as the "
                "number of elements in `within_categorical_covs`."
            )

    @classmethod
    def _initialize(cls, dataset_tensors, target_obs_key_ind, data_split_identifier, splitter_index, config):

        definitions = DatapointDefinitionsKeyGenerator.create_definitions(
            dict_items=dataset_tensors, target_obs_key_ind=target_obs_key_ind, config=config
        )
        unique_definitions, inverse_indices = np.unique(definitions, axis=0, return_inverse=True)

        obs_key_items = {tuple(map(float, row)): [] for row in unique_definitions}
        for idx, inverse_idx in enumerate(inverse_indices):
            obs_key_items[tuple(unique_definitions[inverse_idx])].append(idx)

        for unique_definition_tuple in obs_key_items:
            obs_key_items[unique_definition_tuple] = np.intersect1d(
                np.array(obs_key_items[unique_definition_tuple]), splitter_index  # only keep the one in the data split
            )
        for unique_definition_tuple in obs_key_items:
            # return the index, relative to the splitter index, but not relative to the dataset
            obs_key_items[unique_definition_tuple] = np.array(
                [np.where(splitter_index == a)[0][0] for a in obs_key_items[unique_definition_tuple]]
            )

        cls._items[data_split_identifier][target_obs_key_ind] = obs_key_items

        # train/test etc -> (obs_key_index) -> (group, label, batch, *categorical_covariate_keys) -> np.ndarray

        # If empty or none -> then just (label, batch)
        # If it is False -> then just (label, batch, all-zero)

    @classmethod
    def reset(cls):
        cls._items = {
            "training": {},
            "validation": {},
            "test": {},
        }

    @classmethod
    def get(cls, dataset_tensors, target_obs_key_ind, data_split_identifier, splitter_index, config):
        try:
            return cls._items[data_split_identifier][target_obs_key_ind]
        except KeyError:
            if data_split_identifier not in cls._items:
                raise ValueError("The `reset` method should be called in the beginning.")
            obs_key = DisentanglementManager.configurations.get_by_index(target_obs_key_ind).obs_key
            cls._initialize_verify_input(dataset_tensors=dataset_tensors)
            cls._initialize_verify_config(config=config)
            cls._initialize(dataset_tensors, target_obs_key_ind, data_split_identifier, splitter_index, config)
            lengths_to_report = ",".join(
                [
                    str(len(cls._items[data_split_identifier][target_obs_key_ind][c]))
                    for c in cls._items[data_split_identifier][target_obs_key_ind].keys()
                ]
            )
            warnings.warn(
                message=(
                    f"Possible group definition indices are calculated for `{obs_key}` for "
                    f"`{data_split_identifier}` set. Number of elements in each group: {lengths_to_report}"
                ),
                category=UserWarning,
                stacklevel=settings.warnings_stacklevel,
            )
            return cls._items[data_split_identifier][target_obs_key_ind]


class CounteractiveGenerator:

    _cached_group_definition_possible_indices_dict: dict = dict()

    @classmethod
    def main(cls, target_obs_key_ind: int, **kwargs):
        """
        Generates a counteractive minibatch based on the specified method.

        Parameters:
        - method: Name of the method to use for generating the minibatch.
        - **kwargs: Keyword arguments passed to the method, including method-specific kwargs and common parameters.
        """
        method = DisentanglementManager.configurations.get_by_index(
            target_obs_key_ind
        ).counteractive_minibatch_settings.method
        if not hasattr(CounteractiveGenerator, method) or method == "main" or not callable(getattr(cls, method)):
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
    def categorical_random(
        cls,
        target_obs_key_ind: int,
        minibatch_tensors: dict[str, torch.Tensor],
        dataset_tensors: dict[str, np.ndarray | spmatrix],
        splitter_index: np.ndarray,
        data_split_identifier: str,
        minibatch_relative_index: list[int],
    ) -> Dict[str, list[int]]:

        config = DisentanglementManager.configurations.items[target_obs_key_ind].counteractive_minibatch_settings
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
            seed=CounteractiveGenerator.configuration_random_seed(config.method_kwargs["seed"])
        )
        n_cat = DisentanglementManager.anndata_manager_state_registry[REGISTRY_KEY_DISENTANGLEMENT_TARGETS][
            "n_cats_per_key"
        ][target_obs_key_ind]
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
            for datapoint, datapoint_index in zip(selection_minibatch_definitions, minibatch_relative_index):
                try:
                    if selection_key == POSITIVE_EXAMPLE_KEY:
                        # choose randomly but exclude the datapoint itself.
                        _selected_element = datapoint_index
                        overhead_counter = 0
                        while _selected_element == datapoint_index:
                            _selected_element = rng.choice(possible_indices[tuple(datapoint)])
                            overhead_counter += 1
                            if overhead_counter > 1e3:
                                # as masking etc is costly, simply raise error after trying generous amount of time
                                to_report = {i: len(possible_indices[i]) for i in possible_indices}
                                raise ValueError(
                                    "The positive example could not be chosen randomly when the anchor itself "
                                    "is excluded. It is likely that some categories in the cached possible indices "
                                    f"dictionary contains only one element. The keys of this dict is `{to_report}`."
                                )
                        _selected_elements.append(_selected_element)
                    else:
                        _selected_elements.append(rng.choice(possible_indices[tuple(datapoint)]))
                except KeyError as e:
                    to_report = {i: len(possible_indices[i]) for i in possible_indices}
                    raise KeyError(
                        f"The minibatch definition `{tuple(datapoint)}` is not found in possible cached indice "
                        f"dictionary. The keys of this dict is `{to_report}` (given as key and number of elements "
                        "within). It happens when the `within` statements are so strict, giving rise to there is no "
                        "corresponding element with such a configuration in the original dataset.In general, "
                        "please note that `within_batch` is the most frequent problem as it is actually "
                        "correlated with many possible metadatas directly."
                    ) from e
            selected_elements[selection_key] = _selected_elements

        return selected_elements
