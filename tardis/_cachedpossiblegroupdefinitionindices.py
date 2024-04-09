#!/usr/bin/env python3

import copy
import warnings
from typing import Dict

import numpy as np
import torch
from scvi import REGISTRY_KEYS, settings

from ._disentenglementtargetmanager import DisentenglementTargetManager
from ._myconstants import REGISTRY_KEY_DISENTENGLEMENT_TARGETS


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

        if config.method_kwargs["within_other_groups"] and operation_mode_torch:
            # This is called when minibatch_definitions is being created, not during indice caching.
            group_definitions = dict_items[REGISTRY_KEY_DISENTENGLEMENT_TARGETS][:, target_obs_key_ind].view(-1, 1)
            # The if the calculated group_definitions is not changed, then the
            # counteractive minibatch will be always within the same group.
            # The randomization of this vector to any other category than the original one
            # is done after `minibatch_definitions` is created.
        elif config.method_kwargs["within_other_groups"]:
            group_definitions = (
                dict_items[REGISTRY_KEY_DISENTENGLEMENT_TARGETS].iloc[:, target_obs_key_ind].values.reshape(-1, 1)
            )
        elif operation_mode_torch:
            group_definitions = torch.zeros_like(batch_definitions)
        else:
            group_definitions = np.zeros_like(batch_definitions)

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
                REGISTRY_KEY_DISENTENGLEMENT_TARGETS,
                REGISTRY_KEYS.X_KEY,
                REGISTRY_KEYS.BATCH_KEY,
                REGISTRY_KEYS.LABELS_KEY,
            ],
            warning_keys=[REGISTRY_KEYS.CONT_COVS_KEY, REGISTRY_KEYS.SIZE_FACTOR_KEY],
        )
        possible_keys = {j for i in known_keys for j in known_keys[i]}
        possible_keys.add(REGISTRY_KEYS.CAT_COVS_KEY)  # will be used if provided but not must-have or warning-raising.

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
        _required_kwargs = {"within_batch", "within_labels", "within_categorical_covs", "within_other_groups"}
        if len(_required_kwargs - set(config.method_kwargs.keys())) > 0:
            raise ValueError(
                "Required method kwarg is missing for possible indice caching step "
                f"of counteractive minibatch generation.\n{config}"
            )

        ecc = copy.deepcopy(DisentenglementTargetManager.anndata_manager_state_registry[REGISTRY_KEYS.CAT_COVS_KEY])
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
            obs_key = DisentenglementTargetManager.configurations.get_by_index(target_obs_key_ind).obs_key
            warnings.warn(
                message=f"Possible group definition indices are calculating for `{obs_key}`.",
                category=UserWarning,
                stacklevel=settings.warnings_stacklevel,
            )
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
                message=f"Number of elements in each group for `{obs_key}`: {lengths_to_report}",
                category=UserWarning,
                stacklevel=settings.warnings_stacklevel,
            )
            return cls._items[data_split_identifier][target_obs_key_ind]
