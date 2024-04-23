#!/usr/bin/env python3

import numpy as np

from ._disentanglement import Disentanglements
from ._myconstants import REGISTRY_KEY_DISENTANGLEMENT_TARGETS
from ._utils.functions import isnumeric


class DisentanglementManager:

    configurations: "Disentanglements"
    anndata_manager_state_registry: dict
    _categorical_to_value: dict

    @classmethod
    def set_configurations(cls, value):
        cls.configurations = value

    @classmethod
    def set_anndata_manager_state_registry(cls, value):
        cls.anndata_manager_state_registry = value

    @classmethod
    def reset(cls):
        cls.configurations = None
        cls.anndata_manager_state_registry = None
        cls._categorical_to_value = dict()

    @classmethod
    def get_categorical_to_value_dict(cls, obs_key):
        try:
            return cls._categorical_to_value[obs_key]
        except KeyError:
            array = cls.anndata_manager_state_registry[REGISTRY_KEY_DISENTANGLEMENT_TARGETS]["mappings"][obs_key]
            is_numeric = all([isnumeric(element) for element in array])
            # print(obs_key, is_numeric, [element.isnumeric() for element in array])
            cls._categorical_to_value[obs_key] = {i: (float(j) if is_numeric else j) for i, j in enumerate(array)}
            return cls._categorical_to_value[obs_key]

    @classmethod
    def convert_array_categorical_to_value(cls, obs_key, array):
        def _map_values(x):
            return cls.get_categorical_to_value_dict(obs_key)[x]

        vectorized_map = np.vectorize(_map_values)
        return vectorized_map(array)
