#!/usr/bin/env python3

from typing import List, Optional

from ._disentenglement import Disentanglement
from ._myconstants import REGISTRY_KEY_DISENTANGLEMENT_TARGETS


class DisentanglementManager:
    disentanglements: List[Disentanglement] = None
    anndata_manager_state_registry: dict = {}
    _obs_key_to_index: Optional[dict] = {}
    n_total_reserved_latent: int = 0
    n_latent = 0

    @classmethod
    def reset(cls):
        cls.configurations = None
        cls.anndata_manager_state_registry = None
        cls._categorical_to_value = dict()

    @classmethod
    def set_disentanglements(cls, disentanglement_configs):
        cls._validate_configurations(disentanglement_configs)
        cls.disentanglements = []

        for index, config in enumerate(disentanglement_configs):
            obs_key = config["obs_key"]
            disentanglement = Disentanglement(**config)
            cls.disentanglements.append(disentanglement)
            cls._obs_key_to_index[obs_key] = index
            disentanglement.index = index

    @classmethod
    def set_anndata_manager_state_registry(cls, value):
        cls.anndata_manager_state_registry = value

        for disentanglement in cls.disentanglements:
            obs_key = disentanglement.obs_key
            mappings = value[REGISTRY_KEY_DISENTANGLEMENT_TARGETS]["mappings"][obs_key]
            disentanglement.set_mappings(mappings)

    @classmethod
    def get_disentanglement(cls, at) -> str:
        if isinstance(at, int):
            if at >= len(cls.disentanglements):
                raise ValueError("`index` is not available.")
            index = at
        elif isinstance(at, str):
            if at not in cls._obs_key_to_index:
                raise ValueError("`obs_key` is not available.")
            index = cls._obs_key_to_index[at]
        else:
            raise ValueError("Invalid input.")
        return cls.disentanglements[index]

    @classmethod
    def get_ordered_disentanglement_keys(cls):
        return [disentanglement.obs_key for disentanglement in cls.disentanglements]

    @classmethod
    def _validate_configurations(cls, configurations):
        obs_keys = set()
        for config in configurations:
            if config["obs_key"] in obs_keys:
                raise ValueError("Duplicate `obs_key` is not allowed.")
            obs_keys.add(config["obs_key"])

    @classmethod
    def set_indices(cls, n_latent):
        cls.n_latent = n_latent
        for disentanglement in cls.disentanglements:
            start = cls.n_total_reserved_latent
            end = start + disentanglement.n_reserved_latent

            disentanglement.reserved_indices = list(range(start, end))
            disentanglement.unreserved_indices = list(range(0, start)) + list(range(end, n_latent))
            disentanglement.complete_indices = list(range(n_latent))

            cls.n_total_reserved_latent = end
        
        for disentanglement in cls.disentanglements:
            disentanglement.complete_unreserved_indices = list(range(cls.n_total_reserved_latent, cls.n_latent))

        if n_latent - cls.n_total_reserved_latent < 1:
            raise ValueError("Not enough latent space variables to reserve for targets.")
