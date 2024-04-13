#!/usr/bin/env python3

from typing import Any, List, Optional

from pydantic import (
    BaseModel,
    StrictInt,
    StrictStr,
    field_validator,
)  # ValidationError

from ._myconstants import (
    LOSS_NAMING_DELIMITER,
    LOSS_NAMING_PREFIX,
    LATENT_INDEX_GROUP_NAMES,
)
from ._progressbarmanager import ProgressBarManager


class CounteractiveMinibatchSettings(BaseModel):
    method: StrictStr
    # Accepts any dict without specific type checking.
    method_kwargs: dict

    # for now only `random` implemented for counteractive_minibatch_method, raise ValueError in method selection.
    # seed should be in method_kwargs: str or `global_seed` etc


class AuxillaryLosses(BaseModel):
    complete_latent: Any
    reserved_subset: Any
    unreserved_subset: Any
    # items: List[str] = {"complete_latent", "reserved_subset", "unreserved_subset"}


class DisentenglementTargetConfiguration(BaseModel):
    obs_key: StrictStr
    n_reserved_latent: StrictInt
    counteractive_minibatch_settings: CounteractiveMinibatchSettings
    auxillary_losses: AuxillaryLosses
    # This is set after the initialization based on the index of the target in the provided list.
    index: Optional[int] = None
    # This is called once during model initialization.
    reserved_latent_indices: Optional[List[int]] = None  # reserved by only this target
    unreserved_latent_indices: Optional[List[int]] = (
        None  # unreserved by only this target
    )

    @field_validator("n_reserved_latent")
    def n_reserved_latent_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError("`n_reserved_latent` must be more than or equal to 0.")
        return v

    @field_validator("index")
    def index_must_undefined_before_init(cls, v):
        if v is not None:
            raise ValueError("`index` should not be defined by the user.")
        return v

    @field_validator("reserved_latent_indices")
    def reserved_latent_indices_must_undefined_before_init(cls, v):
        if v is not None:
            raise ValueError(
                "`reserved_latent_indices` should not be defined by the user."
            )
        return v

    @field_validator("unreserved_latent_indices")
    def unreserved_latent_indices_must_undefined_before_init(cls, v):
        if v is not None:
            raise ValueError(
                "`unreserved_latent_indices` should not be defined by the user."
            )
        return v

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class DisentenglementTargetConfigurations(BaseModel):
    items: List[DisentenglementTargetConfiguration] = []
    # unreserved by any of the configuration.
    unreserved_latent_indices: Optional[List[int]] = None
    # complete list of indices, simply range(n_latent)
    latent_indices: Optional[List[int]] = None
    # filled by __init__
    _index_to_obs_key: dict[str:int] = {}
    _obs_key_to_index: dict[str:int] = {}
    # progress_bar_metrics: set() = copy.deepcopy()

    @field_validator("unreserved_latent_indices")
    def unreserved_latent_indices_must_undefined_before_init(cls, v):
        if v is not None:
            raise ValueError(
                "`unreserved_latent_indices` should not be defined by the user."
            )
        return v

    @field_validator("latent_indices")
    def latent_indices_must_undefined_before_init(cls, v):
        if v is not None:
            raise ValueError("`latent_indices` should not be defined by the user.")
        return v

    @field_validator("items")
    def check_unique_obs_keys(cls, configurations):
        obs_keys = set()
        for config in configurations:
            if config.obs_key in obs_keys:
                raise ValueError(f"Duplicate 'obs_key' detected: {config.obs_key}")
            obs_keys.add(config.obs_key)
        return configurations

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Add index to each configuration post-initialization
        for index, config in enumerate(self.items):
            config.index = index
            self._index_to_obs_key[config.index] = config.obs_key
            self._obs_key_to_index[config.obs_key] = config.index

            for auxillary_loss_key in LATENT_INDEX_GROUP_NAMES:
                loss_config = getattr(config.auxillary_losses, auxillary_loss_key)
                loss_identifier_string = LOSS_NAMING_DELIMITER.join(
                    [LOSS_NAMING_PREFIX, config.obs_key, auxillary_loss_key]
                )
                if loss_config["progress_bar"]:
                    ProgressBarManager.add(loss_identifier_string)

    def __len__(self):
        return len(self.items)

    def get_by_index(self, index: int) -> str:
        if index not in self._index_to_obs_key:
            raise ValueError("`index` is not available.")
        return self.items[index]

    def get_by_obs_key(self, obs_key: str) -> int:
        if obs_key not in self._obs_key_to_index:
            raise ValueError("`obs_key` is not available.")
        index = self._obs_key_to_index[obs_key]
        return self.get_by_index(index=index)

    def get_ordered_obs_key(self) -> list[str]:
        return [self._index_to_obs_key[i] for i in range(len(self.items))]
