#!/usr/bin/env python3

from typing import List, Literal, Optional

from pydantic import BaseModel, StrictBool, StrictFloat, StrictInt, StrictStr, field_validator

from ._myconstants import EXAMPLE_KEYS, LATENT_INDEX_GROUP_NAMES, LOSS_NAMING_DELIMITER, LOSS_NAMING_PREFIX
from ._mymonitor import AuxillaryLossWarmupManager, ProgressBarManager


class TardisLoss(BaseModel):
    apply: StrictBool
    target_type: Literal["categorical", "pseudo_categorical"]
    warmup_epoch_range: List[int] | None
    weight: StrictFloat
    transformation: StrictStr
    method: StrictStr
    progress_bar: StrictBool
    latent_group: StrictStr
    counteractive_example: StrictStr
    # if type is pseudo_categorical this should be set to something
    non_categorical_coefficient_method: str | None = None
    # Accepts any dict without specific type checking.
    method_kwargs: dict
    # This is set after the initialization based on the index of the target in the provided list.
    index: int | None = None
    loss_identifier_string: str | None = None

    @field_validator("non_categorical_coefficient_method")
    def non_categorical_coefficient_method_must_be_defined_for_pseudo_categorical_loss(cls, v, values):
        target_type = values.data["target_type"]
        if target_type == "pseudo_categorical" and not isinstance(v, str):
            raise ValueError(
                "`non_categorical_coefficient_method` should be defined if `target_type` is `pseudo_categorical`."
            )
        elif target_type != "pseudo_categorical" and v is not None:
            raise ValueError(
                f"`non_categorical_coefficient_method` (`{v}`) should be `None` "
                f"if `target_type` is not `pseudo_categorical` (`{target_type}`)."
            )
        return v

    @field_validator("warmup_epoch_range")
    def warmup_epoch_range_must_have_two_length(cls, v):
        if v is None or (isinstance(v, list) and len(v) == 2 and all([isinstance(k, int) for k in v]) and v[0] <= v[1]):
            return v
        raise ValueError(f"`warmup_epoch_range` should be `None` or list of integers with lenght 2: {v}")

    @field_validator("index")
    def index_must_undefined_before_init(cls, v):
        if v is not None:
            raise ValueError("`index` should not be defined by the user.")
        return v

    @field_validator("weight")
    def weight_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError("`weight` must be more than or equal to 0.")
        return v

    @field_validator("loss_identifier_string")
    def loss_identifier_string_must_undefined_before_init(cls, v):
        if v is not None:
            raise ValueError("`loss_identifier_string` should not be defined by the user.")
        return v

    @field_validator("latent_group")
    def latent_group_must_be_defined_in_constants(cls, v):
        if v not in LATENT_INDEX_GROUP_NAMES:
            raise ValueError(f"`latent_group` (`{v}`) should be one of `{LATENT_INDEX_GROUP_NAMES}`.")
        return v

    @field_validator("counteractive_example")
    def counteractive_example_must_be_defined_in_constants(cls, v):
        if v not in EXAMPLE_KEYS:
            raise ValueError(f"`counteractive_example` (`{v}`) should be one of `{EXAMPLE_KEYS}`.")
        return v

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Add index to each configuration post-initialization


class CounteractiveSettings(BaseModel):
    method: StrictStr
    # Accepts any dict without specific type checking.
    method_kwargs: dict

    # for now only `random` implemented for counteractive_minibatch_method, raise ValueError in method selection.
    # seed should be in method_kwargs: str or `global_seed` etc


class AuxillaryLosses(BaseModel):
    items: List[TardisLoss] = []

    def __len__(self):
        return len(self.items)

    def get_by_identifier(self, identifier: str) -> TardisLoss:
        for item in self.items:
            if item.loss_identifier_string == identifier:
                return item
        raise KeyError("`identifier` is not amongst the defined losses.")


class Disentanglement(BaseModel):
    obs_key: StrictStr
    n_reserved_latent: StrictInt
    counteractive_minibatch_settings: CounteractiveSettings
    auxillary_losses: List[TardisLoss] = []
    # This is set after the initialization based on the index of the target in the provided list.
    index: Optional[int] = None
    # This is called once during model initialization.
    reserved_latent_indices: Optional[List[int]] = None  # reserved by only this target
    unreserved_latent_indices: Optional[List[int]] = None  # unreserved by only this target

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
            raise ValueError("`reserved_latent_indices` should not be defined by the user.")
        return v

    @field_validator("unreserved_latent_indices")
    def unreserved_latent_indices_must_undefined_before_init(cls, v):
        if v is not None:
            raise ValueError("`unreserved_latent_indices` should not be defined by the user.")
        return v

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Add full_name and config progress bar each configuration post-initialization

        for index, auxillary_loss in enumerate(self.auxillary_losses):
            auxillary_loss.index = index
            auxillary_loss.loss_identifier_string = LOSS_NAMING_DELIMITER.join(
                [LOSS_NAMING_PREFIX, self.obs_key, str(auxillary_loss.index)]
            )


class Disentanglements(BaseModel):
    items: List[Disentanglement] = []
    # unreserved by any of the configuration.
    unreserved_latent_indices: Optional[List[int]] = None
    reserved_latent_indices: Optional[List[int]] = None
    # complete list of indices, simply range(n_latent)
    latent_indices: Optional[List[int]] = None
    # filled by __init__
    _index_to_obs_key: dict[str:int] = {}
    _obs_key_to_index: dict[str:int] = {}
    # progress_bar_metrics: set() = copy.deepcopy()

    @field_validator("unreserved_latent_indices")
    def unreserved_latent_indices_must_undefined_before_init(cls, v):
        if v is not None:
            raise ValueError("`unreserved_latent_indices` should not be defined by the user.")
        return v

    @field_validator("reserved_latent_indices")
    def reserved_latent_indices_must_undefined_before_init(cls, v):
        if v is not None:
            raise ValueError("`reserved_latent_indices` should not be defined by the user.")
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
        for config in self.items:
            self._index_to_obs_key[config.index] = config.obs_key
            self._obs_key_to_index[config.obs_key] = config.index
        for config in self.items:
            for auxillary_loss in config.auxillary_losses:
                if auxillary_loss.progress_bar:
                    ProgressBarManager.add(auxillary_loss.loss_identifier_string)
                AuxillaryLossWarmupManager.add(auxillary_loss.loss_identifier_string, auxillary_loss.warmup_epoch_range)

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
