#!/usr/bin/env python3

import torch
from ._disentenglementtargetconfigurations import DisentenglementTargetConfigurations


class DisentenglementTargetManager:

    configurations: "DisentenglementTargetConfigurations"
    anndata_manager_state_registry: dict

    @classmethod
    def set_configurations(cls, value):
        cls.configurations = value

    @classmethod
    def set_anndata_manager_state_registry(cls, value):
        cls.anndata_manager_state_registry = value

    @staticmethod
    def random(
        minibatch_index: list,
        splitter_index: torch.Tensor,
        # tensors from setup_anndata
        tensors: dict[torch.Tensor],
        # labels: torch.Tensor,
        # batch: torch.Tensor,
        # REGISTRY_KEY_DISENTENGLEMENT_TARGETS: torch.Tensor,
        # settings
        exclude_itself: bool,
        exclude_group: bool,
        group_size_aware: bool,
        within_label: bool,
        within_batch: bool,
        seed: int | str = "forward_iteration",
    ):
        # TODO: rename the function
        # TODO: write a one main function and subfunctions for this

        # TODO: _disentenglement_targets_configurations has more keys to be put here
        # directly e.g. key, index, indexer_method (for now it is only random) loss
        # Find a way to seamlessly use many functions other than random (for now it will
        # raise Notimplementederror)
        pass