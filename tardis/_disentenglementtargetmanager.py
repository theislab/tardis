#!/usr/bin/env python3


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
