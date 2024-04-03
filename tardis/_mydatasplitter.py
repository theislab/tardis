#!/usr/bin/env python3

import torch.utils.data._utils
from scvi.dataloaders import DataSplitter
from scvi.dataloaders._ann_dataloader import AnnDataLoader
from torch.utils.data.dataloader import _BaseDataLoaderIter, _SingleProcessDataLoaderIter

from ._myconstants import MODEL_NAME


class CounteractiveMinibatchGenerator:
    # TODO: Rename it for a more general use case other than minibatch generation, e.g. latent index

    _disentenglement_targets_configurations = list()
    _anndata_manager_state_registry = dict()
    # TODO: global_step counter
    # TODO: reserved variable indices

    @classmethod
    def set_disentenglement_targets_configurations(cls, value):
        cls._disentenglement_targets_configurations = value

    @classmethod
    def set_anndata_manager_state_registry(cls, value):
        cls._anndata_manager_state_registry = value

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
        seed: int | str = "global_step",
    ):
        # TODO: rename the function
        # TODO: write a one main function and subfunctions for this

        # TODO: _disentenglement_targets_configurations has more keys to be put here
        # directly e.g. key, index, indexer_method (for now it is only random) loss
        # Find a way to seamlessly use many functions other than random (for now it will
        # raise Notimplementederror)
        pass


class _MySingleProcessDataLoaderIter(_SingleProcessDataLoaderIter):

    def _next_data(self):
        index = self._next_index()  # may raise StopIteration
        data = self._dataset_fetcher.fetch(index)  # may raise StopIteration

        # TODO

        # print("\n#######")
        # print(data["extra_categorical_covs"].shape)
        # print("#######")

        # Full dataset is here, loaded tensors can be configured by setup_anndata
        # self._dataset.dataset.data

        # Train/Validation indices
        # self._dataset.indices

        # Minibatch is here, loaded tensors can be configured by setup_anndata
        # data

        # minibatch index is here:
        # note that this is the index of self._dataset.indices, not the index of real full dataset!!!
        # index (list)

        # use dataset_fetcher... it will create another data.. this is actually
        # all you need for another forward run, so keep it separate, maybe
        # create_counteractive_minibatch_index and call `self._dataset_fetcher.fetch(counteractive_minibatch_index)`

        # a list of dict saying which strategy to use, and settings in strategy method.
        # it should be created in setup_anndata

        # look _disentenglement_targets_configurations to get with method to choose in CounteractiveMinibatchGenerator..
        # use CounteractiveMinibatchGenerator method to get indexes for each metadata....

        # add a key to data: disentenglement_targets.
        # add a key to data[REGISTRY_KEY_DISENTENGLEMENT_TARGETS_TENSORS][age/sex etc] = \ 
        # {X: tensor, labels: tensor, batch: tensor...}
        # sanirim bu yuzden, pin_memory'yi de yenilemen gerekiyor, bu yapi icin uygun degil zira.

        if self._pin_memory:
            data = torch.utils.data._utils.pin_memory.pin_memory(data, self._pin_memory_device)
        return data


class MyAnnDataLoader(AnnDataLoader):

    def _get_iterator(self) -> "_BaseDataLoaderIter":  # torch.DataLoader method
        if self.num_workers == 0:
            return _MySingleProcessDataLoaderIter(self)
        else:
            raise NotImplementedError(
                f"Multiprocessed data loaader not implemented for {MODEL_NAME} model. "
                "_MultiProcessingDataLoaderIter should be edited."
            )


class MyDataSplitter(DataSplitter):
    data_loader_cls = MyAnnDataLoader
