#!/usr/bin/env python3

import torch.utils.data._utils
from scvi.dataloaders import DataSplitter
from scvi.dataloaders._ann_dataloader import AnnDataLoader
from torch.utils.data.dataloader import _BaseDataLoaderIter, _SingleProcessDataLoaderIter

from ._disentenglementtargetmanager import DisentenglementTargetManager, CounteractiveMinibatchGenerator
from ._myconstants import MODEL_NAME, REGISTRY_KEY_DISENTENGLEMENT_TARGETS_TENSORS, REGISTRY_KEY_DISENTENGLEMENT_TARGETS


class _MySingleProcessDataLoaderIter(_SingleProcessDataLoaderIter):

    def _next_data(self):
        index = self._next_index()  # may raise StopIteration
        data = self._dataset_fetcher.fetch(index)  # may raise StopIteration

        data[REGISTRY_KEY_DISENTENGLEMENT_TARGETS_TENSORS] = dict()
        if len(DisentenglementTargetManager.configurations) > 0:
            # `data` is simply the minibatch itself.
            original_tensor_keys_sorted = sorted(data.keys())
            # The aim is create alternative minibatches that has the same keys as the original one
            # These minibatches, called counteractive minibatch, will be fed through `forward` method.
            for target_obs_key_ind, target_obs_key in enumerate(
                DisentenglementTargetManager.configurations.get_ordered_obs_key()
            ):
                target_config = DisentenglementTargetManager.configurations.items[
                    target_obs_key_ind
                ].counteractive_minibatch_settings
                target_obs_key_tensors_indices = CounteractiveMinibatchGenerator.main(
                    # Full dataset, loaded tensors can be configured by setup_anndata.
                    dataset_tensors=self._dataset.dataset.data,
                    target_obs_key=target_obs_key,
                    # Train/validation/test indices. Maximum indice is the dataset size.
                    # Number of indices are defined by `train_size` parameter of Trainer.
                    splitter_index=self._dataset.indices,
                    # Minibatch indices. Maximum indice is defined by `dataset` length and `train_size`.
                    # Note that this is index of `self._dataset.indices`, not the index of real full dataset.
                    minibatch_relative_index=index,
                    # The method and method kwargs for counteractive_minibatch_settings, defined by the user.
                    method=target_config.method,
                    **target_config.method_kwargs,
                )
                # assert len(target_obs_key_tensors_indices) == len(index), "Remove after development - 1."
                # target_obs_key_tensors = self._dataset_fetcher.fetch(target_obs_key_tensors_indices)
                # assert original_tensor_keys_sorted == sorted(target_obs_key_tensors.keys()), "Remove after development - 2."
                # data[REGISTRY_KEY_DISENTENGLEMENT_TARGETS_TENSORS][target_obs_key] = target_obs_key_tensors

        if False:
            print("\n###Remove here after model development!###")
            print(data[REGISTRY_KEY_DISENTENGLEMENT_TARGETS_TENSORS]["sample_ID"])
            print("#")

        if self._pin_memory:
            raise NotImplementedError(  # TODO
                "pin_memory needs to be adapted, it is not suitable for deep dictionary structure (or empty dict)."
            )
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
