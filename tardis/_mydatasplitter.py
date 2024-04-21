#!/usr/bin/env python3

import torch.utils.data._utils
from scvi.dataloaders import DataSplitter
from scvi.dataloaders._ann_dataloader import AnnDataLoader
from torch.utils.data.dataloader import _BaseDataLoaderIter, _SingleProcessDataLoaderIter

from ._counteractivegenerator import CounteractiveGenerator
from ._disentanglementmanager import DisentanglementManager
from ._myconstants import MODEL_NAME, REGISTRY_KEY_DISENTANGLEMENT_TARGETS_TENSORS


class _MySingleProcessDataLoaderIter(_SingleProcessDataLoaderIter):

    def __init__(self, *args, data_split_identifier, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_split_identifier = data_split_identifier

    def _next_data(self):
        index = self._next_index()  # may raise StopIteration
        data = self._dataset_fetcher.fetch(index)  # may raise StopIteration

        if len(DisentanglementManager.configurations) > 0:
            data[REGISTRY_KEY_DISENTANGLEMENT_TARGETS_TENSORS] = dict()
            # `data` is simply the minibatch itself.
            # The aim is create alternative minibatches that has the same keys as the original one
            # These minibatches, called counteractive minibatch, will be fed through `forward` method.
            for target_obs_key_ind, target_obs_key in enumerate(
                DisentanglementManager.configurations.get_ordered_obs_key()
            ):
                data[REGISTRY_KEY_DISENTANGLEMENT_TARGETS_TENSORS][target_obs_key] = dict()
                target_obs_key_tensors_indices_dict = CounteractiveGenerator.main(
                    target_obs_key_ind=target_obs_key_ind,
                    # Full dataset and minibatch, loaded tensors can be configured by setup_anndata.
                    minibatch_tensors=data,
                    dataset_tensors=self._dataset.dataset.data,
                    # Train/validation/test indices. Maximum indice is the dataset size.
                    # Number of indices are defined by `train_size` parameter of Trainer.
                    splitter_index=self._dataset.indices,
                    data_split_identifier=self.data_split_identifier,
                    # Minibatch indices. Maximum indice is defined by `dataset` length and `train_size`.
                    # Note that this is index of `self._dataset.indices`, not the index of real full dataset.
                    minibatch_relative_index=index,
                )
                for selection_key, target_obs_key_tensors_indices in target_obs_key_tensors_indices_dict.items():
                    target_obs_key_tensors = self._dataset_fetcher.fetch(target_obs_key_tensors_indices)
                    data[REGISTRY_KEY_DISENTANGLEMENT_TARGETS_TENSORS][target_obs_key][
                        selection_key
                    ] = target_obs_key_tensors

        if self._pin_memory:
            # It is a recursive function, so deep dict objects should not interfere the process.
            data = torch.utils.data._utils.pin_memory.pin_memory(data, self._pin_memory_device)
        return data


class MyAnnDataLoader(AnnDataLoader):

    def __init__(self, *args, data_split_identifier, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_split_identifier = data_split_identifier

    def _get_iterator(self) -> "_BaseDataLoaderIter":  # torch.DataLoader method
        if self.num_workers == 0:
            return _MySingleProcessDataLoaderIter(loader=self, data_split_identifier=self.data_split_identifier)
        else:
            raise NotImplementedError(
                f"Multiprocessed data loaader not implemented for {MODEL_NAME} model. "
                "_MultiProcessingDataLoaderIter should be edited."
            )


class MyDataSplitter(DataSplitter):
    data_loader_cls = MyAnnDataLoader

    def train_dataloader(self):
        """Create train data loader."""
        return self.data_loader_cls(
            self.adata_manager,
            indices=self.train_idx,
            shuffle=True,
            drop_last=False,
            load_sparse_tensor=self.load_sparse_tensor,
            pin_memory=self.pin_memory,
            data_split_identifier="training",
            **self.data_loader_kwargs,
        )

    def val_dataloader(self):
        """Create validation data loader."""
        if len(self.val_idx) > 0:
            return self.data_loader_cls(
                self.adata_manager,
                indices=self.val_idx,
                shuffle=False,
                drop_last=False,
                load_sparse_tensor=self.load_sparse_tensor,
                pin_memory=self.pin_memory,
                data_split_identifier="validation",
                **self.data_loader_kwargs,
            )
        else:
            pass

    def test_dataloader(self):
        """Create test data loader."""
        if len(self.test_idx) > 0:
            return self.data_loader_cls(
                self.adata_manager,
                indices=self.test_idx,
                shuffle=False,
                drop_last=False,
                load_sparse_tensor=self.load_sparse_tensor,
                pin_memory=self.pin_memory,
                data_split_identifier="test",
                **self.data_loader_kwargs,
            )
        else:
            pass

    def predict_dataloader(self):
        raise NotImplementedError

    @staticmethod
    def convert_sparse_to_dense(batch):
        """Recursively converts all sparse tensors in a nested dictionary to dense tensors."""
        for key, value in batch.items():
            if isinstance(value, dict):
                batch[key] = MyDataSplitter.convert_sparse_to_dense(value)  # Recursive call for nested dictionaries
            elif isinstance(value, torch.Tensor) and value.layout in (torch.sparse_csr, torch.sparse_csc):
                batch[key] = value.to_dense()  # Convert sparse tensor to dense
        return batch

    def on_after_batch_transfer(self, batch, dataloader_idx):
        """Converts sparse tensors to dense if necessary."""
        if self.load_sparse_tensor:
            return MyDataSplitter.convert_sparse_to_dense(batch)
        return batch
