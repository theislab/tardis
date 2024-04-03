#!/usr/bin/env python3

from scvi.dataloaders import DataSplitter
from scvi.dataloaders._ann_dataloader import AnnDataLoader
from torch.utils.data import DataLoader

class MyDataLoader(DataLoader):
    pass

class MyAnnDataLoader(AnnDataLoader):
    pass

class MyDataSplitter(DataSplitter):
    data_loader_cls = MyAnnDataLoader
