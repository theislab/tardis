#!/usr/bin/env python3

import numpy as np


class AuxillaryLossWarmupManager:

    items: dict
    _items_helper: dict

    @classmethod
    def reset(cls):
        cls.items = dict()
        cls._items_helper = dict()

    @classmethod
    def add(cls, key, range_list: list | None):
        if range_list is None:
            i = -1
            j = -2
        else:
            i, j = range_list

        cls.items[key] = range_list
        cls._items_helper[key] = (np.arange(i, j + 1) - i) / (j + 1 - i)

    @classmethod
    def get(cls, key, epoch):
        i, j = cls.items[key]
        if epoch < i:
            return 0.0
        elif epoch > j:
            return 1.0
        else:
            return cls._items_helper[key][epoch - i]
