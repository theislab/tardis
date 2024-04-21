#!/usr/bin/env python3

import copy

import numpy as np
from tardis._myconstants import PROGRESS_BAR_METRICS_KEYS, PROGRESS_BAR_METRICS_MODES


class TrainingEpochLogger:
    current: int

    @classmethod
    def reset(cls):
        cls.current = 0

    @classmethod
    def set_epoch(cls, key, value):
        if hasattr(cls, key):
            setattr(cls, key, value)
        else:
            raise AttributeError(f"{cls.__name__} does not have an attribute '{key}'.")


class TrainingStepLogger:
    forward: int
    gglobal: int
    training: int
    validation: int
    test: int
    predict: int

    @classmethod
    def reset(cls):
        cls.forward = 0
        cls.gglobal = 0
        cls.training = 0
        cls.validation = 0
        cls.test = 0
        cls.predict = 0

    @classmethod
    def set_step(cls, key, value):
        if hasattr(cls, key):
            setattr(cls, key, value)
        else:
            raise AttributeError(f"{cls.__name__} does not have an attribute '{key}'.")

    @classmethod
    def increment_step(cls, key):
        if hasattr(cls, key):
            current_value = getattr(cls, key)
            setattr(cls, key, current_value + 1)
        else:
            raise AttributeError(f"{cls.__name__} does not have an attribute '{key}'.")

    @classmethod
    def print_steps(cls):
        for key in cls.__dict__.keys():
            if not key.startswith("__") and not callable(getattr(cls, key)):
                print(f"{key} = {getattr(cls, key)}")


class ProgressBarManager:

    keys: set[str]
    modes: set[str]

    @classmethod
    def reset(cls):
        cls.keys = copy.deepcopy(PROGRESS_BAR_METRICS_KEYS)
        cls.modes = copy.deepcopy(PROGRESS_BAR_METRICS_MODES)

    @classmethod
    def add(cls, metric_name):
        try:
            cls.keys.add(metric_name)
        except NameError as e:
            raise NameError("The class should be initialized in `setup_anndata` by `reset` method.") from e


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
