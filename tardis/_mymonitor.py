#!/usr/bin/env python3

import copy

import numpy as np

from tardis._myconstants import PROGRESS_BAR_METRICS_KEYS, PROGRESS_BAR_METRICS_MODES


class ModelLevelMetrics:

    _training_sets = ["train", "validation"]
    items: dict

    @classmethod
    def reset(cls):
        cls.items = {i: dict() for i in cls._training_sets}

    @classmethod
    def _add(cls, metric_settings):

        if not isinstance(metric_settings["every_n_epoch"], int):
            raise ValueError
        if not isinstance(metric_settings["subsample"], float):
            raise ValueError
        if not isinstance(metric_settings["metric_identifier"], str):
            raise ValueError
        if not isinstance(metric_settings["training_set"], list):
            raise ValueError
        if not isinstance(metric_settings["progress_bar"], bool):
            raise ValueError
        if not isinstance(metric_settings["metric_kwargs"], dict):
            raise ValueError
        if not all([i in cls._training_sets for i in metric_settings["training_set"]]):
            raise ValueError

        for training_set in metric_settings["training_set"]:
            if metric_settings["metric_identifier"] in cls.items[training_set]:
                raise ValueError
            cls.items[training_set][metric_settings["metric_identifier"]] = {
                "every_n_epoch": metric_settings["every_n_epoch"],
                "subsample": metric_settings["subsample"],
                "metric_kwargs": metric_settings["metric_kwargs"],
            }

        if metric_settings["progress_bar"]:
            if metric_settings["metric_identifier"] in ProgressBarManager.keys:
                raise ValueError
            ProgressBarManager.add(metric_name=metric_settings["metric_identifier"])

    @classmethod
    def add(cls, definitions_dict_list):
        for defs in definitions_dict_list:
            cls._add(defs)


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

        cls.items[key] = [i, j]
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
