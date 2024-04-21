#!/usr/bin/env python3

import copy

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
