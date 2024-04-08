#!/usr/bin/env python3


class TrainingStepLogger:
    forward = 0
    gglobal = 0
    training = 0
    validation = 0
    test = 0
    predict = 0

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