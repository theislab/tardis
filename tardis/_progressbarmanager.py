#!/usr/bin/env python3

import copy

from ._myconstants import PROGRESS_BAR_METRICS_KEYS, PROGRESS_BAR_METRICS_MODES


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
            raise NameError(
                "The class should be initialized in `setup_anndata` by `reset` method."
            ) from e
