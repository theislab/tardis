#!/usr/bin/env python3

import logging
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from scvi import settings
from scvi.train import TrainRunner

logger = logging.getLogger(__name__)


class MyTrainRunner(TrainRunner):

    def __init__(self, *args, simple_logger_indice: Optional[int] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.simple_logger_indice = simple_logger_indice

    def _update_history(self):
        if self.simple_logger_indice is None:
            the_simple_logger = self.trainer.logger
        else:
            the_simple_logger = self.trainer.loggers[self.simple_logger_indice]

        # model is being further trained
        # this was set to true during first training session
        if self.model.is_trained_ is True:
            # if not using the default logger (e.g., tensorboard)
            if not isinstance(self.model.history_, dict):
                warnings.warn(
                    "Training history cannot be updated. Logger can be accessed from " "`model.trainer.logger`",
                    UserWarning,
                    stacklevel=settings.warnings_stacklevel,
                )
                return
            else:
                new_history = the_simple_logger.history
                for key, val in self.model.history_.items():
                    # e.g., no validation loss due to training params
                    if key not in new_history:
                        continue
                    prev_len = len(val)
                    new_len = len(new_history[key])
                    index = np.arange(prev_len, prev_len + new_len)
                    new_history[key].index = index
                    self.model.history_[key] = pd.concat(
                        [
                            val,
                            new_history[key],
                        ]
                    )
                    self.model.history_[key].index.name = val.index.name
        else:
            # set history_ attribute if it exists
            # other pytorch lightning loggers might not have history attr
            try:
                self.model.history_ = the_simple_logger.history
            except AttributeError:
                self.history_ = None
