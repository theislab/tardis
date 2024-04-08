#!/usr/bin/env python3

import copy

from scvi.train import TrainingPlan

from ._trainingsteplogger import TrainingStepLogger


class MyTrainingPlan(TrainingPlan):

    def forward(self, *args, **kwargs):
        TrainingStepLogger.set_step(key="gglobal", value=copy.deepcopy(self.global_step))
        TrainingStepLogger.increment_step(key="forward")
        return self.module(*args, **kwargs)

    def training_step(self, *args, **kwargs):
        TrainingStepLogger.increment_step(key="training")
        return super().training_step(*args, **kwargs)

    def validation_step(self, *args, **kwargs):
        TrainingStepLogger.increment_step(key="validation")
        return super().validation_step(*args, **kwargs)

    def test_step(self, *args, **kwargs):
        TrainingStepLogger.increment_step(key="test")
        return super().test_step(*args, **kwargs)

    def predict_step(self, *args, **kwargs):
        TrainingStepLogger.increment_step(key="predict")
        return super().predict_step(*args, **kwargs)
