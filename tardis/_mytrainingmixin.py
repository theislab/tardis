#!/usr/bin/env python3

from __future__ import annotations


from scvi.model.base import UnsupervisedTrainingMixin

from ._mydatasplitter import MyDataSplitter
from ._mytrainrunner import MyTrainRunner
from ._mytrainingplan import MyTrainingPlan


class MyUnsupervisedTrainingMixin(UnsupervisedTrainingMixin):

    _data_splitter_cls = MyDataSplitter
    _training_plan_cls = MyTrainingPlan
    _train_runner_cls = MyTrainRunner

    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)
