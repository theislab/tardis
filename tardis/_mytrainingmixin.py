#!/usr/bin/env python3

from __future__ import annotations

from scvi._types import Tunable
from scvi.model._utils import get_max_epochs_heuristic, use_distributed_sampler
from scvi.model.base import UnsupervisedTrainingMixin
from scvi.train._logger import SimpleLogger

from ._mydatasplitter import MyDataSplitter
from ._mytrainingplan import MyTrainingPlan
from ._mytrainrunner import MyTrainRunner


class MyUnsupervisedTrainingMixin(UnsupervisedTrainingMixin):

    _data_splitter_cls = MyDataSplitter
    _training_plan_cls = MyTrainingPlan
    _train_runner_cls = MyTrainRunner

    def train(
        self,
        max_epochs: int | None = None,
        accelerator: str = "auto",
        devices: int | list[int] | str = "auto",
        train_size: float = 0.9,
        validation_size: float | None = None,
        shuffle_set_split: bool = True,
        load_sparse_tensor: bool = False,
        batch_size: Tunable[int] = 128,
        early_stopping: bool = False,
        datasplitter_kwargs: dict | None = None,
        plan_kwargs: dict | None = None,
        **trainer_kwargs,
    ):

        if "logger" in trainer_kwargs.keys() and hasattr(self, "wandb_logger"):
            raise ValueError(
                "If a custom `logger` (tensorboard etc) is defined in `trainer_kwargs` to be used in Trainer, "
                "then `wandb_logger` should not have been initialized by the method in MyModel."
            )
        elif hasattr(self, "wandb_logger"):
            # If `wandb_logger` is initialized by the method in MyModel,
            # then Trainer should accept both wandb_logger and standard scvi-tools logger.
            # It was not a must but it is nice to have both to have a deeper look.
            trainer_kwargs["logger"] = [self.wandb_logger, SimpleLogger()]
            # Keep the indice of the SimpleLogger in above list to be used in `TrainerRunner._update_history`.
            simple_logger_indice = 1

            if self.wandb_logger_verbose:
                wandb_message_initialized = (
                    f"W&B logger initialized with the following parameters: \n"
                    f"Entity: {self.wandb_logger.experiment.entity}\n"
                    f"Project: {self.wandb_logger.experiment.project}\n"
                    f"ID: {self.wandb_logger.experiment.id}\n"
                    f"Name: {self.wandb_logger.experiment.name}\n"
                    f"Tags: {', '.join(self.wandb_logger.experiment.tags)}\n"
                    f"Notes: {self.wandb_logger.experiment.notes}\n"
                    f"URL: {self.wandb_logger.experiment.url}\n"
                    f"Directory: {self.wandb_logger.experiment.dir}\n"
                )
                print(wandb_message_initialized)
        else:
            # If wandb_logger is not defined, there will be already only SimpleLogger as the logger, this will create
            # `trainer.logger` instead of `trainer.loggers` which is a list object.
            simple_logger_indice = None

        try:
            exit_code = 1
            if max_epochs is None:
                max_epochs = get_max_epochs_heuristic(self.adata.n_obs)

            datasplitter_kwargs = datasplitter_kwargs or {}
            data_module = self._data_splitter_cls(
                self.adata_manager,
                train_size=train_size,
                validation_size=validation_size,
                batch_size=batch_size,
                shuffle_set_split=shuffle_set_split,
                distributed_sampler=use_distributed_sampler(trainer_kwargs.get("strategy", None)),
                load_sparse_tensor=load_sparse_tensor,
                **datasplitter_kwargs,
            )

            plan_kwargs = plan_kwargs or {}
            training_plan = self._training_plan_cls(self.module, **plan_kwargs)

            es = "early_stopping"
            trainer_kwargs[es] = early_stopping if es not in trainer_kwargs.keys() else trainer_kwargs[es]
            runner = self._train_runner_cls(
                self,
                training_plan=training_plan,
                data_splitter=data_module,
                max_epochs=max_epochs,
                accelerator=accelerator,
                devices=devices,
                simple_logger_indice=simple_logger_indice,
                **trainer_kwargs,
            )
            runner()
            exit_code = 0

        finally:
            if hasattr(self, "wandb_logger"):
                self.wandb_logger.experiment.finish(exit_code=exit_code, quiet=True)
                if self.wandb_logger_verbose:
                    wandb_message_finalized = f"W&B logger finalized successfully: \n" f"Exit Code: {exit_code}\n"
                    print(wandb_message_finalized)
