#!/usr/bin/env python3

MODEL_NAME = "tardis"
REGISTRY_KEY_DISENTENGLEMENT_TARGETS = "disentanglement_target"
REGISTRY_KEY_DISENTENGLEMENT_TARGETS_TENSORS = "disentanglement_target_tensors"

minified_method_not_supported_message = (
    f"{MODEL_NAME} model currently does not support minified data."
)

LOSS_NAMING_DELIMITER = "|"
LOSS_NAMING_PREFIX = "tardis"
LOSS_MEAN_BEFORE_WEIGHT = LOSS_NAMING_DELIMITER.join(
    [LOSS_NAMING_PREFIX, "mean", "before_weight"]
)

PROGRESS_BAR_METRICS_KEYS = {"total_loss", "kl_local"}
PROGRESS_BAR_METRICS_MODES = {"train"}  # "validation"
LOSS_TYPES = ["complete", "reserved", "unreserved"]
