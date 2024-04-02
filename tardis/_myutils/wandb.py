#!/usr/bin/env python3

import wandb


def check_wandb_configurations(wandb_configurations):
    

    # check/start wandb login
    if wandb.api.api_key is None:
        raise ValueError("Not logged in to Weights & Biases. No API Key found. Please login prior to the training.")

    si = wandb.api.viewer_server_info()
    assert len(si) == 2, f"Unexpected error with `viewer_server_info`:\n{len(si)}\n{si}"
    si = si[0]
    for verify_key in ["username", "email"]:
        if si[verify_key] != wandb_configurations["login_credentials"][verify_key]:
            raise ValueError(f"Current `wandb` login conflicts with wandb configuration YAML file: `{verify_key}`")

    try:
        possible_entities = [i["node"]["name"] for i in wandb.api.viewer_server_info()[0]["teams"]["edges"]]
    except KeyError:
        raise KeyError("Unexpected error: Different structure for wandb teams.")
    assert wandb_configurations["login_credentials"]["username"] in possible_entities
    assert wandb_configurations["wandblogger_kwargs"]["entity"] in possible_entities

    e = wandb_configurations["wandblogger_kwargs"]["entity"]
    if wandb.api.entity_is_team(e):
        assert e in possible_entities