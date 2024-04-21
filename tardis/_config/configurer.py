#!/usr/bin/env python3

import logging
import os
from functools import cached_property

import yaml


class Configurer:
    """Validate and read config files."""

    CONFIG_FILE_DIRECTORIES_RELATIVE = "directories_relative.yaml"
    CONFIG_FILE_DIRECTORIES_USER = "directories_user.yaml"
    CONFIG_FILE_WANDB_RUN = "wandb_run.yaml"
    CONFIG_FILE_WANDB_USER = "wandb_user.yaml"

    def __init__(self, where):
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.logger = logging.getLogger(__name__)
        self.where = where

    def read_config_yaml(self, file_name):
        """
        This function reads a YAML file and returns its content.
        """
        file_path = os.path.join(self.script_dir, f"{file_name}")
        if not os.path.isfile(file_path) or not os.access(file_path, os.R_OK):
            error_msg = f"Config file is not accessible: {file_name}: {file_path}"
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        with open(file_path) as stream:
            read_yaml = yaml.safe_load(stream)

        return read_yaml

    def _config_directories(self, the_key):
        """
        This function reads the YAML file and validates the directory.
        """
        read_yaml = self.read_config_yaml(Configurer.CONFIG_FILE_DIRECTORIES_RELATIVE)
        read_yaml.update(self.read_config_yaml(Configurer.CONFIG_FILE_DIRECTORIES_USER))

        if the_key not in read_yaml:
            error_msg = f"Key '{the_key}' is not found in directory configurations."
            self.logger.error(error_msg)
            raise KeyError(error_msg)

        read_yaml = read_yaml[the_key]

        if not isinstance(read_yaml, dict) or len(read_yaml) == 0:
            error_msg = f"'{the_key}' should be a non-empty dictionary."
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        return read_yaml

    @cached_property
    def absolute_paths(self):
        """
        This function validates the absolute paths.
        Finds whether it works in local or server.
        """
        read_yaml = self._config_directories(the_key="absolute_paths")
        read_yaml = read_yaml[self.where]

        for k, v in read_yaml.items():
            if not isinstance(k, str) or not isinstance(v, str):
                error_msg = "Keys and values in 'absolute_paths' should be strings."
                self.logger.error(error_msg)
                raise ValueError(error_msg)

            self._check_writable_absolute_path(v, k, v)

        return read_yaml

    def _check_relative_path_yaml_format(self, key, value, return_absolute=False):
        """
        This function checks if the relative path format is valid.
        """
        relative_to = "relative_to"
        path = "path"

        if (
            not isinstance(key, str)
            or not isinstance(value, dict)
            or len(value) != 2
            or relative_to not in value
            or path not in value
        ):
            error_msg = (
                f"Invalid relative path format for '{key}': {value}. Relative path should be defined "
                "with only '{relative_to}' and '{path}' keys."
            )
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        if value[relative_to] not in self.absolute_paths:
            error_msg = f"Relative path refers to an undefined absolute path for '{key}': {value}."
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        if return_absolute:
            return os.path.join(self.absolute_paths[value[relative_to]], value[path])

        return None

    def _check_writable_absolute_path(self, abs_path, k, v):
        """
        This function checks if a directory is writable.
        """
        if not os.path.isdir(abs_path) or not os.access(abs_path, os.W_OK) or not os.path.isabs(abs_path):
            error_msg = f"The directory '{v}' for key '{k}' is not a writable absolute directory."
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        return None

    @cached_property
    def io_directories(self):
        """
        This function validates the IO directories.
        """
        read_yaml = self._config_directories(the_key="io_directories")

        result = dict()
        for k, v in read_yaml.items():
            if not isinstance(k, str) or not isinstance(v, dict):
                error_msg = "Keys in 'io_directories' should be strings and values should be dictionaries."
                self.logger.error(error_msg)
                raise TypeError(error_msg)

            _ap = self._check_relative_path_yaml_format(key=k, value=v, return_absolute=True)
            self._check_writable_absolute_path(_ap, k, v)
            result[k] = _ap

        return result

    @cached_property
    def wandb(self):
        """
        This function validates the wandb configuration.
        """
        read_yaml = self.read_config_yaml(Configurer.CONFIG_FILE_WANDB_RUN)
        read_yaml.update(self.read_config_yaml(Configurer.CONFIG_FILE_WANDB_USER))

        available_keys = {
            "wandblogger_kwargs",
            "environment_variables",
            "login_credentials",
        }

        for key, _ in read_yaml.items():
            if key not in available_keys:
                error_msg = f"Unknown key in W&B configurations: '{key}'"
                self.logger.error(error_msg)
                raise KeyError(error_msg)
            available_keys.remove(key)

        if len(available_keys) > 0:
            error_msg = f"Available remaining keys in W&B configurations': {available_keys}"
            self.logger.error(error_msg)
            raise KeyError(error_msg)

        assert read_yaml["environment_variables"] is None or isinstance(read_yaml["environment_variables"], dict)
        assert isinstance(read_yaml["environment_variables"], dict)
        assert isinstance(read_yaml["login_credentials"], dict)

        k = "save_dir"
        v = read_yaml["wandblogger_kwargs"][k]
        _ap = self._check_relative_path_yaml_format(key=k, value=v, return_absolute=True)
        self._check_writable_absolute_path(_ap, k, v)
        read_yaml["wandblogger_kwargs"][k] = _ap

        return read_yaml


config_server = Configurer(where="server")
config_local = Configurer(where="local")
