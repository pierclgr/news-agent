import os
import json
from typing import Any


class Config:
    def __init__(self, config_path: str = None) -> None:
        """
        Initialize an AgentConfig with a config file.

        Parameters
        ----------
        config_path : str, optional
            The path to the configuration file. If not provided, the default path is used.
        """

        self.config = {}
        self.load(config_path=config_path)

    def load(self, config_path: str = None):
        """
        Load the configuration from a JSON file and store it in the config attribute.

        Parameters
        ----------
        config_path : str, optional
            The path to the configuration file. If not provided, the default path is used.
        """

        # If no config path is provided, use the default path
        if not config_path:
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "config.json")

        # Load the configuration
        with open(config_path) as f:
            self.config = json.load(f)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get the value of a configuration key.

        Parameters
        ----------
        key : str
            The configuration key to retrieve. Can be a dot-notated string for nested fields.
        default : Any, optional
            The default value to return if the key is not found.

        Returns
        -------
        Any
            The value associated with the key or the default value.
        """
        keys = key.split(".")
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value


