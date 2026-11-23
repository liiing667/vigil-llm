import os
import sys

from loguru import logger

import configparser

from typing import Optional, List, Dict


# Value shown in place of a secret when config is exposed publicly
REDACTED_VALUE = '**********'

# Substrings that mark a config key as holding a secret
# (model keys, scanner tokens, vectordb / network credentials, ...)
SENSITIVE_MARKERS = ('key', 'token', 'secret', 'password', 'credential')


def is_sensitive_key(key: str) -> bool:
    """ Classify a config key as secret based on its name """
    normalized = key.lower()
    return any(marker in normalized for marker in SENSITIVE_MARKERS)


def redact_params(params: Dict[str, str]) -> Dict[str, str]:
    """ Return a copy of a config section with sensitive values masked """
    return {
        key: (REDACTED_VALUE if is_sensitive_key(key) else value)
        for key, value in params.items()
    }


class Config:
    def __init__(self, config_file: str):
        self.config_file = config_file
        self.config = configparser.ConfigParser()
        if not os.path.exists(self.config_file):
            logger.error(f'Config file not found: {self.config_file}')
            raise ValueError(f'Config file not found: {self.config_file}')

        logger.info(f'Loading config file: {self.config_file}')
        self.config.read(config_file)

    def get_val(self, section: str, key: str) -> Optional[str]:
        answer = None

        try:
            answer = self.config.get(section, key)
        except Exception as err:
            logger.error(f'Config file missing section: {section} - {err}')

        return answer

    def get_bool(self, section: str, key: str, default: bool = False) -> bool:
        try:
            return self.config.getboolean(section, key)
        except Exception as err:
            logger.error(f'Failed to parse boolean - returning default "False": {section} - {err}')
            return default

    def get_section(self, section: str) -> Dict[str, str]:
        """ Full (internal) parameters for a config section """
        if not self.config.has_section(section):
            logger.error(f'Config file missing section: {section}')
            return {}
        return dict(self.config.items(section))

    def get_scanner_config(self, scanner_name: str, redact: bool = False) -> Dict[str, str]:
        params = self.get_section(f'scanner:{scanner_name}')
        return redact_params(params) if redact else params

    def get_general_config(self):
        """ Full config for internal startup use (contains secrets) """
        return {section: self.get_section(section) for section in self.config.sections()}

    def get_public_config(self):
        """ Redacted config that is safe to expose (e.g. /settings endpoint) """
        return {section: redact_params(self.get_section(section)) for section in self.config.sections()}
    
    def get_scanner_names(self, scanner_type: str) -> List[str]:
        return self.get_val('scanners', scanner_type).split(',')

