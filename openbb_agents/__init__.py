import logging
import logging.config

from openbb_agents.utils import get_verbosity

VERBOSE = get_verbosity()

logging_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "json": {
            "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
            "format": "%(asctime)s %(levelname)s %(name)s %(message)s",
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",  # Default is stderr
            "formatter": "json",
        },
    },
    "loggers": {  # define loggers for specific modules
        "openbb_agents.agent": {
            "handlers": ["console"],
            "level": "INFO",
            "propagate": False,
        },
        "openbb_agents.utils": {
            "handlers": ["console"],
            "level": "ERROR",
            "propagate": False,
        },
        "openbb_agents.chains": {
            "handlers": ["console"],
            "level": "INFO",
            "propagate": False,
        },
    },
    "root": {  # root logger
        "handlers": ["console"],
        "level": "WARNING",
    },
}

logging.config.dictConfig(logging_config)
