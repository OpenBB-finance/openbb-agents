import argparse
import logging
import logging.config
import os

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

parser = argparse.ArgumentParser(description="Query the OpenBB agent.")
parser.add_argument(
    "query", metavar="query", type=str, help="The query to send to the agent."
)
parser.add_argument(
    "-v", "--verbose", action="store_true", help="Enable verbose logging."
)
args = parser.parse_args()
if args.verbose:
    os.environ["VERBOSE"] = "True"
else:
    os.environ["VERBOSE"] = "False"


# We only import after passing in command line args to have verbosity propagate.
from openbb_agents import agent

query = args.query
result = agent.openbb_agent(query)

print("============")
print("Final Answer")
print("============")
print(result)
