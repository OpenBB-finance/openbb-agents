import argparse
import logging
import logging.config

from openbb_agents import agent

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
    "-v",
    "--verbose",
    type=bool,
    help="Include verbose output.",
    default=False,
)

args = parser.parse_args()
query = args.query

result = agent.openbb_agent(query)

print("============")
print("Final Answer")
print("============")
print(result)
