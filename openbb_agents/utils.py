import logging
import logging.config
import os

from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings

from .models import AnsweredSubQuestion, SubQuestion


def get_verbosity() -> bool:
    return os.environ.get("VERBOSE", "False") == "True"


def configure_logging(verbose: bool):
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",  # Default is stderr
                "formatter": "standard",
            },
        },
        "loggers": {
            "openbb_agents.agent": {
                "handlers": ["console"],
                "level": "INFO" if verbose else "CRITICAL",
                "propagate": False,
            },
            "openbb_agents.utils": {
                "handlers": ["console"],
                "level": "INFO" if verbose else "CRITICAL",
                "propagate": False,
            },
            "openbb_agents.tools": {
                "handlers": ["console"],
                "level": "INFO" if verbose else "CRITICAL",
                "propagate": False,
            },
            "openbb_agents.chains": {
                "handlers": ["console"],
                "level": "INFO" if verbose else "CRITICAL",
                "propagate": False,
            },
        },
        "root": {
            "handlers": ["console"],
            "level": "WARNING" if verbose else "CRITICAL",
        },
    }
    logging.config.dictConfig(logging_config)


def get_dependencies(
    answered_subquestions: list[AnsweredSubQuestion], subquestion: SubQuestion
) -> list[AnsweredSubQuestion]:
    dependency_subquestions = [
        answered_subq
        for answered_subq in answered_subquestions
        if answered_subq.subquestion.id in (subquestion.depends_on or [])
    ]
    return dependency_subquestions

def get_chat_model_name(default: str):
    return os.environ.get("CHAT_MODEL", default=default)

def get_chat_model_base_url():
    return os.environ.get("CHAT_MODEL_BASE_URL", None)

def get_embeddings():
    ollama_embedding_model = os.environ.get("OLLAMA_EMBEDDING_MODEL", None)
    if ollama_embedding_model is not None:
        return OllamaEmbeddings(
            model=ollama_embedding_model
        )
    return OpenAIEmbeddings()
