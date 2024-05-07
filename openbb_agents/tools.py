"""Load OpenBB functions at OpenAI tools for function calling in Langchain"""
import logging
from typing import Any

from langchain.schema import Document
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.vectorstores import VectorStore
from langchain_openai import OpenAIEmbeddings
from openbb import obb

from .models import OpenBBFunctionDescription

logger = logging.getLogger(__name__)


def enable_openbb_llm_mode():
    from openbb import obb

    obb.user.preferences.output_type = "llm"  # type: ignore
    obb.system.python_settings.docstring_sections = ["description", "examples"]  # type: ignore
    obb.system.python_settings.docstring_max_length = 1024  # type: ignore

    import openbb

    openbb.build()


enable_openbb_llm_mode()


def _get_openbb_coverage_providers() -> dict:
    return obb.coverage.providers  # type: ignore


def _get_openbb_user_credentials() -> dict:
    return obb.user.credentials.model_dump()  # type: ignore


def _get_openbb_coverage_command_schemas() -> dict:
    return obb.coverage.command_schemas()  # type: ignore


def get_valid_list_of_providers() -> list[str]:
    credentials = _get_openbb_user_credentials()

    # By default we include yfinance, since it doesn't need a key
    valid_providers = ["yfinance"]
    for name, value in credentials.items():
        if value is not None:
            valid_providers.append(name.split("_api_key")[0].split("_token")[0])
    return valid_providers


def get_valid_openbb_function_names() -> list[str]:
    valid_providers = get_valid_list_of_providers()
    valid_function_names = set()
    for provider in valid_providers:
        try:
            valid_function_names |= set(_get_openbb_coverage_providers()[provider])
        except KeyError:
            pass
    return sorted(list(valid_function_names))


def get_valid_openbb_function_descriptions() -> list[OpenBBFunctionDescription]:
    obb_function_descriptions = []
    for obb_function_name in get_valid_openbb_function_names():
        obb_function_descriptions.append(
            map_name_to_openbb_function_description(obb_function_name)
        )
    return obb_function_descriptions


def map_name_to_openbb_function_description(
    obb_function_name: str,
) -> OpenBBFunctionDescription:
    command_schemas = _get_openbb_coverage_command_schemas()
    dict_ = command_schemas[obb_function_name]
    return OpenBBFunctionDescription(
        name=obb_function_name,
        input_model=dict_["input"],
        output_model=dict_["output"],
        callable=dict_["callable"],
    )


def _get_flat_properties_from_pydantic_model_as_str(model: Any) -> str:
    output_str = ""
    schema_properties = model.schema()["properties"]
    for name, props in schema_properties.items():
        output_str += f"{name}: {props['description']}\n"
    return output_str


def make_vector_index_description(
    openbb_function_description: OpenBBFunctionDescription,
) -> str:
    output_str = ""
    output_str += openbb_function_description.name
    output_str += openbb_function_description.callable.__doc__
    output_str += "\nOutputs:\n"
    output_str += _get_flat_properties_from_pydantic_model_as_str(
        openbb_function_description.output_model
    )
    return output_str


def build_vector_index_from_openbb_function_descriptions(
    openbb_function_descriptions: list[OpenBBFunctionDescription],
) -> VectorStore:
    documents = []
    for function_description in openbb_function_descriptions:
        documents.append(
            Document(
                page_content=make_vector_index_description(function_description),
                metadata={
                    "callable": function_description.callable,
                    "tool_name": function_description.name,
                },
            )
        )
    vector_store = FAISS.from_documents(documents, embedding=OpenAIEmbeddings())
    return vector_store


def build_openbb_tool_vector_index() -> VectorStore:
    logger.info("Building OpenBB tool vector index...")
    return build_vector_index_from_openbb_function_descriptions(
        get_valid_openbb_function_descriptions()
    )
