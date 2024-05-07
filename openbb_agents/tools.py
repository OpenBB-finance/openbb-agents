"""Load OpenBB functions at OpenAI tools for function calling in Langchain"""
from typing import Any

from langchain.schema import Document
from langchain.tools import StructuredTool
from langchain_community.vectorstores import FAISS, VectorStore
from langchain_openai import OpenAIEmbeddings
from openbb import obb
from pydantic import BaseModel


def enable_openbb_llm_mode():
    from openbb import obb

    obb.user.preferences.output_type = "llm"  # type: ignore
    obb.system.python_settings.docstring_sections = ["description", "examples"]  # type: ignore
    obb.system.python_settings.docstring_max_length = 1024  # type: ignore

    import openbb

    openbb.build()


enable_openbb_llm_mode()


def create_tool_index(tools: list[StructuredTool]) -> VectorStore:
    """Create a tool index of LangChain StructuredTools."""
    docs = [
        Document(page_content=t.description, metadata={"index": i})
        for i, t in enumerate(tools)
    ]

    vector_store = FAISS.from_documents(docs, OpenAIEmbeddings())
    return vector_store


def create_document(dict):
    ...


class OpenBBFunctionDescription(BaseModel):
    name: str
    input: Any
    output: Any
    callable: Any


def get_openbb_coverage_providers() -> dict:
    return obb.coverage.providers  # type: ignore


def get_openbb_user_credentials() -> dict:
    return obb.user.credentials.model_dump()  # type: ignore


def get_openbb_coverage_command_schemas() -> dict:
    return obb.coverage.command_schemas()  # type: ignore


def get_valid_list_of_providers() -> list[str]:
    credentials = get_openbb_user_credentials()
    valid_providers = []
    for name, value in credentials.items():
        if value is not None:
            valid_providers.append(name.split("_api_key")[0].split("_token")[0])
    return valid_providers


def get_valid_openbb_function_names() -> list[str]:
    valid_providers = get_valid_list_of_providers()
    valid_function_names = set()
    for provider in valid_providers:
        valid_function_names |= set(get_openbb_coverage_providers()[provider])
    return sorted(list(valid_function_names))


def get_valid_openbb_function_descriptions() -> list[OpenBBFunctionDescription]:
    command_schemas = get_openbb_coverage_command_schemas()
    obb_function_descriptions = []
    for obb_function_name in get_valid_openbb_function_names():
        dict_ = command_schemas[obb_function_name]
        obb_function_descriptions.append(
            OpenBBFunctionDescription(
                name=obb_function_name,
                input=dict_["input"],
                output=dict_["output"],
                callable=dict_["callable"],
            )
        )
    return obb_function_descriptions


def make_vector_index_description(
    openbb_function_description: OpenBBFunctionDescription,
) -> str:
    ...
