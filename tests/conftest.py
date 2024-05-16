from typing import Any, Callable
from unittest.mock import patch

import pytest
from langchain_core.vectorstores import VectorStore
from openbb import obb
from pydantic import BaseModel, Field

from openbb_agents.tools import build_openbb_tool_vector_index


@pytest.fixture
def mock_obb_user_credentials(monkeypatch):
    class TestCredentials(BaseModel):
        fmp_api_key: str | None
        intrinio_token: str | None
        benzinga_api_key: str | None

    mock_credentials = TestCredentials(
        # NB: We explicitly set the benzinga key to None!
        fmp_api_key="some-value",
        intrinio_token="another-value",
        benzinga_api_key=None,
    )
    monkeypatch.setattr(obb.user, "credentials", mock_credentials)  # type: ignore


@pytest.fixture
def mock_obb_coverage_providers(mock_obb_user_credentials):
    mock_provider_coverage_dict = {
        "fmp": ["function_a", "function_b"],
        "intrinio": ["function_a", "function_c"],
        "benzinga": ["function_d"],
    }
    with patch("openbb_agents.tools._get_openbb_coverage_providers") as mock:
        mock.return_value = mock_provider_coverage_dict
        yield mock


@pytest.fixture
def mock_openbb_function_output_model() -> Any:
    class TestOutputModel(BaseModel):
        first_property: str = Field(description="The first property")
        second_property: int = Field(description="The second property")
        third_property: list[float] = Field(description="The third property")

    return TestOutputModel


@pytest.fixture
def mock_openbb_function_callable() -> Callable:
    def test_callable():
        """A callable test function that does nothing, but has a docstring."""
        ...

    return test_callable


@pytest.fixture
def mock_obb_coverage_command_schema(
    mock_obb_coverage_providers,
):
    mock_coverage_command_schema_dict = {
        "function_a": {
            "input": "mock input model for a",
            "output": "mock output model for a",
            "callable": "<callable for a>",
        },
        "function_b": {
            "input": "mock input model for b",
            "output": "mock output model for b",
            "callable": "<callable for b>",
        },
        "function_c": {
            "input": "mock input model for c",
            "output": "mock output model for c",
            "callable": "<callable for c>",
        },
        "function_d": {
            "input": "mock input model for d",
            "output": "mock output model for d",
            "callable": "<callable for d>",
        },
    }
    with patch("openbb_agents.tools._get_openbb_coverage_command_schemas") as mock:
        mock.return_value = mock_coverage_command_schema_dict
        yield mock


@pytest.fixture
def openbb_tool_vector_index() -> VectorStore:
    return build_openbb_tool_vector_index()
