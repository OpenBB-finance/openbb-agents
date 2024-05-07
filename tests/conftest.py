from unittest.mock import patch

import pytest
from openbb import obb
from pydantic import BaseModel


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
    monkeypatch.setattr(obb.user, "credentials", mock_credentials)


@pytest.fixture
def mock_obb_coverage_providers(mock_obb_user_credentials):
    mock_provider_coverage_dict = {
        "fmp": ["function_a", "function_b"],
        "intrinio": ["function_a", "function_c"],
        "benzinga": ["function_d"],
    }
    with patch("openbb_agents.tools.get_openbb_coverage_providers") as mock:
        mock.return_value = mock_provider_coverage_dict
        yield mock


@pytest.fixture
def mock_obb_coverage_command_schema(mock_obb_coverage_providers):
    mock_coverage_command_schema_dict = {
        "function_a": {
            "input": "mock input model for a",
            "output": "mock output model for a",
            "callable": "mock callable for a",
        },
        "function_b": {
            "input": "mock input model for b",
            "output": "mock output model for b",
            "callable": "mock callable for b",
        },
        "function_c": {
            "input": "mock input model for c",
            "output": "mock output model for c",
            "callable": "mock callable for c",
        },
        "function_d": {
            "input": "mock input model for d",
            "output": "mock output model for d",
            "callable": "mock callable for d",
        },
    }
    with patch("openbb_agents.tools.get_openbb_coverage_command_schemas") as mock:
        mock.return_value = mock_coverage_command_schema_dict
        yield mock
