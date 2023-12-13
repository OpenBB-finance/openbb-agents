from openbb import obb

from openbb_agents.tools import from_openbb_to_langchain_func


def test_from_openbb_to_langchain_func():
    """Test that we can create a StructuredTool from an OpenBB function."""
    test_openbb_command_root = "/equity/profile"

    test_openbb_callable = obb.equity.profile
    test_openbb_schema = obb.coverage.command_model[".equity.profile"]

    actual_result = from_openbb_to_langchain_func(
        openbb_command_root=test_openbb_command_root,
        openbb_callable=test_openbb_callable,
        openbb_schema=test_openbb_schema,
    )

    assert actual_result.name == "/equity/profile"
    assert "Equity Info" in actual_result.description  # Check for docstring
    assert "name" in actual_result.description  # Check for output field
    assert actual_result.args_schema.__name__ == "/equity/profileInputModel"
    assert actual_result.args_schema.schema() == {
        "title": "/equity/profileInputModel",
        "type": "object",
        "properties": {
            "symbol": {
                "title": "Symbol",
                "description": "Symbol to get data for.",
                "type": "string",
            }
        },
        "required": ["symbol"],
    }
