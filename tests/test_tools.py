from pydantic import BaseModel, Field

from openbb_agents.models import OpenBBFunctionDescription
from openbb_agents.tools import (
    _get_flat_properties_from_pydantic_model_as_str,
    append_tools_to_vector_index,
    build_vector_index_from_openbb_function_descriptions,
    get_valid_list_of_providers,
    get_valid_openbb_function_descriptions,
    get_valid_openbb_function_names,
    make_vector_index_description,
)


def test_get_valid_list_of_providers(mock_obb_user_credentials):
    actual_result = get_valid_list_of_providers()
    expected_result = ["yfinance", "fmp", "intrinio"]
    assert actual_result == expected_result


def test_get_valid_openbb_function_names(mock_obb_coverage_providers):
    actual_result = get_valid_openbb_function_names()
    expected_result = ["function_a", "function_b", "function_c"]
    assert actual_result == expected_result


def test_get_valid_openbb_function_descriptions(mock_obb_coverage_command_schema):
    actual_result = get_valid_openbb_function_descriptions()
    expected_result = [
        OpenBBFunctionDescription(
            name="function_a",
            input_model="mock input model for a",
            output_model="mock output model for a",
            callable="<callable for a>",
        ),
        OpenBBFunctionDescription(
            name="function_b",
            input_model="mock input model for b",
            output_model="mock output model for b",
            callable="<callable for b>",
        ),
        OpenBBFunctionDescription(
            name="function_c",
            input_model="mock input model for c",
            output_model="mock output model for c",
            callable="<callable for c>",
        ),
    ]
    assert actual_result == expected_result


def test_get_flat_properties_from_pydantic_model_as_str():
    class TestModel(BaseModel):
        first_property: str = Field(description="The first property")
        second_property: int = Field(description="The second property")
        third_property: list[float] = Field(description="The third property")

    actual_result = _get_flat_properties_from_pydantic_model_as_str(model=TestModel)
    expected_result = """\
first_property: The first property
second_property: The second property
third_property: The third property
"""
    assert actual_result == expected_result


def test_make_vector_index_description(
    mock_openbb_function_callable, mock_openbb_function_output_model
):
    test_obb_function_description = OpenBBFunctionDescription(
        name="Test Function",
        input_model="<input model>",
        output_model=mock_openbb_function_output_model,
        callable=mock_openbb_function_callable,
    )

    actual_result = make_vector_index_description(
        openbb_function_description=test_obb_function_description
    )
    assert (
        "A callable test function that does nothing, but has a docstring."
        in actual_result
    )
    assert "first_property: The first property" in actual_result
    assert "second_property: The second property" in actual_result
    assert "third_property: The third property" in actual_result


def test_build_vector_index(
    mock_openbb_function_output_model, mock_openbb_function_callable
):
    test_openbb_function_descriptions = [
        OpenBBFunctionDescription(
            name="function_a",
            input_model="mock input model for a",
            output_model=mock_openbb_function_output_model,
            callable=mock_openbb_function_callable,
        ),
        OpenBBFunctionDescription(
            name="function_b",
            input_model="mock input model for b",
            output_model=mock_openbb_function_output_model,
            callable=mock_openbb_function_callable,
        ),
    ]

    actual_result = build_vector_index_from_openbb_function_descriptions(
        openbb_function_descriptions=test_openbb_function_descriptions
    )
    assert len(actual_result.docstore._dict) == 2  # type: ignore


def test_append_tools_to_vector_index(openbb_tool_vector_index):
    def test_tool() -> str:
        """This is a test tool."""
        return "test"

    actual_result = append_tools_to_vector_index(
        vector_store=openbb_tool_vector_index,
        tools=[test_tool],
    )

    all_tools = [doc for doc in actual_result.docstore._dict.values()]

    inserted_tool = None
    for tool in all_tools:
        if tool.metadata["tool_name"] == "test_tool":
            inserted_tool = tool
            break

    assert inserted_tool is not None
    assert inserted_tool.page_content == "test_tool\nThis is a test tool.\nOutput:\nstr"
    assert inserted_tool.metadata["callable"].__name__ == test_tool.__name__


def test_append_tools_to_vector_index_pydantic_return_type(openbb_tool_vector_index):
    class TestReturnModel(BaseModel):
        price: float = Field(description="The price of the stock")
        volume: int = Field(description="The volume of the stock")

    def test_tool() -> TestReturnModel:
        """This is a test tool."""
        return TestReturnModel(price=100, volume=1000)

    actual_result = append_tools_to_vector_index(
        vector_store=openbb_tool_vector_index,
        tools=[test_tool],
    )

    all_tools = [doc for doc in actual_result.docstore._dict.values()]

    inserted_tool = None
    for tool in all_tools:
        if tool.metadata["tool_name"] == "test_tool":
            inserted_tool = tool
            break

    assert inserted_tool is not None
    assert (
        inserted_tool.page_content
        == "test_tool\nThis is a test tool.\nOutput:\nprice: The price of the stock\nvolume: The volume of the stock\n"  # noqa:E501
    )  # noqa: E501
    assert inserted_tool.metadata["callable"].__name__ == test_tool.__name__


def test_append_tools_to_vector_index_duplicate_input(openbb_tool_vector_index):
    def test_tool() -> str:
        """This is a test tool."""
        return "test"

    vector_store = append_tools_to_vector_index(
        vector_store=openbb_tool_vector_index,
        tools=[test_tool],
    )

    all_tools = [doc for doc in vector_store.docstore._dict.values()]

    inserted_tool = None
    for tool in all_tools:
        if tool.metadata["tool_name"] == "test_tool":
            inserted_tool = tool
            break

    assert inserted_tool is not None

    # Attempt to insert the same tool again
    new_vector_store = append_tools_to_vector_index(
        vector_store=vector_store,
        tools=[test_tool],
    )

    assert len(new_vector_store.docstore._dict) == len(vector_store.docstore._dict)
