from openbb_agents.tools import (
    OpenBBFunctionDescription,
    get_valid_list_of_providers,
    get_valid_openbb_function_descriptions,
    get_valid_openbb_function_names,
)


def test_get_valid_list_of_providers(mock_obb_user_credentials):
    actual_result = get_valid_list_of_providers()
    expected_result = ["fmp", "intrinio"]

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
            input="mock input model for a",
            output="mock output model for a",
            callable="mock callable for a",
        ),
        OpenBBFunctionDescription(
            name="function_b",
            input="mock input model for b",
            output="mock output model for b",
            callable="mock callable for b",
        ),
        OpenBBFunctionDescription(
            name="function_c",
            input="mock input model for c",
            output="mock output model for c",
            callable="mock callable for c",
        ),
    ]
    assert actual_result == expected_result
