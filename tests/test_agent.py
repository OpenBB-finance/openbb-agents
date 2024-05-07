import pytest

from openbb_agents.agent import aopenbb_agent, openbb_agent
from openbb_agents.testing import with_llm


def test_openbb_agent(openbb_tool_vector_index):
    test_query = "What is the stock price of AAPL and MSFT?"
    actual_result = openbb_agent(
        query=test_query,
        openbb_tools=[".equity.price.quote", ".equity.fundamental.metrics"],
    )
    assert isinstance(actual_result, str)
    assert with_llm(actual_result, "MSFT's stock price is in the model output.")
    assert with_llm(actual_result, "AAPL's stock price is in the model output.")
    assert with_llm(actual_result, "One of the stock prices is higher than the other.")


@pytest.mark.asyncio
async def test_aopenbb_agent(openbb_tool_vector_index):
    test_query = "What is the stock price of AAPL and MSFT? Which is higher?"
    actual_result = await aopenbb_agent(
        query=test_query,
        openbb_tools=[".equity.price.quote", ".equity.fundamental.metrics"],
    )
    assert isinstance(actual_result, str)
    assert with_llm(actual_result, "MSFT's stock price is in the model output.")
    assert with_llm(actual_result, "AAPL's stock price is in the model output.")
    assert with_llm(actual_result, "One of the stock prices is higher than the other.")
