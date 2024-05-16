from typing import Literal

import pytest
from openbb import obb
from pydantic import BaseModel

from openbb_agents.chains import (
    agenerate_final_answer,
    agenerate_subquestion_answer,
    agenerate_subquestions_from_query,
    asearch_tools,
    generate_final_answer,
    generate_subquestion_answer,
    generate_subquestions_from_query,
    search_tools,
)
from openbb_agents.models import AnsweredSubQuestion, SubQuestion
from openbb_agents.testing import with_llm


def test_generate_subquestions_from_query():
    test_query = "Calculate the P/E ratio of AAPL."
    actual_result = generate_subquestions_from_query(user_query=test_query)
    assert isinstance(actual_result, list)
    assert len(actual_result) > 0
    assert isinstance(actual_result[0], SubQuestion)


@pytest.mark.asyncio
async def test_agenerate_subquestions_from_query():
    test_query = "Calculate the P/E ratio of AAPL."
    actual_result = await agenerate_subquestions_from_query(user_query=test_query)
    assert isinstance(actual_result, list)
    assert len(actual_result) > 0
    assert isinstance(actual_result[0], SubQuestion)


def test_search_tools_no_dependencies(openbb_tool_vector_index):
    test_subquestion = SubQuestion(id=1, question="What is the stock price of AAPL?")
    actual_result = search_tools(
        subquestion=test_subquestion,
        answered_subquestions=None,
        tool_vector_index=openbb_tool_vector_index,
    )

    assert len(actual_result) > 0
    assert actual_result[0].__name__ == "quote"
    assert callable(actual_result[0])


@pytest.mark.asyncio
async def test_asearch_tools_no_dependencies(openbb_tool_vector_index):
    test_subquestion = SubQuestion(id=1, question="What is the stock price of AAPL?")
    actual_result = await asearch_tools(
        subquestion=test_subquestion,
        answered_subquestions=None,
        tool_vector_index=openbb_tool_vector_index,
    )

    assert len(actual_result) > 0
    assert actual_result[0].__name__ == "quote"
    assert callable(actual_result[0])


def test_generate_subquestion_answer_no_dependencies():
    test_user_query = "What is the current stock price of AAPL?"
    test_subquestion = SubQuestion(
        id=1, question="What is the stock price of AAPL? Use yfinance as the provider."
    )
    test_tool = obb.equity.price.quote  # type: ignore
    actual_result: AnsweredSubQuestion = generate_subquestion_answer(
        user_query=test_user_query,
        subquestion=test_subquestion,
        dependencies=[],
        tools=[test_tool],
    )
    assert with_llm(
        actual_result.answer, "the stock price for apple was retrieved successfully"
    )


@pytest.mark.asyncio
async def test_agenerate_subquestion_answer_no_dependencies():
    test_user_query = "What is the current stock price of AAPL?"
    test_subquestion = SubQuestion(
        id=1, question="What is the stock price of AAPL? Use yfinance as the provider."
    )
    test_tool = obb.equity.price.quote  # type: ignore
    actual_result: AnsweredSubQuestion = await agenerate_subquestion_answer(
        user_query=test_user_query,
        subquestion=test_subquestion,
        dependencies=[],
        tools=[test_tool],
    )
    assert with_llm(
        actual_result.answer, "the stock price for apple was retrieved successfully"
    )


def test_generate_subquestion_answer_with_dependencies():
    test_user_query = "What is the current stock price of MSFT's biggest competitor?"
    test_subquestion = SubQuestion(
        id=1,
        question="What is the stock price of MSFT's biggest competitor? Use yfinance as the provider.",  # noqa: E501
        depends_on=[2],
    )
    test_dependencies = [
        AnsweredSubQuestion(
            subquestion=SubQuestion(
                id=2, question="What is the current biggest competitor to MSFT?"
            ),
            answer="The current biggest competitor to MSFT is AAPL.",
        )
    ]
    test_tool = obb.equity.price.quote  # type: ignore
    actual_result: AnsweredSubQuestion = generate_subquestion_answer(
        user_query=test_user_query,
        subquestion=test_subquestion,
        dependencies=test_dependencies,
        tools=[test_tool],
    )
    assert with_llm(
        actual_result.answer, "the stock price for apple was retrieved successfully"
    )


@pytest.mark.asyncio
async def test_agenerate_subquestion_answer_with_dependencies():
    test_user_query = "What is the current stock price of MSFT's biggest competitor?"
    test_subquestion = SubQuestion(
        id=1,
        question="What is the stock price of MSFT's biggest competitor? Use yfinance as the provider.",  # noqa: E501
        depends_on=[2],
    )
    test_dependencies = [
        AnsweredSubQuestion(
            subquestion=SubQuestion(
                id=2, question="What is the current biggest competitor to MSFT?"
            ),
            answer="The current biggest competitor to MSFT is AAPL.",
        )
    ]
    test_tool = obb.equity.price.quote  # type: ignore
    actual_result: AnsweredSubQuestion = await agenerate_subquestion_answer(
        user_query=test_user_query,
        subquestion=test_subquestion,
        dependencies=test_dependencies,
        tools=[test_tool],
    )
    assert with_llm(
        actual_result.answer, "the stock price for apple was retrieved successfully"
    )


def test_generate_subquestion_answer_with_generic_error_in_function_call():
    test_user_query = "What is the current stock price of AAPL?"
    test_subquestion = SubQuestion(id=1, question="What is the stock price of AAPL?")

    def _get_stock_price(symbol: str) -> str:
        raise ValueError("The backend is offline.")

    actual_result: AnsweredSubQuestion = generate_subquestion_answer(
        user_query=test_user_query,
        subquestion=test_subquestion,
        dependencies=[],
        tools=[_get_stock_price],
    )
    assert isinstance(actual_result, AnsweredSubQuestion)
    assert with_llm(
        actual_result.answer,
        "The backend is offline, and the answer could not be retrieved.",
    )


@pytest.mark.asyncio
async def test_agenerate_subquestion_answer_with_generic_error_in_function_call():
    test_user_query = "What is the current stock price of AAPL?"
    test_subquestion = SubQuestion(id=1, question="What is the stock price of AAPL?")

    def _get_stock_price(symbol: str) -> str:
        raise ValueError("The backend is currently offline.")

    actual_result: AnsweredSubQuestion = await agenerate_subquestion_answer(
        user_query=test_user_query,
        subquestion=test_subquestion,
        dependencies=[],
        tools=[_get_stock_price],
    )
    assert isinstance(actual_result, AnsweredSubQuestion)
    assert with_llm(
        actual_result.answer,
        "The backend is offline, and the answer could not be retrieved.",
    )


def test_generate_subquestion_answer_self_heals_with_input_validation_error_in_function_call():  # noqa: E501
    test_user_query = "What is the current stock price of AAPL? Preferably in EUR."
    test_subquestion = SubQuestion(id=1, question="What is the stock price of AAPL?")

    def _get_stock_price(symbol: str, currency: Literal["USD", "EUR"]) -> str:
        class StockPricePayload(BaseModel):
            symbol: str
            currency: Literal["USD"]  # Only USD is allowed, but we ask for EUR.

        _ = StockPricePayload(symbol=symbol, currency=currency)  # type: ignore
        return "The stock price is USD 95."

    actual_result: AnsweredSubQuestion = generate_subquestion_answer(
        user_query=test_user_query,
        subquestion=test_subquestion,
        dependencies=[],
        tools=[_get_stock_price],
    )
    assert isinstance(actual_result, AnsweredSubQuestion)
    assert with_llm(actual_result.answer, "The stock price is 95 USD.")


@pytest.mark.asyncio
async def test_agenerate_subquestion_answer_self_heals_with_input_validation_error_in_function_call():  # noqa: E501
    test_user_query = "What is the current stock price of AAPL? Preferably in EUR."
    test_subquestion = SubQuestion(id=1, question="What is the stock price of AAPL?")

    def _get_stock_price(symbol: str, currency: Literal["USD", "EUR"]) -> str:
        class StockPricePayload(BaseModel):
            symbol: str
            currency: Literal["USD"]  # Only USD is allowed, but we ask for EUR.

        _ = StockPricePayload(symbol=symbol, currency=currency)  # type: ignore
        return "The stock price is USD 95."

    actual_result: AnsweredSubQuestion = generate_subquestion_answer(
        user_query=test_user_query,
        subquestion=test_subquestion,
        dependencies=[],
        tools=[_get_stock_price],
    )
    assert isinstance(actual_result, AnsweredSubQuestion)
    assert with_llm(actual_result.answer, "The stock price is 95 USD.")


def test_generate_final_answer():
    test_user_query = "Who has the highest stock price? AMZN or TSLA?"
    test_answered_subquestions = [
        AnsweredSubQuestion(
            subquestion=SubQuestion(id=1, question="What is the stock price of AMZN?"),
            answer="The stock price of AMZN is $100.",
        ),
        AnsweredSubQuestion(
            subquestion=SubQuestion(id=2, question="What is the stock price of TSLA?"),
            answer="The stock price of TSLA is $200.",
        ),
    ]

    actual_result = generate_final_answer(
        user_query=test_user_query,
        answered_subquestions=test_answered_subquestions,
    )
    assert with_llm(actual_result, "The answer says TSLA has the highest stock price.")


@pytest.mark.asyncio
async def test_agenerate_final_answer():
    test_user_query = "Who has the highest stock price? AMZN or TSLA?"
    test_answered_subquestions = [
        AnsweredSubQuestion(
            subquestion=SubQuestion(id=1, question="What is the stock price of AMZN?"),
            answer="The stock price of AMZN is $100.",
        ),
        AnsweredSubQuestion(
            subquestion=SubQuestion(id=2, question="What is the stock price of TSLA?"),
            answer="The stock price of TSLA is $200.",
        ),
    ]

    actual_result = await agenerate_final_answer(
        user_query=test_user_query,
        answered_subquestions=test_answered_subquestions,
    )
    assert with_llm(actual_result, "The answer says TSLA has the highest stock price.")
