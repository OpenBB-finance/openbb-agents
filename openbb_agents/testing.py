from magentic import OpenaiChatModel, prompt
from pydantic import BaseModel, Field


class AssertResult(BaseModel):
    assessment: str = Field(
        description="Your assessment of whether the assertion is true or false."
    )
    result: bool = Field(description="The final assertion result.")


def with_llm(model_output, assertion) -> bool:
    """Use an LLM to assert a result.

    Works best for short, simple inputs and asserts.

    This is useful for unstructured outputs that cannot be easily or
    deterministically parsed.  Just keep in mind, it remains an LLM evaluator
    under-the-hood, so it's not as fast as a direct assertion, costs money to
    use, and may be less accurate and reliable (especially for longer inputs or
    complicated asserts).

    Examples
    --------
    >>> assert with_llm(model_output="I could not retrieve the stock price for apple", assertion="the stock price for apple was retrieved successfully")
    AssertionError: The stock price for Apple was not retrieved successfully.
    """  # noqa: E501

    @prompt(
        "Given the following model output: {model_output}, determine if the following assertion is true: {assertion}",  # noqa: E501
        model=OpenaiChatModel(
            model="gpt-3.5-turbo",
            temperature=0.0,
        ),
    )
    def _llm_assert(model_output: str, assertion: str) -> AssertResult:
        ...

    result = _llm_assert(model_output, assertion)
    assert result.result, f"Assertion '{assertion}' for output '{model_output}' failed: {result.assessment}"  # noqa: E501
    return result.result
