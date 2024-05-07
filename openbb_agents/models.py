from typing import Any

from pydantic import BaseModel, Field


class SubQuestion(BaseModel):
    id: int = Field(description="The unique ID of the subquestion.")
    question: str = Field(description="The subquestion itself.")
    depends_on: list[int] | None = Field(
        description="The list of subquestion ids whose answer is required to answer this subquestion.",  # noqa: E501
        default=None,
    )


class AnsweredSubQuestion(BaseModel):
    subquestion: SubQuestion = Field(
        description="The subquestion that has been answered."
    )
    answer: str = Field(description="The answer to the subquestion.")


class OpenBBFunctionDescription(BaseModel):
    name: str
    input_model: Any
    output_model: Any
    callable: Any
