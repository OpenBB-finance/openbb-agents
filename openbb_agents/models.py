from typing import Optional
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import StructuredTool


class SubQuestion(BaseModel):
    "Pydantic data model we want each subquestion to have, including each field and what they represent"
    id: int = Field(description="The unique ID of the subquestion.")
    question: str = Field(description="The subquestion itself.")
    depends_on: list[int] = Field(
        description="The list of subquestion ids whose answer is required to answer this subquestion.",
        default=[],
    )


class SelectedTool(BaseModel):
    """A tool selected by an agent to perform a task."""

    name: str = Field(description="The name of the tool.")


class SelectedToolsList(BaseModel):
    """A list of tools selected by an agent to perform a task."""

    tools: list[SelectedTool] = Field(
        description="A list of SelectedTool objects chosen by an agent."
    )


class SubQuestionList(BaseModel):
    "Pydantic data model output we want to enforce, which is a list of the previous SubQuestion Pydantic model"
    subquestions: list[SubQuestion] = Field(
        description="The list of SubQuestion objects."
    )


class AnsweredSubQuestion(BaseModel):
    """An answered subquestion."""

    subquestion: SubQuestion = Field(
        description="The subquestion that has been answered."
    )
    answer: str = Field(description="The answer to the subquestion.")


class SubQuestionAgentConfig(BaseModel):
    """Config required to instantiate an agent, and have it answer a subquestion using tools."""

    query: str = Field(description="The top-level query to be answered.")
    subquestion: SubQuestion = Field(
        description="The specific subquestion to be answered by the agent."
    )
    tools: list[StructuredTool] = Field(
        description="A list of langchain StructuredTools for the agent to use."
    )
    dependencies: list[AnsweredSubQuestion] = Field(
        description="A list of previously-answered subquestions required by the agent to answer the question.",
        default=[],
    )
