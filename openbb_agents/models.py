from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import StructuredTool


class SubQuestion(BaseModel):
    id: int = Field(description="The unique ID of the subquestion.")
    question: str = Field(description="The subquestion itself.")
    depends_on: list[int] = Field(
        description="The list of subquestion ids whose answer is required to answer this subquestion.",  # noqa: E501
        default=[],
    )


class SelectedTool(BaseModel):
    name: str = Field(description="The name of the tool.")


class SelectedToolsList(BaseModel):
    tools: list[SelectedTool] = Field(
        description="A list of SelectedTool objects chosen by an agent."
    )


class SubQuestionList(BaseModel):
    subquestions: list[SubQuestion] = Field(
        description="The list of SubQuestion objects."
    )


class AnsweredSubQuestion(BaseModel):
    subquestion: SubQuestion = Field(
        description="The subquestion that has been answered."
    )
    answer: str = Field(description="The answer to the subquestion.")


class SubQuestionAgentConfig(BaseModel):
    query: str = Field(description="The top-level query to be answered.")
    subquestion: SubQuestion = Field(
        description="The specific subquestion to be answered by the agent."
    )
    tools: list[StructuredTool] = Field(
        description="A list of langchain StructuredTools for the agent to use."
    )
    dependencies: list[AnsweredSubQuestion] = Field(
        description="A list of previously-answered subquestions required by the agent to answer the question.",  # noqa: E501
        default=[],
    )
