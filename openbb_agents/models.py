from langchain.pydantic_v1 import BaseModel, Field

class SubQuestion(BaseModel):
    "Pydantic data model we want each subquestion to have, including each field and what they represent"
    id: int = Field(
        description="The unique ID of the subquestion."
    )
    question: str = Field(
        description="The subquestion itself."
    )
    query: str = Field(
        description="The query to pass to the `fetch_tools` function to retrieve the appropriate tool to answer the question."
    )
    depends_on: list[int] = Field(
        description="The list of subquestion ids whose answer is required to answer this subquestion.",
        default=[]
    )


class SubQuestionList(BaseModel):
    "Pydantic data model output we want to enforce, which is a list of the previous SubQuestion Pydantic model"
    subquestions: list[SubQuestion] = Field(
        description="The list of SubQuestion objects."
    )
