from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate
from openbb_agents.agents.chains import (
    generate_final_response,
    generate_subquestion_answer,
)
from openbb_agents.agents.utils import _create_tool_index, _get_dependencies, _get_tools
from openbb_agents.models import (
    AnsweredSubQuestion,
    SubQuestionAgentConfig,
    SubQuestionList,
)
from langchain.tools import StructuredTool
import logging
from openbb_agents.agents.prompts import SUBQUESTION_GENERATOR_PROMPT
from openbb_agents.utils import get_all_openbb_tools

logger = logging.getLogger(__name__)


def generate_subquestions(query: str, model="gpt-4"):
    """Generate subquestions from a query."""

    subquestion_parser = PydanticOutputParser(pydantic_object=SubQuestionList)

    system_message = SUBQUESTION_GENERATOR_PROMPT
    human_message = """\
        ## User Question
        {input}
        """

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            ("human", human_message),
        ]
    )
    prompt = prompt.partial(
        format_instructions=subquestion_parser.get_format_instructions()
    )

    llm = ChatOpenAI(model=model, temperature=0.1)
    subquestion_chain = (
        {"input": lambda x: x["input"]} | prompt | llm | subquestion_parser
    )
    subquestion_list = subquestion_chain.invoke({"input": query})

    return subquestion_list


def openbb_agent_from_tools(
    openbb_tools: list[StructuredTool],
    query: str,
):
    logger.info("Creating vector db of %i tools.", len(openbb_tools))
    vector_index = _create_tool_index(tools=openbb_tools)

    logger.info("Generating subquestions for query: %s", query)
    subquestions = generate_subquestions(query)
    logger.info(
        "Generated %i subquestions.",
        len(subquestions.subquestions),
        extra={"subquestions": subquestions},
    )

    answered_subquestions: list[AnsweredSubQuestion] = []
    for subq in subquestions.subquestions:
        subquestion_agent_config = SubQuestionAgentConfig(
            query=query,
            subquestion=subq,
            tools=_get_tools(
                vector_index=vector_index, tools=openbb_tools, query=subq.tool_query
            ),
            dependencies=_get_dependencies(
                answered_subquestions=answered_subquestions, subquestion=subq
            ),
        )

        # Answer the subquestion
        answered_subquestion = generate_subquestion_answer(subquestion_agent_config)
        logging.info(
            "Answered Subquestion.",
            extra={
                "subquestion": answered_subquestion.subquestion.question,
                "answer": answered_subquestion.answer,
            },
        )

        answered_subquestions.append(answered_subquestion)

    result = generate_final_response(
        query=query, answered_subquestions=answered_subquestions
    )

    logger.info("Final Answer: %s", result)
    return result


def openbb_agent(query: str):
    tools = get_all_openbb_tools()
    return openbb_agent_from_tools(query=query, openbb_tools=tools)
