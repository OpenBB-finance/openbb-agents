import logging

from openbb_agents.agents.utils import make_react_agent
from openbb_agents.models import AnsweredSubQuestion, SubQuestionAgentConfig
from openbb_agents.agents.prompts import (
    FINAL_RESPONSE_PROMPT_TEMPLATE,
    SUBQUESTION_ANSWER_PROMPT,
)

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)


def _render_subquestions_and_answers(
    answered_subquestions: list[AnsweredSubQuestion],
) -> str:
    "Combines all subquestions and their answers"
    output = ""
    for answered_subq in answered_subquestions:
        output += "Subquestion: " + answered_subq.subquestion.question + "\n"
        output += "Observations: \n" + answered_subq.answer + "\n\n"

    return output


def generate_final_response(
    query: str, answered_subquestions: list[AnsweredSubQuestion]
) -> str:
    """Generate the final response to a query given answer to a list of subquestions."""

    logger.info(
        "Request to generate final response.",
        extra={
            "query": query,
            "answered_subquestions": [
                {
                    "subquestion": subq_and_a.subquestion.question,
                    "answer": subq_and_a.answer,
                }
                for subq_and_a in answered_subquestions
            ],
        },
    )

    system_message = FINAL_RESPONSE_PROMPT_TEMPLATE
    prompt = ChatPromptTemplate.from_messages([("system", system_message)])

    llm = ChatOpenAI(
        model="gpt-4", temperature=0.1
    )  # Let's use the big model for the final answer.

    chain = (
        {
            "input": lambda x: x["input"],
            "subquestions": lambda x: _render_subquestions_and_answers(
                x["answered_subquestions"]
            ),
        }
        | prompt
        | llm
    )

    result = chain.invoke(
        {"input": query, "answered_subquestions": answered_subquestions}
    )
    return str(result.content)


def generate_subquestion_answer(
    subquestion_agent_config: SubQuestionAgentConfig,
) -> AnsweredSubQuestion:
    """Generate an answer to a subquestion, using tools and dependencies as necessary."""

    logger.info(
        "Request to generate answer for subquestion.",
        extra={
            "subquestion": subquestion_agent_config.subquestion.question,
            "dependencies": [
                {
                    "subquestion": subq_and_a.subquestion.question,
                    "answer": subq_and_a.answer,
                }
                for subq_and_a in subquestion_agent_config.dependencies
            ],
            "tools": [tool.name for tool in subquestion_agent_config.tools],
        },
    )

    # Format the dependency strings
    dependencies_str = ""
    for answered_subquestion in subquestion_agent_config.dependencies:
        dependencies_str += (
            "subquestion: " + answered_subquestion.subquestion.question + "\n"
        )
        dependencies_str += "observations:\n" + answered_subquestion.answer + "\n\n"

    prompt = SUBQUESTION_ANSWER_PROMPT.format(
        query=subquestion_agent_config.query,
        subquestion_query=subquestion_agent_config.subquestion.question,
        dependencies=dependencies_str,
    )

    try:
        result = make_react_agent(tools=subquestion_agent_config.tools).invoke(
            {"input": prompt}
        )
        output = str(result["output"])
    except Exception as err:  # Terrible practice, but it'll do for now.
        print(err)
        # We'll include the error message in the future
        output = "I was unable to answer the subquestion using the available tools."

    answered_subquestion = AnsweredSubQuestion(
        subquestion=subquestion_agent_config.subquestion, answer=output
    )

    logger.info(
        "Answered subquestion.",
        extra={
            "subquestion": answered_subquestion.subquestion.question,
            "answer": answered_subquestion.answer,
        },
    )

    return answered_subquestion
