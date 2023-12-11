import logging
import langchain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.agents.output_parsers import JSONAgentOutputParser
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents import AgentExecutor
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain.vectorstores import FAISS, VectorStore
from langchain import hub
from langchain.tools.render import render_text_description_and_args
from langchain.tools import StructuredTool
from langchain.output_parsers import PydanticOutputParser

from typing import Optional

from openbb import obb

from .utils import map_openbb_collection_to_langchain_tools, get_all_openbb_tools
from .models import (
    AnsweredSubQuestion,
    SubQuestionList,
    SubQuestion,
    SubQuestionAgentConfig,
)
from .prompts import (
    SUBQUESTION_GENERATOR_PROMPT,
    FINAL_RESPONSE_PROMPT_TEMPLATE,
    SUBQUESTION_ANSWER_PROMPT,
)

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


def make_react_agent(tools, model="gpt-4-1106-preview"):
    """Create a new ReAct agent from a list of tools."""

    # This retrieves the ReAct agent chat prompt template available in Langchain Hub
    # https://smith.langchain.com/hub/hwchase17/react-json?organizationId=10beea65-e722-5aa1-9f93-034c22e3cd6e
    prompt = hub.pull("hwchase17/react-multi-input-json")

    # Replace the 'tools' and 'tool_names' content of the prompt with
    # information given to the agent Note that tool_names is a field available
    # in each tool, so it can be inferred from same argument
    prompt = prompt.partial(
        tools=render_text_description_and_args(tools),
        tool_names=", ".join([t.name for t in tools]),
    )

    llm = ChatOpenAI(model=model, temperature=0.0).bind(stop=["\nObservation"])

    chain = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_log_to_str(x["intermediate_steps"]),
        }
        | prompt
        | llm
        | JSONAgentOutputParser()
    )

    agent_executor = AgentExecutor(
        agent=chain,
        tools=tools,
        verbose=False,
        return_intermediate_steps=False,
        handle_parsing_errors=True,
    )
    return agent_executor


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


def _render_subquestions_and_answers(
    answered_subquestions: list[AnsweredSubQuestion],
) -> str:
    "Combines all subquestions and their answers"
    output = ""
    for answered_subq in answered_subquestions:
        output += "Subquestion: " + answered_subq.subquestion.question + "\n"
        output += "Observations: \n" + answered_subq.answer + "\n\n"

    return output


def _create_tool_index(tools: list[StructuredTool]) -> VectorStore:
    """Create a tool index of LangChain StructuredTools."""
    docs = [
        Document(page_content=t.description, metadata={"index": i})
        for i, t in enumerate(tools)
    ]

    vector_store = FAISS.from_documents(docs, OpenAIEmbeddings())
    return vector_store


def _get_tools(
    vector_index: VectorStore, tools: list[StructuredTool], query: str
) -> list[StructuredTool]:
    """Retrieve tools from a vector index given a query."""
    retriever = vector_index.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.65},
    )
    docs = retriever.get_relevant_documents(query)

    # This is a fallback mechanism in case the threshold is too high,
    # causing too few tools to be returned.  In this case, we fall back to
    # getting the top k=2 results with higher similarity scores.
    if len(docs) < 4:
        retriever = vector_index.as_retriever(search_kwargs={"k": 2})
        docs = retriever.get_relevant_documents(query)

    tools = [tools[d.metadata["index"]] for d in docs]
    return tools


def _get_dependencies(
    answered_subquestions: list[AnsweredSubQuestion], subquestion: SubQuestion
) -> list[AnsweredSubQuestion]:
    dependency_subquestions = [
        answered_subq
        for answered_subq in answered_subquestions
        if answered_subq.subquestion.id in subquestion.depends_on
    ]
    return dependency_subquestions
