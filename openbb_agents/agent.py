import logging
from typing import Optional

from langchain import hub
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import (
    format_log_to_str,
    format_to_openai_function_messages,
)
from langchain.agents.output_parsers import (
    JSONAgentOutputParser,
    OpenAIFunctionsAgentOutputParser,
)
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain.tools import StructuredTool
from langchain.tools.render import (
    format_tool_to_openai_function,
    render_text_description_and_args,
)
from langchain.vectorstores import FAISS, VectorStore

from openbb_agents.chains import (
    generate_final_response,
    generate_subquestion_answer,
    generate_subquestions,
    select_tools,
)
from openbb_agents.models import SubQuestionAgentConfig
from openbb_agents.tools import (
    get_all_openbb_tools,
    map_openbb_routes_to_langchain_tools,
)
from openbb_agents.utils import get_dependencies

logger = logging.getLogger(__name__)


def openbb_agent(query: str, openbb_tools: Optional[list[str]] = None) -> str:
    """Answer a query using the OpenBB Agent equipped with tools.

    By default all available openbb tools are used. You can have a query
    answered using a smaller subset of OpenBB tools by using the `openbb_tools`
    argument.

    Parameters
    ----------
    query : str
        The query you want to have answered.
    openbb_tools : optional[list[str]]
        Optional. Specify the OpenBB collections or commands that you use to use. If not
        specified, every available OpenBB tool will be used.

    Examples
    --------
    >>> # Use all OpenBB tools to answer the query
    >>> openbb_agent("What is the market cap of TSLA?")
    >>> # Use only the specified tools to answer the query
    >>> openbb_agent("What is the market cap of TSLA?",
    ...              openbb_tools=["/equity/fundamental", "/equity/price/historical"])

    """

    subquestion_list = generate_subquestions(query)
    logger.info("Generated subquestions: %s", subquestion_list)

    if openbb_tools:
        tools = map_openbb_routes_to_langchain_tools(openbb_tools)
    else:
        tools = get_all_openbb_tools()
    vector_index = _create_tool_index(tools=tools)

    answered_subquestions = []
    for subquestion in subquestion_list.subquestions:  # TODO: Do in parallel
        # Fetch tool for subquestion
        print(f"Attempting to select tools for: {subquestion.question}")
        selected_tools = select_tools(
            vector_index=vector_index,
            tools=tools,
            subquestion=subquestion,
            answered_subquestions=answered_subquestions,
        )
        # TODO: Improve filtering of tools (probably by storing them in a dict)
        tool_names = [tool.name for tool in selected_tools.tools]
        subquestion_tools = [tool for tool in tools if tool.name in tool_names]
        print(f"Retrieved tool(s): {tool_names}")

        # Then attempt to answer subquestion
        print(f"Attempting to answer question: {subquestion.question}")
        answered_subquestion = generate_subquestion_answer(
            SubQuestionAgentConfig(
                query=query,
                subquestion=subquestion,
                tools=subquestion_tools,
                dependencies=get_dependencies(
                    answered_subquestions, subquestion
                ),  # TODO: Just do this in gneerate_subquestion_answer
            )
        )
        answered_subquestions.append(answered_subquestion)
        print(answered_subquestion)

    # Answer final question
    return generate_final_response(
        query=query, answered_subquestions=answered_subquestions
    )


def make_openai_agent(prompt, tools, model="gpt-4-1106-preview", verbose=False):
    """Create a new OpenAI agent from a list of tools."""
    llm = ChatOpenAI(model=model)
    llm_with_tools = llm.bind(
        functions=[format_tool_to_openai_function(t) for t in tools]
    )
    chain = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_to_openai_function_messages(
                x["intermediate_steps"]
            ),
        }
        | prompt
        | llm_with_tools
        | OpenAIFunctionsAgentOutputParser()
    )

    return AgentExecutor(agent=chain, tools=tools, verbose=verbose)


def make_react_agent(tools, model="gpt-4-1106-preview", temperature=0.2, verbose=True):
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

    llm = ChatOpenAI(model=model, temperature=temperature).bind(stop=["\nObservation"])

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
        verbose=verbose,
        return_intermediate_steps=False,
        handle_parsing_errors=True,
    )
    return agent_executor


def _create_tool_index(tools: list[StructuredTool]) -> VectorStore:
    """Create a tool index of LangChain StructuredTools."""
    docs = [
        Document(page_content=t.description, metadata={"index": i})
        for i, t in enumerate(tools)
    ]

    vector_store = FAISS.from_documents(docs, OpenAIEmbeddings())
    return vector_store
