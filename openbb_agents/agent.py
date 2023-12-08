import logging
import langchain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.output_parsers import (
    JSONAgentOutputParser,
    OpenAIFunctionsAgentOutputParser,
)
from langchain.agents.format_scratchpad import (
    format_log_to_str,
    format_to_openai_function_messages,
)
from langchain.agents import AgentExecutor, AgentType, initialize_agent
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain.vectorstores import FAISS, VectorStore
from langchain import hub
from langchain.tools.render import (
    format_tool_to_openai_function,
    render_text_description_and_args,
)
from langchain.tools import StructuredTool
from langchain.output_parsers import PydanticOutputParser

from typing import Optional

from openbb import obb

from .utils import map_openbb_collection_to_langchain_tools, get_all_openbb_tools
from .models import (
    AnsweredSubQuestion,
    SelectedToolsList,
    SubQuestionList,
    SubQuestion,
    SubQuestionAgentConfig,
)
from .prompts import (
    SUBQUESTION_GENERATOR_PROMPT,
    SUBQUESTION_GENERATOR_PROMPT_V2,
    FINAL_RESPONSE_PROMPT_TEMPLATE,
    SUBQUESTION_ANSWER_PROMPT,
    TOOL_SEARCH_PROMPT,
)

logger = logging.getLogger(__name__)


def select_tools(
    vector_index: VectorStore,
    tools: list[StructuredTool],
    subquestion: SubQuestion,
    answered_subquestions: list[AnsweredSubQuestion],
) -> SelectedToolsList:
    """Use an agent to select which tools to use given a subquestion and its dependencies."""

    # Here we define the tool the agent will use to search the tool index.
    def search_tools(query: str) -> list[tuple[str, str]]:
        """Search a vector index for useful funancial tools."""
        returned_tools = _get_tools(
            vector_index=vector_index,
            tools=tools,
            query=query,
        )
        return [(tool.name, tool.description) for tool in returned_tools]

    dependencies = _get_dependencies(
        answered_subquestions=answered_subquestions, subquestion=subquestion
    )
    dependencies_str = _render_subquestions_and_answers(dependencies)

    selected_tools_list_parser = PydanticOutputParser(pydantic_object=SelectedToolsList)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", TOOL_SEARCH_PROMPT),
            ("human", "## User Question:\n{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    prompt = prompt.partial(
        format_instructions=selected_tools_list_parser.get_format_instructions(),
        subquestions=dependencies_str,
    )

    search_tool = StructuredTool.from_function(search_tools)
    agent = make_openai_agent(prompt=prompt, tools=[search_tool])
    result = agent.invoke({"input": subquestion.question})

    # Parse the output into a pydantic model and return
    selected_tools = selected_tools_list_parser.parse(result["output"])
    return selected_tools


def generate_subquestions_v2(query: str) -> SubQuestionList:
    subquestion_parser = PydanticOutputParser(pydantic_object=SubQuestionList)

    system_message = SUBQUESTION_GENERATOR_PROMPT_V2
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

    llm = ChatOpenAI(model="gpt-4", temperature=0.0)
    subquestion_chain = (
        {"input": lambda x: x["input"]} | prompt | llm | subquestion_parser
    )
    subquestion_list = subquestion_chain.invoke({"input": query})

    return subquestion_list


def openbb_agent_v2(query: str):
    print("Generate subquestions...")
    subquestion_list = generate_subquestions_v2(query)
    print(subquestion_list)

    openbb_tools = get_all_openbb_tools()
    vector_index = _create_tool_index(tools=openbb_tools)

    answered_subquestions = []
    for subquestion in subquestion_list.subquestions:  # TODO: Do in parallel
        # Fetch tool for subquestion
        print(f"Attempting to select tools for: {subquestion.question}")
        selected_tools = select_tools(
            vector_index=vector_index,
            tools=openbb_tools,
            subquestion=subquestion,
            answered_subquestions=answered_subquestions,
        )
        # TODO: Improve filtering of tools (probably by storing them in a dict)
        tool_names = [tool.name for tool in selected_tools.tools]
        subquestion_tools = [tool for tool in openbb_tools if tool.name in tool_names]
        print(f"Retrieved tool(s): {tool_names}")

        # Then attempt to answer subquestion
        print(f"Attempting to answer question: {subquestion.question}")
        answered_subquestion = generate_subquestion_answer(
            SubQuestionAgentConfig(
                query=query,
                subquestion=subquestion,
                tools=subquestion_tools,
                dependencies=_get_dependencies(
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
