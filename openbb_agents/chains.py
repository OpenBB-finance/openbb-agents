import logging

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
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import StructuredTool
from langchain.tools.render import (
    format_tool_to_openai_function,
    render_text_description_and_args,
)
from langchain.vectorstores import VectorStore

from openbb_agents.models import (
    AnsweredSubQuestion,
    SelectedToolsList,
    SubQuestion,
    SubQuestionAgentConfig,
    SubQuestionList,
)
from openbb_agents.prompts import (
    FINAL_RESPONSE_PROMPT_TEMPLATE,
    SUBQUESTION_ANSWER_PROMPT,
    SUBQUESTION_GENERATOR_PROMPT,
    TOOL_SEARCH_PROMPT,
)
from openbb_agents.utils import get_dependencies

from . import VERBOSE

logger = logging.getLogger(__name__)


def generate_final_response(
    query: str,
    answered_subquestions: list[AnsweredSubQuestion],
    verbose=VERBOSE,
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

    llm = ChatOpenAI(model="gpt-4", temperature=0.1, verbose=verbose)

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
        {"input": query, "answered_subquestions": answered_subquestions},
    )
    return str(result.content)


def generate_subquestion_answer(
    subquestion_agent_config: SubQuestionAgentConfig, verbose=VERBOSE
) -> AnsweredSubQuestion:
    """Generate an answer to a subquestion using tools and dependencies."""

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
        result = make_react_agent(
            tools=subquestion_agent_config.tools, verbose=verbose
        ).invoke({"input": prompt})
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


def select_tools(
    vector_index: VectorStore,
    tools: list[StructuredTool],
    subquestion: SubQuestion,
    answered_subquestions: list[AnsweredSubQuestion],
    verbose: bool = VERBOSE,
) -> SelectedToolsList:
    """Use an agent to select tools given a subquestion and its dependencies."""

    # Here we define the tool the agent will use to search the tool index.
    def search_tools(query: str) -> list[tuple[str, str]]:
        """Search a vector index for useful funancial tools."""
        returned_tools = _get_tools(
            vector_index=vector_index,
            tools=tools,
            query=query,
        )
        return [(tool.name, tool.description) for tool in returned_tools]

    dependencies = get_dependencies(
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
    agent = make_openai_agent(prompt=prompt, tools=[search_tool], verbose=verbose)
    result = agent.invoke({"input": subquestion.question})

    # Parse the output into a pydantic model and return
    selected_tools = selected_tools_list_parser.parse(result["output"])
    return selected_tools


def generate_subquestions(query: str, verbose=VERBOSE) -> SubQuestionList:
    logger.info("Request to generate subquestions for query: %s", query)
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

    llm = ChatOpenAI(model="gpt-4", temperature=0.0, verbose=verbose)
    subquestion_chain = (
        {"input": lambda x: x["input"]} | prompt | llm | subquestion_parser
    )
    subquestion_list = subquestion_chain.invoke({"input": query})

    return subquestion_list


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


def _render_subquestions_and_answers(
    answered_subquestions: list[AnsweredSubQuestion],
) -> str:
    "Combines all subquestions and their answers"
    output = ""
    for answered_subq in answered_subquestions:
        output += "Subquestion: " + answered_subq.subquestion.question + "\n"
        output += "Observations: \n" + answered_subq.answer + "\n\n"

    return output


def make_openai_agent(prompt, tools, model="gpt-4-1106-preview", verbose=VERBOSE):
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


def make_react_agent(
    tools, model="gpt-4-1106-preview", temperature=0.2, verbose=VERBOSE
):
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
