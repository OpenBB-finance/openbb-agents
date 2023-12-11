from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import StructuredTool
from langchain.vectorstores import VectorStore
from openbb_agents.agents.utils import _get_dependencies
from openbb_agents.agents.chains import (
    _render_subquestions_and_answers,
    generate_final_response,
    generate_subquestion_answer,
)
from openbb_agents.agents.utils import _create_tool_index, _get_tools, make_openai_agent
from openbb_agents.models import (
    AnsweredSubQuestion,
    SelectedToolsList,
    SubQuestion,
    SubQuestionAgentConfig,
    SubQuestionList,
)
from openbb_agents.agents.prompts import (
    SUBQUESTION_GENERATOR_PROMPT_V2,
    TOOL_SEARCH_PROMPT,
)
from openbb_agents.utils import get_all_openbb_tools


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


def openbb_agent(query: str):
    print("Generate subquestions...")
    subquestion_list = generate_subquestions_v2(query)
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
