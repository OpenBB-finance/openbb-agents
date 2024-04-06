import logging
from typing import Optional

from openbb import obb

from openbb_agents.chains import (
    generate_final_response,
    generate_subquestion_answer,
    generate_subquestions,
    select_tools,
)
from openbb_agents.models import SubQuestionAgentConfig
from openbb_agents.tools import (
    create_tool_index,
    get_all_openbb_tools,
    map_openbb_routes_to_langchain_tools,
)
from openbb_agents.utils import get_dependencies

from . import VERBOSE

logger = logging.getLogger(__name__)


class OpenBBAgent:
    def __init__(self, personal_access_token: str, verbose=VERBOSE):
        # Login to OpenBB using person access token
        obb.account.login(pat=personal_access_token)
        self.verbose = verbose

    def answer_query(self, query: str, openbb_tools: Optional[list[str]] = None) -> str:
        """Answer a query using the OpenBB Agent equipped with tools and a personal access token.

        By default all available openbb tools are used. You can have a query
        answered using a smaller subset of OpenBB tools by using the `openbb_tools`
        argument.

        Parameters
        ----------
        query : str
            The query you want to have answered.
        openbb_tools : optional[list[str]]
            Optional. Specify the OpenBB collections or commands that you want to use. If not
            specified, every available OpenBB tool will be used.

        Examples
        --------
        >>> agent = OpenBBAgent(personal_access_token="your_token_here")
        >>> # Use all OpenBB tools to answer the query
        >>> agent.answer_query("What is the market cap of TSLA?")
        >>> # Use only the specified tools to answer the query
        >>> agent.answer_query("What is the market cap of TSLA?",
        ...                    openbb_tools=["/equity/fundamental", "/equity/price/historical"])

        """

        subquestion_list = generate_subquestions(query, verbose=self.verbose)
        logger.info("Generated subquestions: %s", subquestion_list)

        if openbb_tools:
            tools = map_openbb_routes_to_langchain_tools(openbb_tools)
        else:
            tools = get_all_openbb_tools()
        vector_index = create_tool_index(tools=tools)

        answered_subquestions = []
        for subquestion in subquestion_list.subquestions:  # TODO: Do in parallel
            # Fetch tool for subquestion
            logger.info("Attempting to select tools for: %s", {subquestion.question})
            selected_tools = select_tools(
                vector_index=vector_index,
                tools=tools,
                subquestion=subquestion,
                answered_subquestions=answered_subquestions,
                verbose=self.verbose,
            )
            # TODO: Improve filtering of tools (probably by storing them in a dict)
            tool_names = [tool.name for tool in selected_tools.tools]
            subquestion_tools = [tool for tool in tools if tool.name in tool_names]
            logger.info("Retrieved tool(s): %s", tool_names)

            # Then attempt to answer subquestion
            answered_subquestion = generate_subquestion_answer(
                SubQuestionAgentConfig(
                    query=query,
                    subquestion=subquestion,
                    tools=subquestion_tools,
                    dependencies=get_dependencies(
                        answered_subquestions, subquestion
                    ),  # TODO: Just do this in generate_subquestion_answer
                ),
                verbose=self.verbose,
            )
            answered_subquestions.append(answered_subquestion)

        # Answer final question
        return generate_final_response(
            query=query,
            answered_subquestions=answered_subquestions,
            verbose=self.verbose,
        )
