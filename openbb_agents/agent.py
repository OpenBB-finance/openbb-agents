import asyncio
import logging

from langchain.vectorstores import VectorStore

from .chains import (
    agenerate_subquestion_answer,
    agenerate_subquestions_from_query,
    asearch_tools,
    generate_final_answer,
    generate_subquestion_answer,
    generate_subquestions_from_query,
    search_tools,
)
from .models import AnsweredSubQuestion, SubQuestion
from .tools import (
    build_openbb_tool_vector_index,
    build_vector_index_from_openbb_function_descriptions,
    get_valid_list_of_providers,
    map_name_to_openbb_function_description,
)
from .utils import get_dependencies

logger = logging.getLogger(__name__)


def openbb_agent(
    query: str, openbb_tools: list[str] | None = None, openbb_pat: str | None = None
) -> str:
    """Answer a query using the OpenBB Agent equipped with tools.

    By default all available openbb tools are used. You can have a query
    answered using a smaller subset of OpenBB tools by using the `openbb_tools`
    argument.

    Parameters
    ----------
    query : str
        The query to be answered.
    openbb_tools : list[str] | None, optional
        A list of specific OpenBB functions to use. If not provided, all
        available OpenBB tools that you have valid credentials for will be
        utilized. See `openbb_pat`.
    openbb_pat : str | None, optional
        The OpenBB PAT for retrieving credentials from the OpenBB Hub. If not
        provided, local OpenBB credentials will be used.

    Examples
    --------
    >>> # Use all OpenBB tools to answer the query
    >>> openbb_agent("What is the stock price of TSLA?")
    >>> # Use only the specified tools to answer the query
    >>> openbb_agent("What is the stock price of TSLA?",
    ...              openbb_tools=['.equity.price.quote'])

    """
    if openbb_pat:
        from openbb import obb

        obb.account.login(pat=openbb_pat)

    tool_vector_index = _handle_tool_vector_index(openbb_tools)
    subquestions = generate_subquestions_from_query(user_query=query)

    logger.info("Generated subquestions: %s", subquestions)

    answered_subquestions = []
    for subquestion in subquestions:
        if _is_subquestion_answerable(
            subquestion=subquestion, answered_subquestions=answered_subquestions
        ):
            logger.info("Answering subquestion: %s", subquestion)
            answered_subquestion = _fetch_tools_and_answer_subquestion(
                user_query=query,
                subquestion=subquestion,
                tool_vector_index=tool_vector_index,
                answered_subquestions=answered_subquestions,
            )
            answered_subquestions.append(answered_subquestion)
        else:
            logger.info("Skipping unanswerable subquestion: %s", subquestion)
    return generate_final_answer(
        user_query=query,
        answered_subquestions=answered_subquestions,
    )


async def aopenbb_agent(query: str, openbb_tools: list[str] | None = None) -> str:
    """Answer a query using the OpenBB Agent equipped with tools.

    Async variant of `openbb_agent`.

    By default all available openbb tools are used. You can have a query
    answered using a smaller subset of OpenBB tools by using the `openbb_tools`
    argument.

    Parameters
    ----------
    query : str
        The query you want to have answered.
    openbb_tools : list[Callable]
        Optional. Specify the OpenBB functions you want to use. If not
        specified, every available OpenBB tool will be used.

    Examples
    --------
    >>> # Use all OpenBB tools to answer the query
    >>> openbb_agent("What is the stock price of TSLA?")
    >>> # Use only the specified tools to answer the query
    >>> openbb_agent("What is the stock price of TSLA?",
    ...              openbb_tools=['.equity.price.quote'])

    """
    tool_vector_index = _handle_tool_vector_index(openbb_tools)

    subquestions = await agenerate_subquestions_from_query(user_query=query)
    answered_subquestions = await _aprocess_subquestions(
        user_query=query,
        subquestions=subquestions,
        tool_vector_index=tool_vector_index,
    )

    return generate_final_answer(
        user_query=query,
        answered_subquestions=answered_subquestions,
    )


async def _aprocess_subquestions(
    user_query: str, subquestions: list[SubQuestion], tool_vector_index: VectorStore
) -> list[AnsweredSubQuestion]:
    answered_subquestions = []
    queued_subquestions = []

    tasks = []
    while True:
        unanswered_subquestions = _get_unanswered_subquestions(
            answered_subquestions=answered_subquestions, subquestions=subquestions
        )
        logger.info("Pending subquestions: %s", unanswered_subquestions)

        new_answerable_subquestions = _get_answerable_subquestions(
            subquestions=unanswered_subquestions,
            answered_subquestions=answered_subquestions,
        )
        logger.info("Answerable subquestions: %s", new_answerable_subquestions)

        for subquestion in new_answerable_subquestions:
            logger.info("Scheduling subquestion for answer: %s", subquestion)
            # Make sure we only submit newly answerable questions (since the
            # other ones have been submitted already)
            if subquestion not in queued_subquestions:
                task = asyncio.create_task(
                    _afetch_tools_and_answer_subquestion(
                        user_query=user_query,
                        subquestion=subquestion,
                        tool_vector_index=tool_vector_index,
                        answered_subquestions=answered_subquestions,
                    )
                )
                tasks.append(task)
                queued_subquestions.append(subquestion)

        if not tasks:
            break

        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        tasks = [task for task in tasks if not task.done()]

        for task in done:
            if task.exception():
                logger.error("Unexpected error in task: %s", task.exception())
            else:
                answered_subquestion = task.result()
                logger.info("Finished task for subquestion: %s", answered_subquestion)
                answered_subquestions.append(answered_subquestion)

    return answered_subquestions


def _fetch_tools_and_answer_subquestion(
    user_query: str,
    subquestion: SubQuestion,
    tool_vector_index: VectorStore,
    answered_subquestions: list[AnsweredSubQuestion],
) -> AnsweredSubQuestion:
    logger.info("Attempting to select tools for: %s", {subquestion.question})
    dependencies = get_dependencies(
        answered_subquestions=answered_subquestions, subquestion=subquestion
    )
    tools = search_tools(
        subquestion=subquestion,
        tool_vector_index=tool_vector_index,
        answered_subquestions=dependencies,
    )
    tool_names = [tool.__name__ for tool in tools]
    logger.info("Retrieved tool(s): %s", tool_names)

    # Then attempt to answer subquestion
    logger.info("Answering subquestion: %s", subquestion.question)
    answered_subquestion = generate_subquestion_answer(
        user_query=user_query,
        subquestion=subquestion,
        tools=tools,
        dependencies=dependencies,
    )

    logger.info("Answered subquestion: %s", answered_subquestion.answer)
    return answered_subquestion


async def _afetch_tools_and_answer_subquestion(
    user_query: str,
    subquestion: SubQuestion,
    tool_vector_index: VectorStore,
    answered_subquestions: list[AnsweredSubQuestion],
) -> AnsweredSubQuestion:
    logger.info("Attempting to select tools for: %s", {subquestion.question})
    dependencies = get_dependencies(
        answered_subquestions=answered_subquestions, subquestion=subquestion
    )
    tools = await asearch_tools(
        subquestion=subquestion,
        tool_vector_index=tool_vector_index,
        answered_subquestions=dependencies,
    )
    tool_names = [tool.__name__ for tool in tools]
    logger.info("Retrieved tool(s): %s", tool_names)

    # Then attempt to answer subquestion
    logger.info("Answering subquestion: %s", subquestion.question)
    answered_subquestion = await agenerate_subquestion_answer(
        user_query=user_query,
        subquestion=subquestion,
        tools=tools,
        dependencies=dependencies,
    )

    logger.info("Answered subquestion: %s", answered_subquestion.answer)
    return answered_subquestion


def _get_unanswered_subquestions(
    answered_subquestions: list[AnsweredSubQuestion], subquestions: list[SubQuestion]
) -> list[SubQuestion]:
    answered_subquestion_ids = [
        answered_subquestion.subquestion.id
        for answered_subquestion in answered_subquestions
    ]
    return [
        subquestion
        for subquestion in subquestions
        if subquestion.id not in answered_subquestion_ids
    ]


def _is_subquestion_answerable(
    subquestion: SubQuestion, answered_subquestions: list[AnsweredSubQuestion]
) -> bool:
    if not subquestion.depends_on:
        return True

    for id_ in subquestion.depends_on:
        if id_ not in [
            answered_subquestion.subquestion.id
            for answered_subquestion in answered_subquestions
        ]:
            return False
    return True


def _get_answerable_subquestions(
    subquestions: list[SubQuestion], answered_subquestions: list[AnsweredSubQuestion]
) -> list[SubQuestion]:
    return [
        subquestion
        for subquestion in subquestions
        if _is_subquestion_answerable(
            subquestion=subquestion, answered_subquestions=answered_subquestions
        )
    ]


def _handle_tool_vector_index(openbb_tools: list[str] | None) -> VectorStore:
    if not openbb_tools:
        logger.info(
            "Using all available OpenBB tools with providers: %s",
            get_valid_list_of_providers(),
        )
        tool_vector_index = build_openbb_tool_vector_index()
    else:
        logger.info("Using specified OpenBB tools: %s", openbb_tools)
        openbb_function_descriptions = [
            map_name_to_openbb_function_description(obb_function_name)
            for obb_function_name in openbb_tools
        ]
        tool_vector_index = build_vector_index_from_openbb_function_descriptions(
            openbb_function_descriptions
        )
    return tool_vector_index
