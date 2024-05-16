import logging
from datetime import datetime
from typing import Any, Callable

from langchain.vectorstores import VectorStore
from magentic import (
    AssistantMessage,
    AsyncParallelFunctionCall,
    FunctionCall,
    FunctionResultMessage,
    OpenaiChatModel,
    ParallelFunctionCall,
    SystemMessage,
    UserMessage,
    chatprompt,
    prompt,
    prompt_chain,
)

from openbb_agents.models import (
    AnsweredSubQuestion,
    SubQuestion,
)
from openbb_agents.prompts import (
    FINAL_RESPONSE_PROMPT_TEMPLATE,
    GENERATE_SUBQUESTION_SYSTEM_PROMPT_TEMPLATE,
    SUBQUESTION_ANSWER_PROMPT,
    TOOL_SEARCH_PROMPT_TEMPLATE,
)

logger = logging.getLogger(__name__)


def generate_final_answer(
    user_query: str,
    answered_subquestions: list[AnsweredSubQuestion],
) -> str:
    @prompt(
        FINAL_RESPONSE_PROMPT_TEMPLATE,
        model=OpenaiChatModel(model="gpt-4o", temperature=0.0),
    )
    def _final_answer(
        user_query: str, answered_subquestions: list[AnsweredSubQuestion]
    ) -> str:
        ...

    return _final_answer(
        user_query=user_query, answered_subquestions=answered_subquestions
    )


async def agenerate_final_answer(
    user_query: str,
    answered_subquestions: list[AnsweredSubQuestion],
) -> str:
    @prompt(
        FINAL_RESPONSE_PROMPT_TEMPLATE,
        model=OpenaiChatModel(model="gpt-4o", temperature=0.0),
    )
    async def _final_answer(
        user_query: str, answered_subquestions: list[AnsweredSubQuestion]
    ) -> str:
        ...

    return await _final_answer(
        user_query=user_query, answered_subquestions=answered_subquestions
    )


def generate_subquestion_answer(
    user_query: str,
    subquestion: SubQuestion,
    dependencies: list[AnsweredSubQuestion],
    tools: list[Callable],
) -> AnsweredSubQuestion:
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    messages: list[Any] = [SystemMessage(SUBQUESTION_ANSWER_PROMPT)]

    answer = None
    while not answer:

        @chatprompt(
            *messages,
            model=OpenaiChatModel(model="gpt-4o", temperature=0.0),
            functions=tools,
        )
        def _answer_subquestion(
            user_query: str,
            subquestion: str,
            dependencies: list[AnsweredSubQuestion],
            current_datetime: str,
        ) -> str | ParallelFunctionCall:
            ...

        response = _answer_subquestion(  # type: ignore
            user_query=user_query,
            subquestion=subquestion.question,
            dependencies=dependencies,
            current_datetime=current_datetime,
        )

        if isinstance(response, ParallelFunctionCall):
            for function_call in response._function_calls:
                logger.info(
                    "Function call: %s(%s)",
                    function_call.function.__name__,
                    function_call.arguments,
                )
                messages += _handle_function_call(function_call=function_call)
        elif isinstance(response, str):
            answer = response
    return AnsweredSubQuestion(subquestion=subquestion, answer=answer)


async def agenerate_subquestion_answer(
    user_query: str,
    subquestion: SubQuestion,
    dependencies: list[AnsweredSubQuestion],
    tools: list[Callable],
) -> AnsweredSubQuestion:
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    messages: list[Any] = [SystemMessage(SUBQUESTION_ANSWER_PROMPT)]

    answer = None
    while not answer:

        @chatprompt(
            *messages,
            model=OpenaiChatModel(model="gpt-4o", temperature=0.0),
            functions=tools,
        )
        async def _answer_subquestion(
            user_query: str,
            subquestion: str,
            dependencies: list[AnsweredSubQuestion],
            current_datetime: str,
        ) -> str | AsyncParallelFunctionCall:
            ...

        response = await _answer_subquestion(  # type: ignore
            user_query=user_query,
            subquestion=subquestion.question,
            dependencies=dependencies,
            current_datetime=current_datetime,
        )

        if isinstance(response, AsyncParallelFunctionCall):
            async for function_call in response._function_calls:
                logger.info(
                    "Function call: %s(%s)",
                    function_call.function.__name__,
                    function_call.arguments,
                )
                messages += _handle_function_call(function_call=function_call)
        elif isinstance(response, str):
            answer = response
    return AnsweredSubQuestion(subquestion=subquestion, answer=answer)


@chatprompt(
    SystemMessage(GENERATE_SUBQUESTION_SYSTEM_PROMPT_TEMPLATE),
    UserMessage("# User query\n{user_query}"),
    model=OpenaiChatModel(model="gpt-4o", temperature=0.0),
)
def generate_subquestions_from_query(user_query: str) -> list[SubQuestion]:
    ...


@chatprompt(
    SystemMessage(GENERATE_SUBQUESTION_SYSTEM_PROMPT_TEMPLATE),
    UserMessage("# User query\n{user_query}"),
    model=OpenaiChatModel(model="gpt-4o", temperature=0.0),
)
async def agenerate_subquestions_from_query(user_query: str) -> list[SubQuestion]:
    ...


def search_tools(
    subquestion: SubQuestion,
    tool_vector_index: VectorStore,
    answered_subquestions: list[AnsweredSubQuestion] | None = None,
) -> list[Callable]:
    def llm_query_tool_index(query: str) -> str:
        """Use natural language to search the tool index for tools."""
        logger.info("Searching tool index for: %s", query)
        results = tool_vector_index.similarity_search(query=query, k=4)
        return "\n".join([r.page_content for r in results])

    @prompt_chain(
        TOOL_SEARCH_PROMPT_TEMPLATE,
        model=OpenaiChatModel(model="gpt-3.5-turbo", temperature=0.2),
        functions=[llm_query_tool_index],
    )
    def _search_tools(
        subquestion: str, answered_subquestions: list[AnsweredSubQuestion] | None
    ) -> list[str]:
        ...

    tool_names = _search_tools(subquestion.question, answered_subquestions)
    callables = _get_callables_from_tool_search_results(
        tool_vector_index=tool_vector_index, tool_names=tool_names
    )
    return callables


async def asearch_tools(
    subquestion: SubQuestion,
    tool_vector_index: VectorStore,
    answered_subquestions: list[AnsweredSubQuestion] | None = None,
) -> list[Callable]:
    def llm_query_tool_index(query: str) -> str:
        """Use natural language to search the tool index for tools."""
        logger.info("Searching tool index for: %s", query)
        results = tool_vector_index.similarity_search(query=query, k=4)
        return "\n".join([r.page_content for r in results])

    @prompt_chain(
        TOOL_SEARCH_PROMPT_TEMPLATE,
        model=OpenaiChatModel(model="gpt-3.5-turbo", temperature=0.2),
        functions=[llm_query_tool_index],
    )
    async def _search_tools(
        subquestion: str, answered_subquestions: list[AnsweredSubQuestion] | None
    ) -> list[str]:
        ...

    tool_names = await _search_tools(subquestion.question, answered_subquestions)
    callables = _get_callables_from_tool_search_results(
        tool_vector_index=tool_vector_index, tool_names=tool_names
    )
    return callables


def _get_callables_from_tool_search_results(
    tool_vector_index: VectorStore,
    tool_names: list[str],
) -> list[Callable]:
    callables = []
    for tool_name in tool_names:
        for doc in tool_vector_index.docstore._dict.values():  # type: ignore
            if doc.metadata["tool_name"] == tool_name:
                callables.append(doc.metadata["callable"])
                break
    return callables


def _handle_function_call(function_call: FunctionCall) -> list[Any]:
    try:
        result = function_call()
        return _build_messages_for_function_call(
            function_call=function_call, result=result
        )
    except Exception as err:
        return _build_messages_for_generic_error(function_call=function_call, err=err)


def _build_messages_for_function_call(
    function_call: FunctionCall,
    result: Any,
) -> list[Any]:
    return [
        AssistantMessage(function_call),
        FunctionResultMessage(content=str(result), function_call=function_call),
    ]


def _build_messages_for_generic_error(
    function_call: FunctionCall,
    err: Exception,
) -> list[Any]:
    logger.error(f"Error calling function: {err}")
    return [
        AssistantMessage(function_call),
        FunctionResultMessage(content=str(err), function_call=function_call),
    ]
