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
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain.tools import StructuredTool
from langchain.tools.render import (
    format_tool_to_openai_function,
    render_text_description_and_args,
)


from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS, VectorStore

from openbb_agents.models import AnsweredSubQuestion, SubQuestion


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
