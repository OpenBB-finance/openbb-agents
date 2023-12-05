import logging
import langchain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.agents.output_parsers import JSONAgentOutputParser
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents import AgentExecutor
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain import hub
from langchain.tools.render import render_text_description_and_args
from langchain.output_parsers import PydanticOutputParser

from openbb import obb

from .utils import map_openbb_collection_to_langchain_tools
from .models import SubQuestionList
from .prompts import SUBQUESTION_GENERATOR_PROMPT, FINAL_RESPONSE_PROMPT_TEMPLATE

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
    
    llm = ChatOpenAI(model=model)
    subquestion_chain = {"input": lambda x: x["input"]} | prompt | llm | subquestion_parser
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


def _render_subquestions_and_answers(subquestions):
    "Combines all subquestions and their answers"
    output = ""
    for subquestion in subquestions:
        output += "Subquestion: " + subquestion["subquestion"] + "\n"
        output += "Observations: \n" + str(subquestion["observation"]) + "\n\n"

    return output


def generate_final_response(query: str, subquestions: dict):
    """Generate the final response to a query given answer to a list of subquestions."""
    system_message =  FINAL_RESPONSE_PROMPT_TEMPLATE
    prompt = ChatPromptTemplate.from_messages([("system", system_message)])
    
    llm = ChatOpenAI(model="gpt-4")  # Let's use the big model for the final answer.
    
    chain = (
        {
            "input": lambda x: x["input"],
            "subquestions": lambda x: _render_subquestions_and_answers(x["subquestions"]),
        }
        | prompt
        | llm
    )
    
    result = chain.invoke({"input": query, "subquestions": subquestions})

    return result


def openbb_agent(
        openbb_tools: langchain.tools.base.StructuredTool,
        query: str,
    ):

    logger.info("Creating vector db of tools.")
    docs = [
        Document(page_content=t.description, metadata={"index": i})
        for i, t in enumerate(openbb_tools)
    ]

    vector_store = FAISS.from_documents(docs, OpenAIEmbeddings())

    logger.info("Generating subquestions for query: %s", query)

    subquestion_list = generate_subquestions(query)
    logger.info("Generated subquestions.", extra={"subquestions": subquestion_list})

    subquestions_and_tools = []
    for subquestion in subquestion_list.subquestions:

        retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={'score_threshold': 0.65}
        )
        docs = retriever.get_relevant_documents(subquestion.query)
        
        # This is a fallback mechanism in case the threshold is too high,
        # causing too few tools to be returned.  In this case, we fall back to
        # getting the top k=2 results with higher similarity scores.
        if len(docs) < 2:
            retriever = vector_store.as_retriever(
                search_kwargs={"k": 2}
            )
            docs = retriever.get_relevant_documents(subquestion.query)
            
        tools = [openbb_tools[d.metadata["index"]] for d in docs]

        subquestions_and_tools.append(
            {   "id": subquestion.id,
                "subquestion": subquestion.question,
                "query": subquestion.query,
                "tools": tools,
                "depends_on": subquestion.depends_on,
            }
        )

    # Go through each subquestion and create an agent with the necessary tools
    # and context to execute on it\n
    for subquestion in subquestions_and_tools:
        # We handle each question dependency manually since we don't want agents
        # to share memory as this can go over context length
        deps = [dep for dep in subquestions_and_tools if dep["id"] in subquestion["depends_on"]]

        dependencies_str = ""
        for dep in deps:
            if "observation" in dep:
                dependencies_str += "subquestion: " + dep["subquestion"] + "\n"
                dependencies_str += "observations:\n" + str(dep["observation"]) + "\n\n"

        input = f"""\
Given the following high-level question: {query}
Answer the following subquestion: {subquestion['subquestion']}

Give your answer in a bullet-point list.
Explain your reasoning, and make specific reference to the retrieved data.
Provide the relevant retrieved data as part of your answer.
Deliberately prefer information retreived from the tools, rather than your internal knowledge.

Remember to use the tools provided to you to answer the question, and STICK TO THE INPUT SCHEMA.

Example output format:
```
- <the first observation, insight, and/or conclusion> 
- <the second observation, insight, and/or conclusion> 
- <the third observation, insight, and/or conclusion> 
... REPEAT AS MANY TIMES AS NECESSARY TO ANSWER THE SUBQUESTION.
```

If necessary, make use of the following subquestions and their answers to answer your subquestion:
{dependencies_str}

Return only your answer as a bulleted list as a single string. Don't respond with JSON or any other kind of data structure.
"""

        logger.info("Answering subquestion: %s", subquestion["subquestion"])
        try:
            result = make_react_agent(tools=subquestion["tools"]).invoke({"input": input})
            output = result["output"]
        except Exception as err:  # Terrible practice, but it'll do for now.
            print(err)
            # We'll include the error message in the future
            output = "I was unable to answer the subquestion using the available tools."
        subquestion["observation"] = output

        logger.info("Subquestion Answered.", extra={
            "id": subquestion["id"],
            "subquestion": subquestion["subquestion"],
            "dependencies": subquestion["depends_on"],
            "tools": [tool.name for tool in subquestion["tools"]],
            "observation": subquestion["observation"]
        })

    result = generate_final_response(
        query=query,
        subquestions=subquestions_and_tools
    )

    logger.info("Final Answer: %s", result.content)
    print("============")
    print("Final Answer")
    print("============")
    print(result.content)
    return result.content
