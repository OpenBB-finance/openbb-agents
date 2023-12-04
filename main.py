# import dependencies, in specific langchain
import os
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
from langchain.pydantic_v1 import BaseModel, Field
from langchain.output_parsers import RetryWithErrorOutputParser
from langchain.llms import OpenAI
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

os.environ["TOKENIZERS_PARALLELISM"] = "False"  # Avoid some warnings from HuggingFace


# Set up OpenAI API key
import openai
os.environ["OPENAI_API_KEY"] = ""
openai.api_key = ""

# Set up OpenBB Personal Access Token from https://my.openbb.co/app/platform/pat
from openbb import obb
from utils import map_openbb_collection_to_langchain_tools  # provides access to OpenBB Tools
obb.account.login(pat="")


class SubQuestion(BaseModel):
    "Pydantic data model we want each subquestion to have, including each field and what they represent"
    id: int = Field(
        description="The unique ID of the subquestion."
    )
    question: str = Field(
        description="The subquestion itself."
    )
    query: str = Field(
        description="The query to pass to the `fetch_tools` function to retrieve the appropriate tool to answer the question."
    )
    depends_on: list[int] = Field(
        description="The list of subquestion ids whose answer is required to answer this subquestion.",
        default=[]
    )


class SubQuestionList(BaseModel):
    "Pydantic data model output we want to enforce, which is a list of the previous SubQuestion Pydantic model"
    subquestions: list[SubQuestion] = Field(
        description="The list of SubQuestion objects."
    )


def task_decomposition(task: str):
    "Break a larger query down into subquery. Then for each subquery create a set of keywords that allow you to fetch the right tool to execute that same subquery."
    subquestion_parser = PydanticOutputParser(pydantic_object=SubQuestionList)
    
    system_message = """\
    You are a world-class state-of-the-art agent.
    
    You can access multiple tools, via a "fetch_tools" function that will retrieve the necessary tools.
    The `fetch_tools` function accepts a string of keywords as input specifying the type of tool to retrieve.
    Each retrieved tool represents a different data source or API that can retrieve the required data.
    
    Your purpose is to help answer a complex user question by generating a list of subquestions,
    as well as the corresponding keyword query to the "fetch_tools" function
    to retrieve the relevant tools to answer each corresponding subquestion.
    You must also specify the dependencies between subquestions, since sometimes one
    subquestion will require the outcome of another in order to fully answer.
    
    These are the guidelines you consider when completing your task:
    * Be as specific as possible
    * Avoid using acronyms
    * The subquestions should be relevant to the user's question
    * The subquestions should be answerable by the tools retrieved by the query to `fetch_tools`
    * You can generate multiple subquestions
    * You don't need to query for a tool if you don't think it's relevant
    * A subquestion may not depend on a subquestion that proceeds it (i.e. comes after it.)
    
    ## Output format
    {format_instructions}
    
    ### Example responses
    ```json
    {{"subquestions": [
        {{
            "id": 1,
            "question": "What are the latest financial statements of AMZN?", 
            "query": "financial statements",
            "depends_on": []
        }}, 
        {{
            "id": 2,
            "question": "What is the most recent revenue and profit margin of AMZN?", 
            "query": "revenue profit margin ratios",
            "depends_on": []
        }}, 
        {{
            "id": 3,
            "question": "What is the current price to earnings (P/E) ratio of AMZN?", 
            "query": "ratio price to earnings",
            "depends_on": []
        }}, 
        {{
            "id": 4,
            "question": "Who are the peers of AMZN?", 
            "query": "peers",
            "depends_on": []
        }},
        {{
            "id": 5,
            "question": "Which of AMZN's peers have the largest market cap?", 
            "query": "market cap",
            "depends_on": [4]
        }}
    ]}}
    ```
    """
    
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
    
    llm = ChatOpenAI(
        model="gpt-4"
    )  # gpt-3.5-turbo works well, but gpt-4-1106-preview isn't good at returning JSON.
    
    subquestion_chain = {"input": lambda x: x["input"]} | prompt | llm | subquestion_parser

    subquestion_list = subquestion_chain.invoke({"input": task})

    return subquestion_list

def langchain_react_agent(tools):
    "Define a ReAct agent bound with specific tools."
    # This retrieves the ReAct agent chat prompt template available in Langchain Hub
    # https://smith.langchain.com/hub/hwchase17/react-json?organizationId=10beea65-e722-5aa1-9f93-034c22e3cd6e
    prompt = hub.pull("hwchase17/react-multi-input-json")
    # Replace the 'tools' and 'tool_names' content of the prompt with information given to the agent
    # Note that tool_names is a field available in each tool, so it can be inferred from same argument
    prompt = prompt.partial(
        tools=render_text_description_and_args(tools),
        tool_names=", ".join([t.name for t in tools]),
    )

    llm = ChatOpenAI(model="gpt-4-1106-preview").bind(stop=["\nObservation"])

    chain = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_log_to_str(x["intermediate_steps"]),
        }
        | prompt
        | llm
        | JSONAgentOutputParser()
    )

    # Agent executor with access to the chain and tools at its disposal
    agent_executor = AgentExecutor(
        agent=chain,
        tools=tools,
        verbose=False,  # <-- set this to False to cut down on output spam. But it's useful for debugging!
        return_intermediate_steps=False,
        handle_parsing_errors=True,
    )
    return agent_executor


def render_subquestions_and_answers(subquestions):
    "Combines all subquestions and their answers"
    output = ""
    for subquestion in subquestions:
        output += "Subquestion: " + subquestion["subquestion"] + "\n"
        output += "Observations: \n" + str(subquestion["observation"]) + "\n\n"

    return output


def verdict(question: str, subquestions: dict):
    "Based on the high-level question, it combines the subquestions and their answers to give one final concise answer"
    system_message = """\
        Given the following high-level question: 
    
        {input}
    
        And the following subquestions and subsequent observations:
    
        {subquestions}
    
        Answer the high-level question. Give your answer in a bulleted list.
        """
    
    prompt = ChatPromptTemplate.from_messages([("system", system_message)])
    
    llm = ChatOpenAI(model="gpt-4")  # Let's use the big model for the final answer.
    
    final_chain = (
        {
            "input": lambda x: x["input"],
            "subquestions": lambda x: render_subquestions_and_answers(x["subquestions"]),
        }
        | prompt
        | llm
    )
    
    result = final_chain.invoke({"input": question, "subquestions": subquestions})

    return result


def openbb_agent(
        openbb_tools: langchain.tools.base.StructuredTool,
        user_query: str,
    ):
    # Parse the description (i.e. docstring + output fields) for each of these tools
    docs = [
        Document(page_content=t.description, metadata={"index": i})
        for i, t in enumerate(openbb_tools)
    ]

    # Create embeddings from each of these function descriptions
    # this will be important for when we want the agent to know what
    # function to use for a particular query
    vector_store = FAISS.from_documents(docs, OpenAIEmbeddings())

    subquestion_list = task_decomposition(user_query)

    subquestions_and_tools = []
    for subquestion in subquestion_list.subquestions:

        # Tool retrieval
        retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={'score_threshold': 0.65}
        )
        docs = retriever.get_relevant_documents(subquestion.query)
        
        # This is a fallback mechanism in case the threshold is too high, causing too few tools to be returned.
        # In this case, we fall back to getting the top k=2 results with higher similarity scores.
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

    # Go through each subquestion and create an agent with the necessary tools and context to execute on it\n
    for i, subquestion in enumerate(subquestions_and_tools):

        # We handle each dependency manually since we don't want agents to share memory as this can go over context length
        deps = [dep for dep in subquestions_and_tools if dep["id"] in subquestion["depends_on"]]

        dependencies = ""
        for dep in deps:
            dependencies += "subquestion: " + dep["subquestion"] + "\n"
            # if for some reason there's no temporal dependency between the agents being run
            # this ensures the code doesn't break here
            if "observation" in dep:
                dependencies += "observations:\n" + str(dep["observation"]) + "\n\n"

        input = f"""\
Given the following high-level question: {user_query}
Answer the following subquestion: {subquestion['subquestion']}

Give your answer in a bullet-point list.
Explain your reasoning, and make reference to and provide the relevant retrieved data as part of your answer.

Remember to use the tools provided to you to answer the question, and STICK TO THE INPUT SCHEMA.

Example output format:
```
- <the first observation, insight, and/or conclusion> 
- <the second observation, insight, and/or conclusion> 
- <the third observation, insight, and/or conclusion> 
... REPEAT AS MANY TIMES AS NECESSARY TO ANSWER THE SUBQUESTION.
```

If necessary, make use of the following subquestions and their answers to answer your subquestion:
{dependencies}

Return only your answer as a bulleted list as a single string. Don't respond with JSON or any other kind of data structure.
"""

        try:
            result = langchain_react_agent(tools=subquestion["tools"]).invoke({"input": input})
            output = result["output"]
        except Exception as err:  # Terrible practice, but it'll do for now.
            print(err)
            # We'll include the error message in the future
            output = "I was unable to answer the subquestion using the available tool." 


        # This is very cheeky but we are basically going into the subquestions_and_tools and for this current subquestion
        # we are adding the output as an observation. This is important because then above we do the dependencies check-up
        # which allows us to retrieve the correct output to be used in another subquestion.
        # Note: this works because subquestions are done in order to execute prompt. Otherwise it wouldn't since we would
        # be looking for an "observation" that doesn't exist yet.
        subquestion["observation"] = output

    
    result = verdict(
        question=user_query,
        subquestions=subquestions_and_tools
    )

    return result.content


#     user_query =  """\
# Check what are TSLA peers. From those, check which one has the highest market cap.
# Then, on the ticker that has the highest market cap get the most recent price target estimate from an analyst,
# and tell me who it was and on what date the estimate was made.
# """

    #user_query = "Perform a fundamentals financial analysis of AMZN using the most recently available data. What do you find that's interesting?"

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define a Pydantic model for the request body
class Query(BaseModel):
    user_query: str

# Define a POST route
@app.post("/analyze")
def analyze(query: Query):
    openbb_tools = map_openbb_collection_to_langchain_tools(
        openbb_commands_root=[
            "/equity/fundamental",
            "/equity/compare",
            "/equity/estimates"
        ]
    )
    output = openbb_agent(openbb_tools, query.user_query)
    return {"result": output}