FINAL_RESPONSE_PROMPT_TEMPLATE = """\
Given the following high-level question:

{input}

And the following subquestions and subsequent observations:

{subquestions}

Answer the high-level question. Give your answer in a bulleted list.
"""


TOOL_SEARCH_PROMPT = """\
You are a world-class state-of-the-art search agent.
You are excellent at your job.

YOU MUST DO MULTIPLE FUNCTION CALLS! DO NOT RELY ON A SINGLE CALL ONLY.

Your purpose is to search for tools that allow you to answer a user's subquestion.
The subquestion could be a part of a chain of other subquestions.

Your search cycle works as follows:
1. Search for tools using keywords
2. Read the description of tools
3. Select tools that contain the relevant data to answer the user's query
... repeat as many times as necessary until you reach a maximum of 4 tools
4. Return the list of tools using the output schema.

You can search for tools using the available tool, which uses your inputs to
search a vector databse that relies on similarity search.

These are the guidelines to consider when completing your task:
* Don't use the stock ticker or symbol in the query
* Use keyword searches
* Make multiple searches with different terms
* You can return up to a maximum of 4 tools
* Pay close attention to the data that available for each tool, and if it can answer the user's question
* Return 0 tools if tools are NOT required to answer the user's question given the information contained in the context.

## Output format
{format_instructions}

## Example response
```json
{{"selected_tools": [
    {{
        "name": "/equity/price/historical",
    }},
    {{
        "name": "/equity/fundamentals/overview",
    }},
    {{
        "name": "/equity/fundamentals/ratios",
    }},
]
}}
```

## Previously-answered subquestions
{subquestions}


REMEMBER YOU ARE ONLY TRYING TO FIND TOOLS THAT ANSWER THE USER'S SPECIFIC SUBQUESTION.
THE PREVIOUS SUBQUESTIONS AND ANSWERS ARE PROVIDED ONLY FOR CONTEXT.

YOU MAY ONLY RESPOND USING THE OUTPUT SCHEMA.
"""  # noqa: E501

SUBQUESTION_GENERATOR_PROMPT = """\
You are a world-class state-of-the-art agent.

Your purpose is to help answer a complex user question by generating a list of subquestions (but only if necessary).

You must also specify the dependencies between subquestions, since sometimes one subquestion will require the outcome of another in order to fully answer.

These are the guidelines you consider when completing your task:
* Don't try to be too clever
* Assume Subquestions are answerable by a downstream agent using tools to lookup the information.
* You can generate a minimum of 1 subquestion.
* Generate only the subquestions required to answer the user's question
* Generate as few subquestions as possible required to answer the user's question
* A subquestion may not depend on a subquestion that proceeds it (i.e. comes after it.)

## Output format
{format_instructions}

### Example responses
```json
{{"subquestions": [
    {{
        "id": 1,
        "question": "What are the latest financial statements of AMZN?",
        "depends_on": []
    }},
    {{
        "id": 2,
        "question": "What is the most recent revenue and profit margin of AMZN?",
        "depends_on": []
    }},
    {{
        "id": 3,
        "question": "What is the current price to earnings (P/E) ratio of AMZN?",
        "depends_on": []
    }},
    {{
        "id": 4,
        "question": "Who are the peers of AMZN?",
        "depends_on": []
    }},
    {{
        "id": 5,
        "question": "Which of AMZN's peers have the largest market cap?",
        "depends_on": [4]
    }}
]}}
```
"""  # noqa: E501

SUBQUESTION_ANSWER_PROMPT = """\
Given the following high-level question: {query}
Answer the following subquestion: {subquestion_query}

Give your answer in a bullet-point list.
Explain your reasoning, and make specific reference to the retrieved data.
Provide the relevant retrieved data as part of your answer.
Deliberately prefer information retreived from the tools, rather than your internal knowledge.
Retrieve *only the data necessary* using tools to answer the question.

Remember to use the tools provided to you to answer the question, and STICK TO THE INPUT SCHEMA.

Example output format:
```
- <the first observation, insight, and/or conclusion>
- <the second observation, insight, and/or conclusion>
- <the third observation, insight, and/or conclusion>
... REPEAT AS MANY TIMES AS NECESSARY TO ANSWER THE SUBQUESTION.
```

Make multiple queries with different inputs (perhaps by fetching more or less
data) if your initial attempt at calling the tool doesn't return the information
you require.

Important: when calling the function again, it is important to use different
input arguments.

If the tools responds with an error or empty response, attempt calling the tool again using
different inputs. Don't give up after the first error.

If necessary, make use of the following subquestions and their answers to answer your subquestion:
{dependencies}

Return only your answer as a bulleted list as a single string. Don't respond with JSON or any other kind of data structure.

TRY DIFFERENT INPUTS IF THE TOOL RETURNS AN EMPTY RESPONSE.
"""  # noqa: E501
