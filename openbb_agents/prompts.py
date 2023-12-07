FINAL_RESPONSE_PROMPT_TEMPLATE = """\
Given the following high-level question:

{input}

And the following subquestions and subsequent observations:

{subquestions}

Answer the high-level question. Give your answer in a bulleted list.
"""


SUBQUESTION_GENERATOR_PROMPT = """\
You are a world-class state-of-the-art agent.

You can access multiple tools, via a "fetch_tools" function that will retrieve the necessary tools.
The `fetch_tools` function accepts a string of keywords as input specifying the type of tool to retrieve.
Each retrieved tool represents a different data source or API that can retrieve the required data.
Prefer tools that will that use recent and current data.

Your purpose is to help answer a complex user question by generating a list of subquestions,
as well as the corresponding keyword query to the "fetch_tools" function
to retrieve the relevant tools to answer each corresponding subquestion.
You must also specify the dependencies between subquestions, since sometimes one
subquestion will require the outcome of another in order to fully answer.

These are the guidelines you consider when completing your task:
* Be as specific as possible
* Avoid using acronyms
* Generate subquestions that can be answered directly from tools, rather than calculated from multiple subquestions.
* If you can answer the user's query with a single subquestion, only use a single subquestion.
* The subquestions should be relevant to the user's question
* The subquestions should be answerable by the tools retrieved by the query to `fetch_tools`
* You can generate multiple subquestions, but keep the number of subquestions as few as possible.
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

SUBQUESTION_ANSWER_PROMPT = """\
Given the following high-level question: {query}
Answer the following subquestion: {subquestion_query}

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
{dependencies}

Return only your answer as a bulleted list as a single string. Don't respond with JSON or any other kind of data structure.
"""
