# OpenBB-agents
Active work-in-progress. Consider pre-alpha for now.

This is a project that leverages LLMs and OpenBB Platform to create financial
analyst agents that can autonomously perform financial research, and answer
questions using up-to-date data. This is possible as a result of agents
utilizing function calling to interact with the OpenBB platform.


## Set-up
- Create a new virtual environment
- `poetry install`

## Usage
Use the `run.py` script and pass in your query.

Queries can be simple:

``` sh
python run.py "What is the current market cap of TSLA?"
```

Queries can be complex:

``` sh
python run.py "Perform a fundamentals financial analysis of AMZN using the most recently available data. What do you find that's interesting?"
```

Queries can also have temporal dependencies (i.e the answers of previous subquestions are required to answer a later subquestion):

``` sh
python run.py "Who are TSLA's peers? What is their respective market cap? Return the results in _descending_ order of market cap."
```

There is more functionality coming very soon!
