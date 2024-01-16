# OpenBB-agents
Active work-in-progress. Consider pre-alpha for now.

This is a project that leverages LLMs and OpenBB Platform to create financial
analyst agents that can autonomously perform financial research, and answer
questions using up-to-date data. This is possible as a result of agents
utilizing function calling to interact with the OpenBB platform.


## Installation
Currently, we only support Python 3.11. We will be adding support for more version of Python relatively soon.

`openbb-agents` is available as a PyPI package:

``` sh
pip install openbb-agents --upgrade
```

## Usage

``` python
>>> from openbb_agents.agent import openbb_agent
>>> result = openbb_agent("What is the current market cap of TSLA?")  # Will print some logs to show you progress
>>> print(result)
- The current market cap of TSLA (Tesla, Inc.) is approximately $695,833,798,800.00.
- This figure is based on the most recent data available, which is from January 15, 2024.
- The market cap is calculated by multiplying the current stock price ($218.89) by the number of outstanding shares (3,178,920,000).
```

If you've cloned the repository, you can use the `run.py` script and pass in your query:
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


## Development
- Create a new virtual environment, with `poetry `
- `poetry install`

### Linting and Formatting
We're currently experimenting with `ruff` as a drop-in replacement for `black`, `isort` and `pylint`.

You can run linting checks as follows:

``` sh
ruff check
```

Or fix linting errors:

``` sh
ruff check --fix
```

Or format the code:

``` sh
ruff format
```

We've also included these in the `pre-commit`, if you'd prefer to have these checks run automatically before commiting code. 
You can install the `pre-commit` hooks as follows:

``` sh
pre-commit install
```

### Testing

We are in the process of adding tests.

We use `pytest` as our test-runner:

``` sh
pytest tests/
```

