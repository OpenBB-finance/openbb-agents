# OpenBB LLM Agents
Work-in-progress.

This is a project that leverages LLMs and [OpenBB Platform](https://github.com/OpenBB-finance/OpenBBTerminal/tree/develop/openbb_platform) to create financial
analyst agents that can autonomously perform financial research, and answer
questions using up-to-date data. This is possible as a result of agents
utilizing function calling to interact with the OpenBB Platform.


## Installation
Currently, we only support Python 3.11. We will be adding support for more version of Python relatively soon.

`openbb-agents` is available as a PyPI package:

``` sh
pip install openbb-agents --upgrade
```

## Setup
### OpenAI API keys

To use OpenBB LLM Agents, you need an OpenAI API key. Follow these steps:

1. **Get API Key**: Sign up on [OpenAI](https://www.openai.com/) and get your API key.
2. **Set Environment Variable**: Add this to your shell profile (`.bashrc`, `.zshrc`, etc.):
    ```sh
    export OPENAI_API_KEY="your_openai_api_key"
    ```

### OpenBB Platform data provider credentials
To use the OpenBB Platform functions, you need to configure the necessary [data provider API credentials](https://docs.openbb.co/platform/usage/api_keys). This can be done in one of two ways:

1. **Local Configuration**: Specify your credentials in a `~/.openbb_platform/user_settings.json` file. Follow the [local environment setup guide](https://docs.openbb.co/platform/usage/api_keys#local-environment) for detailed instructions.
2. **OpenBB Hub**: Create a personal access token (PAT) via your [OpenBB Hub](https://my.openbb.co/) account. This PAT can then be passed to the agent as an argument.


## Usage

``` python
>>> from openbb_agents.agent import openbb_agent
>>> result = openbb_agent("What is the current market cap of TSLA?")  # Will print some logs to show you progress
>>> print(result)
- The current market cap of TSLA (Tesla, Inc.) is approximately $695,833,798,800.00.
- This figure is based on the most recent data available, which is from January 15, 2024.
- The market cap is calculated by multiplying the current stock price ($218.89) by the number of outstanding shares (3,178,920,000).
```

To use your data provider credentials stored in OpenBB Hub, you can pass in your OpenBB Hub PAT directly to the agent:

``` python
>>> openbb_agent("What is the stock price of AAPL?", openbb_pat="<openbb-hub-pat>")
```

**Note:** The agent dynamically configures itself based on the available data provider credentials. Consequently, certain data sources and functions may be inaccessible without the appropriate API key. By default, `yfinance` is included as a data provider and does not require an API key. For a comprehensive list of functions and their supported data providers, refer to the [OpenBB Platform documentation](https://docs.openbb.co/platform/reference).

Queries can be relatively complex:

```python
>>> openbb_agent("Perform a fundamentals financial analysis of AMZN using the most recently available data. What do you find that's interesting?")
```

Queries can also have temporal dependencies (i.e the answers of previous subquestions are required to answer a later subquestion):

``` python
>>> openbb_agent("Who are TSLA's peers? What is their respective market cap? Return the results in _descending_ order of market cap.")
```

An `async` variant of the agent is also available:

``` python
>>> from openbb_agents.agent import aopenbb_agent
>>> await aopenbb_agent("What is the current market cap of TSLA?")
```


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
We use `pytest` as our test runner:
``` sh
pytest -n 8 tests/
```

