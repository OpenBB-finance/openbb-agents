from openbb_agents.agents import v1, v2


def openbb_agent_v1(query: str):
    return v1.openbb_agent(query=query)


def openbb_agent_v2(query: str):
    return v2.openbb_agent(query=query)
