import argparse

from openbb_agents import agent

parser = argparse.ArgumentParser(description="Query the OpenBB agent.")
parser.add_argument(
    "query", metavar="query", type=str, help="The query to send to the agent."
)
parser.add_argument(
    "-v", "--verbose", action="store_true", help="Enable verbose logging."
)
args = parser.parse_args()

# We only import after passing in command line args to have verbosity propagate.

query = args.query
result = agent.openbb_agent(query, verbose=args.verbose)

print("============")
print("Final Answer")
print("============")
print(result)
