import argparse

from openbb_agents.agent import OpenBBAgent

# Initialize the argument parser
parser = argparse.ArgumentParser(description="Query the OpenBB agent.")
# Add argument for the query
parser.add_argument(
    "query", metavar="query", type=str, help="The query to send to the agent."
)
# Add argument for the personal access token
parser.add_argument(
    "-t",
    "--token",
    type=str,
    required=True,
    help="Your personal access token for the OpenBB agent.",
)
# Add argument for verbose logging
parser.add_argument(
    "-v", "--verbose", action="store_true", help="Enable verbose logging."
)
# Parse the arguments
args = parser.parse_args()

# Create an instance of OpenBBAgent with the provided token and verbosity setting
agent = OpenBBAgent(personal_access_token=args.token, verbose=args.verbose)

# Use the agent to answer the query
result = agent.answer_query(args.query)

print("============")
print("Final Answer")
print("============")
print(result)
