import os

from dotenv import load_dotenv
from github import Auth, Github

load_dotenv()

# This creates one reusable authenticated GitHub client.
github_token = os.getenv("GITHUB_PAT")

if not github_token:
    raise RuntimeError("GITHUB_PAT environment variable is not configured")


github_client = Github(
    auth=Auth.Token(github_token),
    user_agent="github-pr-review-agent",
)
