import os
from dotenv import load_dotenv
from github import Github

load_dotenv()

github = Github(
    auth = os.getenv("GITHUB_PAT"),
    user_agent = "pullrequest-review-bot"
)