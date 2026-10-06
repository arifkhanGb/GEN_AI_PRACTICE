from typing import Any

from app.lib.github_client import github_client
from app.agents.pr_review_agent import analyse_pull_request


def get_pull_request(
    owner: str,
    repo_name: str,
    pull_number: int,
) -> dict[str, Any] | None:

    try:
        repo = github_client.get_repo(
            f"{owner}/{repo_name}"
        )

        pull_request = repo.get_pull(pull_number)

        return {
            "id": pull_request.id,
            "title": pull_request.title,
            "state": pull_request.state,
            "number": pull_request.number,
            "comments": pull_request.comments,
            "url": pull_request.url,
            "html_url": pull_request.html_url,
            "diff_url": pull_request.diff_url,
            "changes": pull_request.changed_files,
            "commits": pull_request.commits,
            "head": {
                "ref": pull_request.head.ref,
                "sha": pull_request.head.sha,
            },
        }

    except Exception as error:
        print(f"Failed to fetch pull request: {error}")
        return None


def get_pull_request_changes(
    owner: str,
    repo_name: str,
    pull_number: int,
) -> list[dict]:

    repo = github_client.get_repo(
        f"{owner}/{repo_name}"
    )

    pull_request = repo.get_pull(pull_number)

    changes = []

    for file in pull_request.get_files():
        changes.append({
            "filename": file.filename,
            "status": file.status,
            "changes": file.changes,
            "patch": file.patch,
            "additions": file.additions,
            "deletions": file.deletions,
            "previous_filename": file.previous_filename,
        })

    return changes


def post_pull_request_comment(
    owner: str,
    repo_name: str,
    pull_number: int,
    review: dict,
) -> None:

    repo = github_client.get_repo(
        f"{owner}/{repo_name}"
    )

    issue = repo.get_issue(
        pull_number
    )

    sections = [
        review["content"]
    ]

    if review.get("critical_fixes"):
        critical_fixes = "\n".join(
            f"- {fix}"
            for fix in review["critical_fixes"]
        )

        sections.append(
            f"**Critical Changes:**\n{critical_fixes}"
        )

    if review.get("suggestions"):
        suggestions = "\n".join(
            f"- {suggestion}"
            for suggestion in review["suggestions"]
        )

        sections.append(
            f"**Suggestions:**\n{suggestions}"
        )

    comment_body = "\n\n".join(
        sections
    )

    issue.create_comment(
        comment_body
    )