# /**
#  * Event: {
#  *  data: {
#  *    "owner": "piyushgarg-dev"
# *     "repo": "chaicode-form-builder",
#       "pull_number": number
#  *    
#  *  }
#  * }
#  */

import inngest
from ..client import inngest_client

@inngest_client.create_function(
    fn_id="github-pr-review",
    trigger=inngest.TriggerEvent(event="github/pullrequest.review"),
)

async def github_pull_request_review(
    ctx: inngest.Context,
):
    
    # -----------------------------------------
    # Get event data
    # -----------------------------------------

    owner = ctx.event.data["owner"]
    repo_name = ctx.event.data["repo"]
    pull_number = ctx.event.data["pull_number"]
    
    # -----------------------------------------
    # STEP 1
    # Fetch Pull Request Information
    # -----------------------------------------
    
    pull_request_info = await ctx.step.run(
        "fetch-pull-request-information",
        lambda: fetch_pull_request_information(
            owner,
            repo_name,
            pull_number,
        ),
    )
    
    def fetch_pull_request_information(
        owner: str,
        repo_name: str,
        pull_number: int,
    ):

        try:
            repo = github.get_repo(
                f"{owner}/{repo_name}"
            )

            pull_request = repo.get_pull(
                pull_number
            )

            return {
                "id": pull_request.id,
                "title": pull_request.title,
                "state": pull_request.state,
                "number": pull_request.number,
                "comments": pull_request.comments,
                "url": pull_request.html_url,
                "diff_url": pull_request.diff_url,
                "changes": pull_request.changed_files,
                "commits": pull_request.commits,
                "head": {
                    "ref": pull_request.head.ref,
                    "sha": pull_request.head.sha,
                },
            }

        except Exception as error:

            print(
                f"Error fetching pull request: {error}"
            )

            return None
    
    # PR not found
    if pull_request_info is None:
        return {
            "message": "Pull request not found",
            "skipped": True,
        }

    # PR is not open
    if pull_request_info["state"] != "open":
        return {
            "message": "Pull request is not open, skipping the review",
            "skipped": True,
            "completed": False,
        }
        
        
    # -----------------------------------------
    # STEP 2
    # Fetch PR changes
    # -----------------------------------------        