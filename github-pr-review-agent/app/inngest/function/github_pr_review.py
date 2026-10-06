import inngest

from app.inngest.client import inngest_client
from app.lib.pull_requests import get_pull_request
from app.lib.pull_requests import get_pull_request_changes
from app.lib.pull_requests import post_pull_request_comment
from app.agents.pr_review_agent import analyse_pull_request

@inngest_client.create_function(
    fn_id="github-pr-review",
    trigger=inngest.TriggerEvent(
        event="github/pullrequest.review"
    ),
)

async def github_pr_review(ctx: inngest.Context):

    owner = ctx.event.data["owner"]
    repo_name = ctx.event.data["repo"]
    pull_number = ctx.event.data["pull_number"]

    ctx.logger.info(
        f"Starting PR review: {owner}/{repo_name}#{pull_number}"
    )
    
    #  // 1. Fetch the Pull Request Information
    
    pull_request_info = await ctx.step.run(
        "fetch-pull-request-information",
        get_pull_request,
        owner,
        repo_name,
        pull_number,
    )
    
    if pull_request_info is None:
        return {
            "message": "Pull request not found",
            "skipped": True,
        }

    if pull_request_info["state"] != "open":
        return {
            "message": "Pull request is not open, skipping the review",
            "skipped": True,
            "completed": False,
        }


     # Step 2: Fetch PR changes
    changes = await ctx.step.run(
        "fetch-changes",
        lambda: get_pull_request_changes(
            owner,
            repo_name,
            pull_number,
        ),
    )
    if not changes:
        return {
            "message": "There are no changes in this PR",
            "skipped": True,
        }
        
        
    # --------------------------------
    # Step 3: AI analysis
    # --------------------------------
    ai_response = await ctx.step.run(
    "ai-analyse",
    analyse_pull_request,
    pull_request_info,
    changes,
    )
    
    
    # --------------------------------
    # Step 4: Post GitHub comment
    # --------------------------------
    await ctx.step.run(
    "post-comment",
    post_pull_request_comment,
    owner,
    repo_name,
    pull_number,
    ai_response,
    )
    
    ctx.logger.info(
        f"Found {len(changes)} changed files"
    )
    
    
     
     
    # return {
    #     "message": "Pull request information fetched successfully",
    #     "pull_request": pull_request_info,
    #     "changes": changes,
    # }

    return {
    "message": "Pull request review completed successfully",
    "review": ai_response,
}
    
    
   