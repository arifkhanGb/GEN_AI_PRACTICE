from fastapi import FastAPI
from app.inngest.client import inngest_client
from app.inngest.function import hello_world
from app.inngest.function import github_pr_review
import inngest
from inngest.fast_api import serve

import json

from fastapi import (
    Header,
    HTTPException,
    Request,
)
from app.webhooks.github import (
    verify_github_signature,
)

app = FastAPI(
    title="GitHub PR Review Agent",
    description="AI-powered GitHub Pull Request Review Agent",
    version="1.0.0",
)


@app.get("/")
async def root():
    return {
        "message": "GitHub PR Review Agent is running"
    }


@app.get("/health")
async def health():
    return {
        "status": "ok"
    }
    
    
@app.post("/webhooks/github")
async def github_webhook(
    request: Request,
    x_github_event: str | None = Header(
        default=None
    ),
    x_github_delivery: str | None = Header(
        default=None
    ),
    x_hub_signature_256: str | None = Header(
        default=None
    ),
):
    
    # --------------------------------------------------
    # 1. Read the raw request body
    # --------------------------------------------------
    payload_body = await request.body()

    # --------------------------------------------------
    # 2. Verify that the request really came from GitHub
    # --------------------------------------------------

    if not verify_github_signature(
        payload_body,
        x_hub_signature_256,
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid GitHub webhook signature",
        )

    print(
    f"GitHub delivery received: "
    f"event={x_github_event}, "
    f"delivery={x_github_delivery}",
    flush=True,
    )


    # -----------------------------
    # 3. Ignore events we don't need
    # -----------------------------

    if x_github_event != "pull_request":
        return {
            "message": "Event ignored",
            "event": x_github_event,
        }

    # -----------------------------
    # 4. Parse GitHub payload
    # -----------------------------

    payload = json.loads(
        payload_body
    )

    action = payload.get("action")
    
    # --------------------------------------------------
    # 5. Only review PRs when relevant actions happen
    # --------------------------------------------------
    
    allowed_actions = {
        "opened",
        "reopened",
        "synchronize",
    }

    if action not in allowed_actions:
        return {
            "message": "Pull request action ignored",
            "action": action,
        }
    
    # --------------------------------------------------
    # 6. Extract repository + PR information
    # --------------------------------------------------
    repository = payload["repository"]

    owner = repository["owner"]["login"]
    repo_name = repository["name"]
    pull_number = payload["number"]
    
     # --------------------------------------------------
    # 7. Send event to Inngest
    # --------------------------------------------------

    event_ids = await inngest_client.send(
        inngest.Event(
            name="github/pullrequest.review",
            id=x_github_delivery,
            data={
                "owner": owner,
                "repo": repo_name,
                "pull_number": pull_number,
                "action": action,
                "delivery_id": x_github_delivery,
            },
        )
    )


    print(
        f"GitHub webhook processed: "
        f"{owner}/{repo_name}#{pull_number} "
        f"action={action} "
        f"delivery={x_github_delivery} "
        f"event_ids={event_ids}"
    )
    
    
     # --------------------------------------------------
    # 8. Respond quickly to GitHub
    # --------------------------------------------------

    return {
        "message": "Webhook received and Inngest event sent",
        "action": action,
        "owner": owner,
        "repo": repo_name,
        "pull_number": pull_number,
        "delivery_id": x_github_delivery,
        "event_ids": event_ids,
    }
  
    
# Expose Inngest functions to Inngest
serve(
    app,
    inngest_client,
    [
        hello_world,
        github_pr_review
        ],
)    
