import os
import logging
import inngest

inngest_client = inngest.Inngest(
    app_id="github-pr-review-agent",
    # Use development mode when INNGEST_DEV is configured locally.
    # is_production=os.getenv("INNGEST_DEV") is None,
    logger=logging.getLogger("uvicorn"),
)
