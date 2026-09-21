import os
import logging
import inngest
from dotenv import load_dotenv

# This module is imported before the webhook module (which also loads `.env`).
# Load it here so Inngest sees INNGEST_DEV while its client is initialized.
load_dotenv()

inngest_client = inngest.Inngest(
    app_id="github-pr-review-agent",
    # Use development mode when INNGEST_DEV is configured locally.
    # is_production=os.getenv("INNGEST_DEV") is None,
    logger=logging.getLogger("uvicorn"),
)
