
import logging

import inngest

inngest_client = inngest.Inngest(
    app_id="repomind",
    logger=logging.getLogger("uvicorn"),
)
