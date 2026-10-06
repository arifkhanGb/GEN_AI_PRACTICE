from fastapi import FastAPI
import inngest
import logging
import inngest.fast_api



# # Create an Inngest client
# inngest_client = inngest.Inngest(
#     app_id="fast_api_example",
#     logger=logging.getLogger("uvicorn"),
# )


# # Create an Inngest function
# @inngest_client.create_function(
#     fn_id="my_function",
#     # Event that triggers this function
#     trigger=inngest.TriggerEvent(event="app/my_function"),
# )
# async def my_function(ctx: inngest.Context) -> str:
#     ctx.logger.info(ctx.event)
#     return "done"


app = FastAPI()

# Serve the Inngest endpoint
inngest.fast_api.serve(app, inngest_client, [my_function])