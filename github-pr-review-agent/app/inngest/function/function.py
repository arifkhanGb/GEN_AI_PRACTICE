import inngest
from app.inngest.client import inngest_client

# Register this Python function as an Inngest function. 
@inngest_client.create_function(
    fn_id="hello-world",
    trigger=inngest.TriggerEvent(
        event="test/hello.world"
    ),
)
async def hello_world(ctx: inngest.Context) -> str:

    ctx.logger.info("Hello World function started")

    return "Hello from Inngest!"