from app.lib.pull_requests import get_pull_request


result = get_pull_request(
    "arifkhanGb",
    "FASTAPI-FOUNDATION",
    1,
)

print(result)