import hashlib
import hmac
import os

from dotenv import load_dotenv


load_dotenv()


github_webhook_secret = os.getenv(
    "GITHUB_WEBHOOK_SECRET"
)

if not github_webhook_secret:
    raise RuntimeError(
        "GITHUB_WEBHOOK_SECRET is not configured"
    )


def verify_github_signature(
    payload_body: bytes,
    signature_header: str | None,
) -> bool:

    if not signature_header:
        return False

    expected_signature = (
        "sha256="
        + hmac.new(
            github_webhook_secret.encode(),
            payload_body,
            hashlib.sha256,
        ).hexdigest()
    )

    return hmac.compare_digest(
        expected_signature,
        signature_header,
    )