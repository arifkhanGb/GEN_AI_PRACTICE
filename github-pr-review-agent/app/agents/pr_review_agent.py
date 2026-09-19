import os

from pydantic import BaseModel
from dotenv import load_dotenv

# OpenAI imports - kept for future use
# from openai import OpenAI

from google import genai


load_dotenv()


class ReviewResult(BaseModel):
    content: str
    critical_fixes: list[str]
    suggestions: list[str]


# ============================================================
# OpenAI Configuration
# Kept commented out for future use
# ============================================================

# openai_api_key = os.getenv("OPENAI_API_KEY")

# if not openai_api_key:
#     raise RuntimeError(
#         "OPENAI_API_KEY is not configured"
#     )

# openai_client = OpenAI(
#     api_key=openai_api_key
# )


# ============================================================
# Gemini Configuration
# ============================================================

gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    raise RuntimeError(
        "GEMINI_API_KEY is not configured"
    )


gemini_client = genai.Client(
    api_key=gemini_api_key
)


# ============================================================
# Pull Request Analysis
# ============================================================

def analyse_pull_request(
    pull_request_info: dict,
    changes: list[dict],
) -> dict:

    prompt = f"""
You are an expert senior software engineer performing
a GitHub Pull Request code review.

Review the following Pull Request.

Pull Request Information:

{pull_request_info}

Changed Files:

{changes}

Review the code carefully.

Focus on:

1. Bugs and incorrect behavior
2. Security issues
3. Performance problems
4. Error handling
5. Maintainability
6. Code quality
7. Potential production issues

Do not invent problems.

Only report an issue when there is reasonable evidence
from the provided code.

Return:

- an overall review
- critical fixes that should be addressed
- useful suggestions for improvement
"""


    # ========================================================
    # OpenAI Implementation
    # Kept commented out for future use
    # ========================================================

    # response = openai_client.responses.parse(
    #     model="gpt-4o-2024-08-06",
    #     input=prompt,
    #     text_format=ReviewResult,
    # )


    # ========================================================
    # Gemini Implementation
    # ========================================================

    response = gemini_client.models.generate_content(
        model="gemini-3.8-flash",
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": ReviewResult,
        },
    )


    # Gemini returns the structured Pydantic object here
    result = response.parsed


    if result is None:
        raise RuntimeError(
            "AI returned no structured review result"
        )


    return result.model_dump()


# ============================================================
# Local Test
# ============================================================

if __name__ == "__main__":

    result = analyse_pull_request(
        pull_request_info={
            "title": "Test PR",
            "description": "Testing Gemini integration",
        },
        changes=[
            {
                "file": "UserService.java",
                "patch": """
+ public User getUser(Long id) {
+     return userRepository.findById(id).get();
+ }
""",
            }
        ],
    )

    print(result)