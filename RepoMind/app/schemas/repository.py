from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator


class RepositoryIndexRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    repository_url: HttpUrl = Field(
        description="Public GitHub repository URL",
        examples=["https://github.com/owner/repository"],
    )

    @field_validator("repository_url")
    @classmethod
    def validate_github_repository_url(cls, value: HttpUrl) -> HttpUrl:
        if value.host not in {"github.com", "www.github.com"}:
            raise ValueError("Only GitHub repository URLs are supported")

        path_parts = value.path.strip("/").split("/")

        if len(path_parts) != 2 or not all(path_parts):
            raise ValueError("URL must point to a repository: /owner/repository")

        path_parts[1] = path_parts[1].removesuffix(".git")

        if not path_parts[1]:
            raise ValueError("Repository name cannot be empty")

        return value
