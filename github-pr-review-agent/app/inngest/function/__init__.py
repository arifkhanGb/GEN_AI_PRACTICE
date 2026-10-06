"""Inngest function registrations exposed to the FastAPI application."""

from .function import hello_world
from .github_pr_review import github_pr_review

__all__ = ["github_pr_review", "hello_world"]
