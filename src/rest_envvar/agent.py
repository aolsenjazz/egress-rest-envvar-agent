"""Hallowell employee-directory assistant: the API base URL comes from the
HALLOWELL_DIRECTORY_BASE_URL environment variable, never from source.

Egress under test: REST requests whose host is only knowable from the
deployment's environment. With no literal in source, identification must come
from the approved env plan / configured values.
"""

from __future__ import annotations

import os

import httpx
from langchain.agents import create_agent
from langchain_core.tools import tool


def _base_url() -> str:
    base = os.environ.get("HALLOWELL_DIRECTORY_BASE_URL")
    if not base:
        raise RuntimeError("HALLOWELL_DIRECTORY_BASE_URL is not configured")
    return base.rstrip("/")


def _client() -> httpx.Client:
    return httpx.Client(
        base_url=_base_url(),
        headers={"X-Api-Key": os.environ.get("HALLOWELL_DIRECTORY_API_KEY", "unset")},
        timeout=30,
    )


@tool
def get_employee(employee_id: str) -> dict:
    """Fetch one employee record from the directory by ID."""
    with _client() as client:
        response = client.get(f"/users/{employee_id}")
        response.raise_for_status()
        return response.json()


@tool
def list_team_members(team_slug: str) -> list[dict]:
    """List the members of a team in the directory."""
    with _client() as client:
        response = client.get("/users")
        response.raise_for_status()
        return response.json()


@tool
def update_employee_title(employee_id: str, new_title: str) -> dict:
    """Update an employee's job title in the directory."""
    with _client() as client:
        response = client.patch(f"/users/{employee_id}", json={"title": new_title})
        response.raise_for_status()
        return response.json()


SYSTEM_PROMPT = """You are an HR operations assistant for Hallowell.
Use the directory tools to answer questions about employees and teams and to
apply requested title changes. Confirm the employee ID in every answer."""

agent = create_agent(
    "openai:gpt-4o-mini",
    tools=[get_employee, list_team_members, update_employee_title],
    system_prompt=SYSTEM_PROMPT,
)
