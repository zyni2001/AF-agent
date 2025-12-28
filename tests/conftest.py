"""Pytest configuration for agent tests."""

import pytest


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--agent-url",
        action="store",
        default="http://localhost:9001",
        help="URL of the agent to test"
    )


@pytest.fixture
def agent_url(request):
    """Fixture to get the agent URL from command line."""
    return request.config.getoption("--agent-url")

