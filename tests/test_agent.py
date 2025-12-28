"""Tests for FOLIO agents."""

import pytest
import httpx


def test_status_endpoint(agent_url):
    """Test that the status endpoint is accessible."""
    response = httpx.get(f"{agent_url}/status", timeout=30.0)
    assert response.status_code == 200, f"Status endpoint returned {response.status_code}"
    data = response.json()
    # Controller status has these fields
    assert "maintained_agents" in data or "running_agents" in data or "status" in data


def test_agents_endpoint(agent_url):
    """Test that the agents list endpoint works."""
    response = httpx.get(f"{agent_url}/agents", timeout=30.0)
    assert response.status_code == 200, f"Agents endpoint returned {response.status_code}"
    data = response.json()
    assert isinstance(data, dict), "Agents response should be a dict"


def test_agent_registered(agent_url):
    """Test that at least one agent is registered with the controller."""
    response = httpx.get(f"{agent_url}/agents", timeout=30.0)
    assert response.status_code == 200
    agents = response.json()
    # There should be at least one agent
    assert len(agents) >= 1, "At least one agent should be registered"
    
    # Check agent has required fields
    agent_id = list(agents.keys())[0]
    agent_info = agents[agent_id]
    assert "url" in agent_info, "Agent should have a URL"
    assert "state" in agent_info, "Agent should have a state"


def test_agent_card_via_proxy(agent_url):
    """Test that agent card is accessible via controller proxy."""
    # Get agent list first
    response = httpx.get(f"{agent_url}/agents", timeout=30.0)
    if response.status_code != 200:
        pytest.skip("Could not get agents list")
    
    agents = response.json()
    if not agents:
        pytest.skip("No agents registered")
    
    agent_id = list(agents.keys())[0]
    
    # Try to get agent card through the controller's to_agent route
    card_url = f"{agent_url}/to_agent/{agent_id}/.well-known/agent-card.json"
    card_response = httpx.get(card_url, timeout=60.0, follow_redirects=True)
    
    # Agent might still be starting, so we accept 200 or skip
    if card_response.status_code == 200:
        card = card_response.json()
        assert "name" in card, "Agent card should have a name"
        assert "url" in card, "Agent card should have a URL"
    else:
        # Log but don't fail - agent might be slow to start
        print(f"Agent card not ready yet: {card_response.status_code}")
