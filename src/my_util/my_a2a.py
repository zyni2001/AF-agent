import httpx
import asyncio
import uuid


from a2a.client import A2ACardResolver, A2AClient
from a2a.types import (
    AgentCard,
    Part,
    TextPart,
    MessageSendParams,
    Message,
    Role,
    SendMessageRequest,
    SendMessageResponse,
)


async def get_agent_card(url: str) -> AgentCard | None:
    """
    Get agent card from URL.
    
    In Cloud Run/AgentBeats environment, the URL should be the controller's proxy path:
    https://controller-url/to_agent/{agent_id}/.well-known/agent-card.json
    
    The A2ACardResolver will handle the path resolution automatically.
    """
    httpx_client = httpx.AsyncClient(timeout=30.0)  # Increase timeout for Cloud Run
    resolver = A2ACardResolver(httpx_client=httpx_client, base_url=url)

    try:
        card: AgentCard | None = await resolver.get_agent_card()
        return card
    except Exception as e:
        print(f"Error getting agent card from {url}: {e}")
        return None


async def wait_agent_ready(url, timeout=30):
    """
    Wait until the A2A server is ready, check by getting the agent card.
    
    Increased default timeout to 30 seconds to account for:
    - Agent process startup time after reset
    - Cloud Run cold starts
    - Network latency
    """
    retry_cnt = 0
    while retry_cnt < timeout:
        retry_cnt += 1
        try:
            card = await get_agent_card(url)
            if card is not None:
                print(f"Agent is ready after {retry_cnt} attempts")
                return True
            else:
                print(
                    f"Agent card not available yet..., retrying {retry_cnt}/{timeout}"
                )
        except Exception as e:
            print(f"Error checking agent readiness (attempt {retry_cnt}/{timeout}): {e}")
        await asyncio.sleep(1)
    print(f"Agent not ready after {timeout} seconds")
    return False


async def send_message(
    url, message, task_id=None, context_id=None, timeout=3600.0
) -> SendMessageResponse:
    """Send message to an A2A agent.
    
    Args:
        timeout: Request timeout in seconds. Default 3600 (1 hour) for long evaluations.
    """
    card = await get_agent_card(url)
    httpx_client = httpx.AsyncClient(timeout=timeout)
    client = A2AClient(httpx_client=httpx_client, agent_card=card)

    message_id = uuid.uuid4().hex
    params = MessageSendParams(
        message=Message(
            role=Role.user,
            parts=[Part(TextPart(text=message))],
            message_id=message_id,
            task_id=task_id,
            context_id=context_id,
        )
    )
    request_id = uuid.uuid4().hex
    req = SendMessageRequest(id=request_id, params=params)
    response = await client.send_message(request=req)
    return response
