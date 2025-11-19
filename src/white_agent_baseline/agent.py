"""Baseline white agent - direct LLM reasoning without formal logic."""

import uvicorn
import os
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentSkill, AgentCard, AgentCapabilities
from a2a.utils import new_agent_text_message
from litellm import completion


# Get Gemini API keys from environment (comma-separated for rotation)
def get_api_keys_from_env():
    """Load API keys from GEMINI_API_KEY environment variable."""
    api_key = os.environ.get('GEMINI_API_KEY', '')
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set")
    # Support comma-separated keys for rotation
    return [key.strip() for key in api_key.split(',') if key.strip()]

GEMINI_API_KEYS = get_api_keys_from_env()
current_key_index = 0


def get_next_api_key():
    """Get next API key in rotation."""
    global current_key_index
    key = GEMINI_API_KEYS[current_key_index]
    current_key_index = (current_key_index + 1) % len(GEMINI_API_KEYS)
    return key


def prepare_white_agent_card(url):
    skill = AgentSkill(
        id="logical_reasoning",
        name="Logical Reasoning",
        description="Determines if conclusions follow from premises using direct LLM reasoning",
        tags=["reasoning", "logic", "baseline"],
        examples=[],
    )
    card = AgentCard(
        name="baseline_reasoning_agent",
        description="Baseline agent that uses direct LLM reasoning for logical inference",
        url=url,
        version="1.0.0",
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        capabilities=AgentCapabilities(),
        skills=[skill],
    )
    return card


class BaselineWhiteAgentExecutor(AgentExecutor):
    def __init__(self):
        self.ctx_id_to_messages = {}

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        # Get user input
        user_input = context.get_user_input()
        
        # Initialize message history for this context
        if context.context_id not in self.ctx_id_to_messages:
            self.ctx_id_to_messages[context.context_id] = []
        
        messages = self.ctx_id_to_messages[context.context_id]
        
        # For FOLIO problems, we want a system prompt that encourages careful reasoning
        if not messages:
            system_prompt = """You are a logical reasoning expert. You will be given premises and a conclusion. 
Your task is to determine whether the conclusion logically follows from the premises.

CRITICAL: Your response must be EXACTLY one word: True, False, or Uncertain
Do NOT add any explanation, reasoning, or additional text.
Just output the single word answer.

- True: The conclusion necessarily follows from the premises
- False: The conclusion does not follow from the premises or contradicts them
- Uncertain: It cannot be determined from the given premises"""
            
            messages.append({
                "role": "system",
                "content": system_prompt,
            })
        
        messages.append({
            "role": "user",
            "content": user_input,
        })
        
        # Call LLM with retry logic for API key rotation
        response = None
        last_error = None
        
        for attempt in range(len(GEMINI_API_KEYS)):
            try:
                api_key = get_next_api_key()
                os.environ['GEMINI_API_KEY'] = api_key
                
                response = completion(
                    messages=messages,
                    model="gemini/gemini-2.5-flash",
                    temperature=0.0,
                    max_tokens=500,
                )
                break
            except Exception as e:
                last_error = e
                print(f"Baseline agent: API call failed (attempt {attempt + 1}/{len(GEMINI_API_KEYS)}): {e}")
                continue
        
        if response is None:
            error_msg = f"ERROR: All API keys failed. Last error: {last_error}"
            print(f"Baseline agent: {error_msg}")
            await event_queue.enqueue_event(
                new_agent_text_message(error_msg, context_id=context.context_id)
            )
            return
        
        # Extract response
        next_message = response.choices[0].message.model_dump()
        assistant_content = next_message.get("content", "")
        
        # Handle empty or None content
        if not assistant_content or assistant_content is None:
            assistant_content = "Uncertain"
            print("Baseline agent: Warning - LLM returned empty content, defaulting to 'Uncertain'")
        
        messages.append({
            "role": "assistant",
            "content": assistant_content,
        })
        
        # Send response
        await event_queue.enqueue_event(
            new_agent_text_message(
                assistant_content, context_id=context.context_id
            )
        )

    async def cancel(self, context, event_queue) -> None:
        raise NotImplementedError


def start_baseline_white_agent(host="localhost", port=9002):
    print("Starting baseline white agent (direct LLM)...")
    
    # Use public URL from environment if available (for Cloud Run)
    public_url = os.environ.get("PUBLIC_URL")
    if public_url:
        url = public_url
        print(f"Using public URL from environment: {url}")
    else:
        url = f"http://{host}:{port}"
        print(f"Using local URL: {url}")
    
    card = prepare_white_agent_card(url)

    request_handler = DefaultRequestHandler(
        agent_executor=BaselineWhiteAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )

    app = A2AStarletteApplication(
        agent_card=card,
        http_handler=request_handler,
    )
    
    # Build the Starlette app and add health check endpoint
    starlette_app = app.build()
    
    # Add /status health check endpoint for AgentBeats
    from starlette.responses import JSONResponse
    from starlette.routing import Route
    
    async def health_check(request):
        return JSONResponse({"status": "ok", "agent": "baseline_reasoning_agent"})
    
    # Add the route to the existing app
    starlette_app.routes.append(Route("/status", health_check))

    uvicorn.run(starlette_app, host=host, port=port)

