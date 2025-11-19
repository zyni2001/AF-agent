"""Autoformalization white agent - LLM converts NL to FOL, then Vampire verifies."""

import uvicorn
import os
import re
import sys
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentSkill, AgentCard, AgentCapabilities
from a2a.utils import new_agent_text_message
from litellm import completion

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.folio_utils.vampire_runner import check_folio_with_vampire
from src.folio_utils.dataset import parse_premises


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
        id="autoformalization_reasoning",
        name="Autoformalization + Logical Verification",
        description="Converts natural language to FOL and uses Vampire theorem prover for verification",
        tags=["reasoning", "logic", "autoformalization", "theorem-proving"],
        examples=[],
    )
    card = AgentCard(
        name="autoform_reasoning_agent",
        description="Agent that automates formalization (NL→FOL) and uses theorem provers for logical verification",
        url=url,
        version="1.0.0",
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        capabilities=AgentCapabilities(),
        skills=[skill],
    )
    return card


def extract_premises_and_conclusion(nl_text: str) -> tuple:
    """Extract premises and conclusion from natural language problem text.
    
    Returns:
        (premises_text, conclusion_text)
    """
    # Look for "Given the following premises:" and "Does the following conclusion"
    premises_match = re.search(r'Given the following premises:\s*(.*?)\s*Does the following conclusion', 
                               nl_text, re.DOTALL | re.IGNORECASE)
    conclusion_match = re.search(r'Conclusion:\s*(.*?)\s*(?:Please answer|Your answer)', 
                                 nl_text, re.DOTALL | re.IGNORECASE)
    
    if premises_match and conclusion_match:
        premises_text = premises_match.group(1).strip()
        conclusion_text = conclusion_match.group(1).strip()
        return premises_text, conclusion_text
    
    # Fallback: try simpler patterns
    parts = nl_text.split("Conclusion:")
    if len(parts) >= 2:
        premises_text = parts[0].strip()
        conclusion_text = parts[1].split("Please answer")[0].split("Your answer")[0].strip()
        return premises_text, conclusion_text
    
    return None, None


def parse_fol_response(llm_response: str) -> tuple:
    """Parse LLM response to extract premises-FOL and conclusion-FOL.
    
    Returns:
        (premises_fol_list, conclusion_fol)
    """
    # Look for PREMISES-FOL: and CONCLUSION-FOL: sections
    premises_match = re.search(r'PREMISES-FOL:\s*(.*?)\s*CONCLUSION-FOL:', 
                               llm_response, re.DOTALL | re.IGNORECASE)
    conclusion_match = re.search(r'CONCLUSION-FOL:\s*(.*?)(?:\n\n|\Z)', 
                                 llm_response, re.DOTALL | re.IGNORECASE)
    
    if premises_match and conclusion_match:
        premises_text = premises_match.group(1).strip()
        conclusion_text = conclusion_match.group(1).strip()
        
        # Split premises by newlines
        premises_list = [p.strip() for p in premises_text.split('\n') if p.strip()]
        
        return premises_list, conclusion_text
    
    return None, None


class AutoformWhiteAgentExecutor(AgentExecutor):
    def __init__(self):
        self.ctx_id_to_messages = {}

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        # Get user input
        user_input = context.get_user_input()
        
        print(f"\n{'='*60}")
        print("Autoform agent: Processing request")
        print(f"{'='*60}\n")
        
        try:
            # Extract premises and conclusion from natural language
            premises_nl, conclusion_nl = extract_premises_and_conclusion(user_input)
            
            if not premises_nl or not conclusion_nl:
                error_msg = "ERROR: Could not extract premises and conclusion from input"
                print(f"Autoform agent: {error_msg}")
                print(f"Autoform agent: Returning Uncertain due to extraction failure")
                await event_queue.enqueue_event(
                    new_agent_text_message("Uncertain", context_id=context.context_id)
                )
                return
            
            print(f"Extracted premises (NL):\n{premises_nl[:200]}...")
            print(f"\nExtracted conclusion (NL):\n{conclusion_nl[:200]}...")
            
            # Stage 1: Autoformalization (LLM converts NL → FOL)
            print("\n--- Stage 1: Autoformalization (NL → FOL) ---")
            
            autoform_prompt = f"""Convert the following natural language premises and conclusion into First-Order Logic (FOL) format.

Be precise and use standard FOL syntax:
- Use ∀ for universal quantification, ∃ for existential quantification
- Use ∧ for AND, ∨ for OR, → for implication, ¬ for negation
- Use predicates like Person(x), Loves(x,y), etc.

Premises:
{premises_nl}

Conclusion:
{conclusion_nl}

Provide your answer in EXACTLY this format (do not add any other text):

PREMISES-FOL:
<premise1 in FOL>
<premise2 in FOL>
...

CONCLUSION-FOL:
<conclusion in FOL>"""

            # Call LLM with retry logic
            response = None
            last_error = None
            
            for attempt in range(len(GEMINI_API_KEYS)):
                try:
                    api_key = get_next_api_key()
                    os.environ['GEMINI_API_KEY'] = api_key
                    
                    response = completion(
                        messages=[{"role": "user", "content": autoform_prompt}],
                        model="gemini/gemini-2.5-flash",
                        temperature=0.0,
                        max_tokens=2000,
                    )
                    break
                except Exception as e:
                    last_error = e
                    print(f"Autoform agent: API call failed (attempt {attempt + 1}/{len(GEMINI_API_KEYS)}): {e}")
                    continue
            
            if response is None:
                error_msg = f"ERROR: All API keys failed during autoformalization. Last error: {last_error}"
                print(f"Autoform agent: {error_msg}")
                await event_queue.enqueue_event(
                    new_agent_text_message("Uncertain", context_id=context.context_id)
                )
                return
            
            # Extract FOL formulas from response
            fol_response = response.choices[0].message.content
            
            # Handle None response
            if fol_response is None:
                error_msg = "ERROR: LLM returned None content"
                print(f"Autoform agent: {error_msg}")
                await event_queue.enqueue_event(
                    new_agent_text_message("Uncertain", context_id=context.context_id)
                )
                return
            
            print(f"\nLLM autoformalization response:\n{fol_response[:500]}...")
            
            premises_fol, conclusion_fol = parse_fol_response(fol_response)
            
            if not premises_fol or not conclusion_fol:
                error_msg = "ERROR: Could not parse FOL formulas from LLM response"
                print(f"Autoform agent: {error_msg}")
                print(f"Full response: {fol_response}")
                await event_queue.enqueue_event(
                    new_agent_text_message("Uncertain", context_id=context.context_id)
                )
                return
            
            print(f"\nParsed {len(premises_fol)} premises in FOL")
            print(f"Conclusion in FOL: {conclusion_fol[:100]}...")
            
            # Stage 2: Logical Verification (Vampire)
            print("\n--- Stage 2: Logical Verification (Vampire) ---")
            
            vampire_path = os.environ.get('VAMPIRE_PATH')
            if not vampire_path:
                # Try default location
                vampire_path = '/home/argustest/logic-reasoning-workspace/zhiyu/folio-agent/folio_correction/vampire/build/vampire'
            
            print(f"Running Vampire theorem prover...")
            print(f"Vampire path: {vampire_path}")
            
            result = check_folio_with_vampire(
                premises_fol=premises_fol,
                conclusion_fol=conclusion_fol,
                time_limit=50,
                vampire_path=vampire_path
            )
            
            print(f"Vampire status: {result['vampire_status']}")
            
            # Check for parse errors
            if result.get('has_parse_error'):
                error_msg = f"PARSE ERROR in TPTP: {result.get('parse_error_msg', 'Unknown error')}"
                print(f"Autoform agent: {error_msg}")
                await event_queue.enqueue_event(
                    new_agent_text_message("Uncertain", context_id=context.context_id)
                )
                return
            
            # Determine final answer
            predicted_label = result['predicted_label']
            
            if predicted_label is True:
                answer = "True"
                print("✓ Vampire verdict: Conclusion FOLLOWS from premises")
            elif predicted_label is False:
                answer = "False"
                print("✗ Vampire verdict: Conclusion does NOT follow from premises")
            else:
                answer = "Uncertain"
                print("? Vampire verdict: Cannot determine (timeout or insufficient information)")
            
            print(f"\n{'='*60}")
            print(f"Final answer: {answer}")
            print(f"{'='*60}\n")
            
            # Send response
            await event_queue.enqueue_event(
                new_agent_text_message(answer, context_id=context.context_id)
            )
            
        except Exception as e:
            # Catch-all for any unexpected errors
            error_msg = f"ERROR: Unexpected exception in autoform agent: {e}"
            print(f"Autoform agent: {error_msg}")
            import traceback
            traceback.print_exc()
            await event_queue.enqueue_event(
                new_agent_text_message("Uncertain", context_id=context.context_id)
            )

    async def cancel(self, context, event_queue) -> None:
        raise NotImplementedError


def start_autoform_white_agent(host="localhost", port=9003):
    print("Starting autoformalization white agent (LLM→FOL→Vampire)...")
    url = f"http://{host}:{port}"
    card = prepare_white_agent_card(url)

    request_handler = DefaultRequestHandler(
        agent_executor=AutoformWhiteAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )

    app = A2AStarletteApplication(
        agent_card=card,
        http_handler=request_handler,
    )

    uvicorn.run(app.build(), host=host, port=port)

