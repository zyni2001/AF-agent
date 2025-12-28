"""Autoformalization white agent - LLM generates Z3 Python code and executes it."""

import uvicorn
import os
import sys
import tempfile
import subprocess
import asyncio
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentSkill, AgentCard, AgentCapabilities
from a2a.utils import new_agent_text_message
from litellm import completion
from litellm.exceptions import RateLimitError

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


# Get Gemini API keys from environment
def get_api_keys_from_env():
    """Load API keys from GEMINI_API_KEY environment variable."""
    api_key = os.environ.get('GEMINI_API_KEY', '')
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set")
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
        id="z3_logical_reasoning",
        name="Z3 Autoformalization",
        description="Generates executable Z3 Python code to solve logical reasoning problems",
        tags=["reasoning", "logic", "autoformalization", "z3"],
        examples=[],
    )
    card = AgentCard(
        name="z3_autoformalization_agent",
        description="Autoformalization agent that generates and executes Z3 Python code for logical reasoning",
        url=url,
        version="1.0.0",
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        capabilities=AgentCapabilities(),
        skills=[skill],
    )
    return card


class AutoformWhiteAgentExecutor(AgentExecutor):
    def __init__(self):
        self.ctx_id_to_messages = {}

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Execute autoformalization reasoning using Z3 code generation with self-refinement."""
        
        try:
            # Get user input
            user_input = context.get_user_input()
            
            print(f"\n{'='*60}")
            print(f"Autoform agent: Processing request")
            print(f"{'='*60}\n")
            
            # Self-refinement parameters
            max_repairs = 3
            z3_code = None
            execution_error = None
            original_user_input = user_input  # Keep original for repair prompts
            
            # Create initial system prompt for Z3 code generation
            system_prompt = """You are an expert in formal logic and Z3 SMT solver.

Given a logical reasoning problem with premises and a conclusion, generate executable Python code using the Z3 library that:
1. Encodes the premises and conclusion as Z3 formulas
2. Checks if the conclusion logically follows from the premises
3. Prints EXACTLY one of: "True", "False", or "Uncertain"

Requirements:
- Use the z3 library (import from z3 import *)
- Define all predicates and constants appropriately
- Check satisfiability to determine the answer:
  * If premises + ¬conclusion is UNSAT → print "True"
  * If premises + conclusion is UNSAT → print "False"  
  * Otherwise → print "Uncertain"
- The last line of output must be EXACTLY "True", "False", or "Uncertain"
- Do NOT include any explanation, only the executable code

CRITICAL Z3 SYNTAX RULES (common mistakes to avoid):
- Use ForAll (capital A), NOT Forall
- Use Exists (capital E), NOT Exists  
- There is NO Iff function in Z3. For biconditional, use: And(Implies(a, b), Implies(b, a))
- Always declare quantifier variables with Const() before using them in ForAll/Exists

Example structure:
```python
from z3 import *

# Define sorts and predicates
Object = DeclareSort('Object')
P = Function('P', Object, BoolSort())
# ... more definitions

# Define constants
x = Const('x', Object)

# Encode premises
premises = [
    # your premises here
]

# Encode conclusion
conclusion = # your conclusion here

# Check: premises + ¬conclusion
solver1 = Solver()
for p in premises:
    solver1.add(p)
solver1.add(Not(conclusion))

if solver1.check() == unsat:
    print("True")
elif solver1.check() == sat:
    # Check premises + conclusion
    solver2 = Solver()
    for p in premises:
        solver2.add(p)
    solver2.add(conclusion)
    if solver2.check() == unsat:
        print("False")
    else:
        print("Uncertain")
else:
    print("Uncertain")
```

Now generate the code for the following problem:"""

            # Self-refinement loop
            for repair_attempt in range(max_repairs + 1):  # 0 to max_repairs (inclusive)
                is_repair = repair_attempt > 0
                
                if is_repair:
                    if execution_error is None:
                        execution_error = "Previous code execution failed"
                    
                    print(f"\n{'='*60}")
                    print(f"Repair attempt {repair_attempt}/{max_repairs}")
                    print(f"{'='*60}\n")
                    
                    # Create repair prompt
                    repair_system_prompt = """You are an expert in fixing Python Z3 code syntax errors.

The previous Z3 code you generated had syntax errors. Please fix the code based on the error message.

CRITICAL Z3 SYNTAX FIXES:
- Use ForAll (capital A) for universal quantification, NOT Forall
- Use Exists (capital E) for existential quantification, NOT Exists
- There is NO Iff function in Z3! For biconditional, use: And(Implies(a, b), Implies(b, a))
- Ensure all quantifier variables are declared with Const() before use in ForAll/Exists
- Check that all function names are correct (e.g., Implies, And, Or, Not, etc.)

Generate the FIXED executable Z3 Python code that:
1. Fixes all syntax errors
2. Encodes the premises and conclusion correctly
3. Prints EXACTLY one of: "True", "False", or "Uncertain"
- Do NOT include any explanation, only the fixed executable code"""

                    repair_user_prompt = f"""Original problem:
{original_user_input}

Previous code that failed:
```python
{z3_code}
```

Error message:
{execution_error}

Please generate the FIXED Z3 Python code:"""
                    
                    messages = [
                        {"role": "system", "content": repair_system_prompt},
                        {"role": "user", "content": repair_user_prompt}
                    ]
                else:
                    # Initial generation
                    user_prompt = f"{user_input}\n\nGenerate the complete executable Z3 Python code:"
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                
                # Call LLM to generate/fix Z3 code
                if is_repair:
                    print("Generating fixed Z3 Python code...")
                else:
                    print("Generating Z3 Python code...")
                
                response = None
                last_error = None
                max_rate_limit_retries = 5
                base_delay = 7  # Base delay in seconds (to stay under 10 req/min)
                
                for rate_retry in range(max_rate_limit_retries):
                for api_attempt in range(len(GEMINI_API_KEYS)):
                    try:
                        api_key = get_next_api_key()
                        os.environ['GEMINI_API_KEY'] = api_key
                        
                        response = completion(
                            messages=messages,
                                model="gemini/gemini-2.5-flash",
                            api_key=api_key,
                            temperature=0.0
                        )
                        break
                        except RateLimitError as e:
                            last_error = e
                            retry_delay = base_delay * (2 ** rate_retry)
                            print(f"Autoform agent: Rate limited (retry {rate_retry + 1}/{max_rate_limit_retries}), waiting {retry_delay}s...")
                            await asyncio.sleep(retry_delay)
                            break  # Break inner loop to retry with backoff
                    except Exception as e:
                        last_error = e
                            print(f"Autoform agent: API call failed (key {api_attempt + 1}/{len(GEMINI_API_KEYS)}): {type(e).__name__}: {str(e)[:100]}")
                        continue
                    
                    if response is not None:
                        break
                
                if response is None:
                    error_msg = f"ERROR: All retries failed. Last error: {last_error}"
                    print(f"Autoform agent: {error_msg}")
                    await event_queue.enqueue_event(
                        new_agent_text_message("Uncertain", context_id=context.context_id)
                    )
                    return
                
                z3_code = response.choices[0].message.content
                if z3_code is None:
                    print("Autoform agent: ERROR: LLM returned None content")
                    await event_queue.enqueue_event(
                        new_agent_text_message("Uncertain", context_id=context.context_id)
                    )
                    return
                
                # Extract code from markdown if present
                if "```python" in z3_code:
                    z3_code = z3_code.split("```python")[1].split("```")[0].strip()
                elif "```" in z3_code:
                    z3_code = z3_code.split("```")[1].split("```")[0].strip()
                
                if is_repair:
                    print(f"\nFixed Z3 code ({len(z3_code)} chars)")
                else:
                    print(f"\nGenerated Z3 code ({len(z3_code)} chars)")
                print(f"First 200 chars: {z3_code[:200]}...\n")
                
                # Execute the Z3 code
                print("Executing Z3 code...")
                
                temp_file = None
                try:
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                        f.write(z3_code)
                        temp_file = f.name
                    
                    result = subprocess.run(
                        ['python', temp_file],
                        capture_output=True,
                        text=True,
                        timeout=60  # 60 second timeout
                    )
                    
                    output = result.stdout.strip()
                    stderr = result.stderr.strip()
                    
                    print(f"Execution return code: {result.returncode}")
                    if output:
                        print(f"Stdout: {output[:200]}")
                    if stderr:
                        print(f"Stderr: {stderr[:500]}")
                    
                    # If execution succeeded, parse result and exit loop
                    if result.returncode == 0:
                        # Parse the result from the last line
                        lines = output.split('\n') if output else []
                        answer = "Uncertain"
                        
                        for line in reversed(lines):
                            line = line.strip()
                            if line in ["True", "False", "Uncertain"]:
                                answer = line
                                break
                        
                        print(f"\n{'='*60}")
                        print(f"Final answer: {answer}")
                        print(f"{'='*60}\n")
                        
                        # Send response
                        await event_queue.enqueue_event(
                            new_agent_text_message(answer, context_id=context.context_id)
                        )
                        return  # Success, exit the refinement loop
                    
                    # Execution failed - prepare for repair
                    execution_error = stderr if stderr else f"Execution failed with return code {result.returncode}"
                    print(f"Warning: Z3 code execution failed with return code {result.returncode}")
                    print(f"Error: {execution_error[:500]}")
                    
                    # If we've exhausted repair attempts, return Uncertain
                    if repair_attempt >= max_repairs:
                        print(f"\nMax repair attempts ({max_repairs}) reached. Returning Uncertain.")
                        await event_queue.enqueue_event(
                            new_agent_text_message("Uncertain", context_id=context.context_id)
                        )
                        return
                    
                    # Continue to next repair attempt
                    print(f"\nAttempting repair {repair_attempt + 1}/{max_repairs}...")
                    
                finally:
                    # Clean up temp file
                    if temp_file and os.path.exists(temp_file):
                        os.unlink(temp_file)
            
            # If we somehow exit the loop without returning, return Uncertain
            print("ERROR: Exited refinement loop without result")
            await event_queue.enqueue_event(
                new_agent_text_message("Uncertain", context_id=context.context_id)
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
    print("Starting autoformalization white agent (LLM→Z3 Code→Execute)...")
    
    print(f"Environment variables for URL resolution:")
    print(f"  AGENT_URL: {os.environ.get('AGENT_URL', 'not set')}")
    print(f"  A2A_AGENT_URL: {os.environ.get('A2A_AGENT_URL', 'not set')}")
    print(f"  PUBLIC_URL: {os.environ.get('PUBLIC_URL', 'not set')}")
    
    # URL priority (for AgentBeats controller compatibility):
    # 1. AGENT_URL - set by earthshaker controller with full /to_agent/{cagent_id} path
    # 2. A2A_AGENT_URL - alternative variable name
    # 3. PUBLIC_URL - base controller URL (for Cloud Run)
    # 4. Local URL - for local development
    agent_url = os.environ.get("AGENT_URL")
    a2a_url = os.environ.get("A2A_AGENT_URL")
    public_url = os.environ.get("PUBLIC_URL")
    
    if agent_url:
        url = agent_url
        print(f"Using AGENT_URL from controller: {url}")
    elif a2a_url:
        url = a2a_url
        print(f"Using A2A_AGENT_URL from controller: {url}")
    elif public_url:
        url = public_url
        print(f"Using PUBLIC_URL from environment: {url}")
    else:
        url = f"http://{host}:{port}"
        print(f"Using local URL: {url}")
    
    card = prepare_white_agent_card(url)

    request_handler = DefaultRequestHandler(
        agent_executor=AutoformWhiteAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )

    # Create Starlette application
    app_builder = A2AStarletteApplication(
        agent_card=card,
        http_handler=request_handler,
    )
    
    starlette_app = app_builder.build()
    
    # Add health check endpoint for AgentBeats
    from starlette.responses import JSONResponse
    from starlette.routing import Route
    
    async def health_check(request):
        return JSONResponse({"status": "ok", "agent": "z3_autoformalization_agent"})
    
    starlette_app.routes.append(Route("/status", health_check))

    uvicorn.run(starlette_app, host=host, port=port)
