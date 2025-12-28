"""Green agent implementation - FOLIO benchmark evaluator."""

import uvicorn
import tomllib
import json
import time
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCard, SendMessageSuccessResponse, Message, JSONRPCErrorResponse
from a2a.utils import new_agent_text_message, get_text_parts

import sys
import os
# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.my_util import parse_tags, my_a2a
from src.folio_utils.dataset import load_validation_dataset, format_folio_problem_as_text


def load_agent_card_toml(agent_name):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    toml_path = os.path.join(current_dir, f"{agent_name}.toml")
    with open(toml_path, "rb") as f:
        return tomllib.load(f)


async def evaluate_white_agent_on_folio(white_agent_url: str, max_examples: int = None) -> dict:
    """Evaluate a white agent on FOLIO validation dataset.
    
    Args:
        white_agent_url: URL of the white agent to evaluate
        max_examples: If provided, only test on first N examples
    
    Returns:
        dict with evaluation metrics
    """
    print(f"\n{'='*60}")
    print(f"Starting FOLIO Evaluation")
    print(f"White agent: {white_agent_url}")
    print(f"{'='*60}\n")
    
    # Load validation dataset
    df = load_validation_dataset(max_examples=max_examples)
    
    metrics = {
        "total_cases": 0,
        "correct": 0,
        "incorrect": 0,
        "parse_errors": 0,
        "total_time": 0.0,
        "results": []
    }
    
    for idx, row in df.iterrows():
        metrics["total_cases"] += 1
        story_id = row.get('story_id', idx)
        expected_label = str(row['label']).strip()
        
        print(f"\n[{metrics['total_cases']}/{len(df)}] Story ID: {story_id}")
        print(f"  Expected: {expected_label}")
        
        # Format problem as text
        problem_text = format_folio_problem_as_text(row)
        
        # Send to white agent
        try:
            start_time = time.time()
            
            print(f"  Sending to white agent...")
            response = await my_a2a.send_message(white_agent_url, problem_text)
            
            elapsed_time = time.time() - start_time
            metrics["total_time"] += elapsed_time
            
            # Parse response
            res_root = response.root
            
            # Handle error responses from white agent
            if isinstance(res_root, JSONRPCErrorResponse):
                error_msg = res_root.error.message if hasattr(res_root, 'error') else "Unknown error"
                print(f"  ✗ ERROR: White agent returned error: {error_msg}")
                # Treat agent errors as "Uncertain" - can't determine the answer
                predicted_label = "Uncertain"
                is_correct = (expected_label.lower() == "uncertain")
                if is_correct:
                    metrics["correct"] += 1
                    print(f"  ✓ CORRECT (error treated as Uncertain)")
                else:
                    metrics["incorrect"] += 1
                    print(f"  ✗ INCORRECT (expected {expected_label}, got Uncertain due to error)")
                
                metrics["results"].append({
                    "story_id": story_id,
                    "expected": expected_label,
                    "predicted": "Uncertain",
                    "correct": is_correct,
                    "time": elapsed_time,
                    "error": error_msg
                })
                continue
            
            if not isinstance(res_root, SendMessageSuccessResponse):
                print(f"  ✗ ERROR: Unexpected response type: {type(res_root)}")
                metrics["parse_errors"] += 1
                metrics["results"].append({
                    "story_id": story_id,
                    "expected": expected_label,
                    "predicted": "ERROR",
                    "correct": False,
                    "time": elapsed_time,
                    "error": f"Invalid response type: {type(res_root)}"
                })
                continue
            
            res_result = res_root.result
            if not isinstance(res_result, Message):
                print(f"  ✗ ERROR: Expected Message result")
                metrics["parse_errors"] += 1
                metrics["results"].append({
                    "story_id": story_id,
                    "expected": expected_label,
                    "predicted": "ERROR",
                    "correct": False,
                    "time": elapsed_time,
                    "error": "Invalid result type"
                })
                continue
            
            # Extract text from response
            text_parts = get_text_parts(res_result.parts)
            if not text_parts:
                print(f"  ✗ ERROR: No text in response")
                metrics["parse_errors"] += 1
                metrics["results"].append({
                    "story_id": story_id,
                    "expected": expected_label,
                    "predicted": "ERROR",
                    "correct": False,
                    "time": elapsed_time,
                    "error": "No text in response"
                })
                continue
            
            white_text = text_parts[0].strip()
            print(f"  White agent response: {white_text[:100]}...")
            
            # Parse the answer (look for True/False/Uncertain)
            white_text_lower = white_text.lower().strip()
            predicted_label = None
            
            # First check if response is exactly one of the labels
            if white_text_lower == "true":
                predicted_label = "True"
            elif white_text_lower == "false":
                predicted_label = "False"
            elif white_text_lower == "uncertain":
                predicted_label = "Uncertain"
            # Otherwise try to extract from text
            elif "uncertain" in white_text_lower:
                predicted_label = "Uncertain"
            elif "true" in white_text_lower and "false" not in white_text_lower:
                predicted_label = "True"
            elif "false" in white_text_lower and "true" not in white_text_lower:
                predicted_label = "False"
            else:
                # Try to find the first occurrence of any label
                for label in ["True", "False", "Uncertain"]:
                    if label.lower() in white_text_lower:
                        predicted_label = label
                        break
            
            if predicted_label is None:
                print(f"  ✗ ERROR: Could not parse answer from: {white_text[:200]}")
                metrics["parse_errors"] += 1
                metrics["results"].append({
                    "story_id": story_id,
                    "expected": expected_label,
                    "predicted": "PARSE_ERROR",
                    "correct": False,
                    "time": elapsed_time,
                    "error": "Could not parse answer"
                })
                continue
            
            print(f"  Predicted: {predicted_label}")
            
            # Compare with expected
            is_correct = (predicted_label.lower() == expected_label.lower())
            
            if is_correct:
                metrics["correct"] += 1
                print(f"  ✓ CORRECT")
            else:
                metrics["incorrect"] += 1
                print(f"  ✗ INCORRECT (expected {expected_label}, got {predicted_label})")
            
            metrics["results"].append({
                "story_id": story_id,
                "expected": expected_label,
                "predicted": predicted_label,
                "correct": is_correct,
                "time": elapsed_time
            })
            
        except Exception as e:
            print(f"  ✗ EXCEPTION: {e}")
            metrics["parse_errors"] += 1
            metrics["results"].append({
                "story_id": story_id,
                "expected": expected_label,
                "predicted": "EXCEPTION",
                "correct": False,
                "time": 0.0,
                "error": str(e)
            })
    
    # Calculate final metrics
    metrics["accuracy"] = metrics["correct"] / metrics["total_cases"] if metrics["total_cases"] > 0 else 0.0
    metrics["avg_time_per_case"] = metrics["total_time"] / metrics["total_cases"] if metrics["total_cases"] > 0 else 0.0
    
    print(f"\n{'='*60}")
    print(f"EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total cases: {metrics['total_cases']}")
    print(f"Correct: {metrics['correct']}")
    print(f"Incorrect: {metrics['incorrect']}")
    print(f"Parse errors: {metrics['parse_errors']}")
    print(f"Accuracy: {metrics['accuracy']:.2%}")
    print(f"Avg time per case: {metrics['avg_time_per_case']:.2f}s")
    print(f"{'='*60}\n")
    
    return metrics


class FolioGreenAgentExecutor(AgentExecutor):
    def __init__(self):
        pass

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        # Parse the task
        print("Green agent: Received evaluation request, parsing...")
        user_input = context.get_user_input()
        tags = parse_tags(user_input)
        
        white_agent_url = tags.get("white_agent_url")
        if not white_agent_url:
            error_msg = "ERROR: No <white_agent_url> tag found in request"
            print(error_msg)
            await event_queue.enqueue_event(new_agent_text_message(error_msg))
            return
        
        # Get optional max_examples parameter
        max_examples_str = tags.get("max_examples")
        max_examples = int(max_examples_str) if max_examples_str else None
        
        print(f"Green agent: Evaluating white agent at {white_agent_url}")
        if max_examples:
            print(f"Green agent: Limited to {max_examples} examples")
        
        # Run evaluation
        timestamp_started = time.time()
        metrics = await evaluate_white_agent_on_folio(white_agent_url, max_examples=max_examples)
        metrics["total_evaluation_time"] = time.time() - timestamp_started
        
        # Format result message
        result_text = f"""✅ FOLIO Evaluation Complete

White Agent: {white_agent_url}

Results:
  • Total Cases: {metrics['total_cases']}
  • Correct: {metrics['correct']}
  • Incorrect: {metrics['incorrect']}
  • Parse Errors: {metrics['parse_errors']}
  • Accuracy: {metrics['accuracy']:.2%}
  • Avg Time/Case: {metrics['avg_time_per_case']:.2f}s
  • Total Time: {metrics['total_evaluation_time']:.2f}s

Detailed results available in metrics JSON.
"""
        
        print("Green agent: Sending results...")
        await event_queue.enqueue_event(new_agent_text_message(result_text))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError


def start_green_agent(agent_name="folio_green_agent", host="localhost", port=9001):
    print("Starting FOLIO green agent...")
    print(f"Environment variables for URL resolution:")
    print(f"  AGENT_URL: {os.environ.get('AGENT_URL', 'not set')}")
    print(f"  A2A_AGENT_URL: {os.environ.get('A2A_AGENT_URL', 'not set')}")  
    print(f"  PUBLIC_URL: {os.environ.get('PUBLIC_URL', 'not set')}")
    print(f"  CLOUDRUN_HOST: {os.environ.get('CLOUDRUN_HOST', 'not set')}")
    
    agent_card_dict = load_agent_card_toml(agent_name)
    
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
    
    agent_card_dict["url"] = url  # complete all required card fields

    request_handler = DefaultRequestHandler(
        agent_executor=FolioGreenAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )

    app = A2AStarletteApplication(
        agent_card=AgentCard(**agent_card_dict),
        http_handler=request_handler,
    )
    
    # Build the Starlette app and add health check endpoint
    starlette_app = app.build()
    
    # Add /status health check endpoint for AgentBeats
    from starlette.responses import JSONResponse
    from starlette.routing import Route
    
    async def health_check(request):
        return JSONResponse({"status": "ok", "agent": "folio_green_agent"})
    
    # Add the route to the existing app
    starlette_app.routes.append(Route("/status", health_check))

    uvicorn.run(starlette_app, host=host, port=port)

