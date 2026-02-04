"""Green agent implementation - FOLIO benchmark evaluator."""

import uvicorn
import tomllib
import json
import time
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import AgentCard, SendMessageSuccessResponse, Message, JSONRPCErrorResponse, Part, DataPart, TextPart, TaskState
from a2a.utils import new_agent_text_message, get_text_parts, new_task

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


TERMINAL_STATES = {
    TaskState.completed,
    TaskState.canceled,
    TaskState.failed,
    TaskState.rejected,
}


class FolioGreenAgentExecutor(AgentExecutor):
    def __init__(self):
        pass

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        # Handle task lifecycle properly (like debate example)
        task = context.current_task
        if task and task.status.state in TERMINAL_STATES:
            print(f"Green agent: Task {task.id} already processed (state: {task.status.state})")
            return
        
        if not task:
            task = new_task(context.message)
            await event_queue.enqueue_event(task)
        
        # Create TaskUpdater for proper artifact handling
        updater = TaskUpdater(event_queue, task.id, context.context_id)
        await updater.start_work()
        
        # Parse the request
        print("Green agent: Received evaluation request, parsing...")
        user_input = context.get_user_input()
        print(f"Green agent: Raw input: {user_input[:500] if user_input else 'None'}...")
        
        participants_to_eval = []
        max_examples = None
        
        # Try to parse as JSON (AgentBeats format)
        try:
            import json
            data = json.loads(user_input)
            print(f"Green agent: Parsed JSON data: {data}")
            
            # AgentBeats sends participants and config
            participants = data.get("participants", [])
            config = data.get("config", {})
            
            # Build list of all participants to evaluate
            if participants:
                if isinstance(participants, list):
                    for p in participants:
                        endpoint = p.get("endpoint")
                        name = p.get("name", "unknown")
                        agent_id = p.get("agent_id", name)
                        if endpoint:
                            participants_to_eval.append({
                                "name": name,
                                "agent_id": agent_id,
                                "endpoint": endpoint
                            })
                elif isinstance(participants, dict):
                    # participants might be {role: endpoint/info} format
                    for role, info in participants.items():
                        if isinstance(info, str):
                            participants_to_eval.append({
                                "name": role,
                                "agent_id": role,
                                "endpoint": info
                            })
                        elif isinstance(info, dict):
                            participants_to_eval.append({
                                "name": role,
                                "agent_id": info.get("agent_id", role),
                                "endpoint": info.get("endpoint")
                            })
            
            # Get max_examples from config
            max_examples = config.get("max_examples")
            
            print(f"Green agent: Found {len(participants_to_eval)} participants to evaluate")
            for p in participants_to_eval:
                print(f"  - {p['name']}: {p['endpoint']}")
        except json.JSONDecodeError:
            print("Green agent: Not JSON, trying tag-based parsing...")
            # Fall back to tag-based parsing (for local testing)
            tags = parse_tags(user_input)
            white_agent_url = tags.get("white_agent_url")
            max_examples_str = tags.get("max_examples")
            max_examples = int(max_examples_str) if max_examples_str else None
            if white_agent_url:
                participants_to_eval.append({
                    "name": "agent",
                    "agent_id": "agent",
                    "endpoint": white_agent_url
                })
        
        if not participants_to_eval:
            error_msg = "ERROR: No participants found in request (tried JSON and tags)"
            print(error_msg)
            print(f"Green agent: Full input was: {user_input}")
            await event_queue.enqueue_event(new_agent_text_message(error_msg))
            return
        
        if max_examples:
            print(f"Green agent: Limited to {max_examples} examples per agent")
        
        # Evaluate ALL participants and collect results
        all_results = []
        result_texts = []
        
        for participant in participants_to_eval:
            agent_name = participant["name"]
            agent_id = participant["agent_id"]
            endpoint = participant["endpoint"]
            
            print(f"\n{'='*60}")
            print(f"Green agent: Evaluating participant '{agent_name}' at {endpoint}")
            print(f"{'='*60}")
            
            # Run evaluation for this participant
            timestamp_started = time.time()
            metrics = await evaluate_white_agent_on_folio(endpoint, max_examples=max_examples)
            metrics["total_evaluation_time"] = time.time() - timestamp_started
            
            # Structure results for this participant
            # Use 'id' and 'score' to match default AgentBeats query format
            participant_result = {
                "id": agent_name,  # Required by default AgentBeats query
                "score": round(metrics['accuracy'] * 100, 2),  # Required by default AgentBeats query
                "accuracy": round(metrics['accuracy'] * 100, 2),
                "agent": agent_name,  # Keep for backwards compatibility
                "agent_id": agent_id,
                "pass_rate": round(metrics['accuracy'] * 100, 2),
                "correct": metrics['correct'],
                "total": metrics['total_cases'],
                "incorrect": metrics['incorrect'],
                "parse_errors": metrics['parse_errors'],
                "time_used": round(metrics['total_evaluation_time'], 2),
                "avg_time_per_case": round(metrics['avg_time_per_case'], 2),
                "max_score": metrics['total_cases']
            }
            all_results.append(participant_result)
            
            # Build human-readable text for this participant
            result_texts.append(f"""
Agent: {agent_name}
  • Total Cases: {metrics['total_cases']}
  • Correct: {metrics['correct']}
  • Incorrect: {metrics['incorrect']}
  • Parse Errors: {metrics['parse_errors']}
  • Accuracy: {metrics['accuracy']:.2%}
  • Avg Time/Case: {metrics['avg_time_per_case']:.2f}s
  • Total Time: {metrics['total_evaluation_time']:.2f}s
""")
        
        # Format combined result message
        result_text = f"""✅ FOLIO Evaluation Complete

Evaluated {len(participants_to_eval)} agent(s):
{"".join(result_texts)}
Detailed results available in metrics JSON.
"""
        
        # Structure final results for AgentBeats leaderboard
        print(f"\nGreen agent: Sending results for {len(all_results)} participants...")
        
        # For single participant, emit fields directly at top level
        # Query: SELECT id, score, accuracy FROM results
        if len(all_results) == 1:
            structured_results = all_results[0]
        else:
            # For multiple, use array
            structured_results = {"data": all_results}
        
        print(f"Green agent: Structured results: {json.dumps(structured_results)}")
        
        # Emit results as A2A artifact with DataPart (AgentBeats expects this format)
        await updater.add_artifact(
            parts=[
                Part(root=TextPart(text=result_text)),
                Part(root=DataPart(data=structured_results)),
            ],
            name="assessment_results",
        )
        
        # Complete the task (this triggers the client to capture artifacts)
        await updater.complete()
        print(f"Green agent: Artifact emitted successfully via TaskUpdater")

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

