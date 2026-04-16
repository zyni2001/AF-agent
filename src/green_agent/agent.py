"""Green agent implementation - FOLIO benchmark evaluator."""

import uvicorn
import tomllib
import json
import time
import asyncio
import httpx
import hashlib
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


def _verbose_logging() -> bool:
    value = os.environ.get("FOLIO_VERBOSE_LOGGING", "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


def _max_concurrent() -> int:
    return int(os.environ.get("FOLIO_MAX_CONCURRENT", "4"))


def _request_timeout_seconds() -> float:
    return float(os.environ.get("FOLIO_A2A_TIMEOUT_SECONDS", "300"))


def _write_metrics_json(payload: dict) -> None:
    metrics_path = os.environ.get("FOLIO_METRICS_PATH", "").strip()
    if not metrics_path:
        return

    metrics_dir = os.path.dirname(metrics_path)
    if metrics_dir:
        os.makedirs(metrics_dir, exist_ok=True)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)

    print(f"Green agent: Metrics written to {metrics_path}")


def _sanitize_filename_component(value: str) -> str:
    cleaned = []
    for ch in value.lower():
        if ch.isalnum():
            cleaned.append(ch)
        else:
            cleaned.append("_")
    collapsed = "".join(cleaned).strip("_")
    return collapsed or "agent"


def _get_checkpoint_path(
    participant_id: str,
    participant_name: str,
    white_agent_url: str,
    max_examples: int | None,
) -> str:
    checkpoint_dir = os.environ.get("FOLIO_CHECKPOINT_DIR", "").strip()
    if not checkpoint_dir:
        return ""

    dataset_variant = os.environ.get("FOLIO_DATASET_VARIANT", "refined").strip().lower()
    scope = f"max{max_examples}" if max_examples is not None else "all"
    component = _sanitize_filename_component(participant_id or participant_name or white_agent_url)
    return os.path.join(checkpoint_dir, f"{dataset_variant}_{scope}_{component}_checkpoint.json")


def _load_checkpoint(checkpoint_path: str) -> dict:
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return {"results": {}}

    with open(checkpoint_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, dict):
        return {"results": {}}

    results = payload.get("results", {})
    if not isinstance(results, dict):
        results = {}
    payload["results"] = results
    return payload


def _write_checkpoint(checkpoint_path: str, payload: dict) -> None:
    if not checkpoint_path:
        return

    checkpoint_dir = os.path.dirname(checkpoint_path)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    tmp_path = f"{checkpoint_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)
    os.replace(tmp_path, checkpoint_path)


def _compute_sample_key(idx, row) -> str:
    story_id = str(row.get("story_id", "")).strip()
    example_id = str(row.get("example_id", "")).strip()
    label = str(row.get("label", "")).strip()
    premises = str(row.get("premises", "")).strip()
    conclusion = str(row.get("conclusion", "")).strip()

    digest_source = f"{idx}\n{story_id}\n{example_id}\n{label}\n{premises}\n{conclusion}"
    digest = hashlib.sha1(digest_source.encode("utf-8")).hexdigest()[:16]

    if story_id and story_id.lower() != "nan":
        prefix = f"story-{story_id}"
    elif example_id and example_id.lower() != "nan":
        prefix = f"example-{example_id}"
    else:
        prefix = f"row-{idx}"
    return f"{prefix}-{digest}"


def _build_checkpoint_payload(
    checkpoint_path: str,
    participant_id: str,
    participant_name: str,
    white_agent_url: str,
    max_examples: int | None,
    total_cases: int,
    results_by_key: dict,
) -> dict:
    existing = _load_checkpoint(checkpoint_path) if checkpoint_path else {"results": {}}
    now = time.time()
    metadata = existing.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}

    metadata.update({
        "checkpoint_version": 1,
        "dataset_variant": os.environ.get("FOLIO_DATASET_VARIANT", "refined"),
        "participant_id": participant_id,
        "participant_name": participant_name,
        "white_agent_url": white_agent_url,
        "max_examples": max_examples,
        "total_cases": total_cases,
        "updated_at_unix": now,
    })
    metadata.setdefault("created_at_unix", now)

    return {
        "metadata": metadata,
        "results": results_by_key,
    }


def _build_exception_result(sample_key: str, idx, row, error: Exception) -> dict:
    story_id = row.get("story_id", idx)
    expected_label = str(row["label"]).strip()
    error_text = str(error)
    is_timeout = isinstance(error, httpx.TimeoutException) or "timed out" in error_text.lower()
    return {
        "sample_key": sample_key,
        "story_id": story_id,
        "expected": expected_label,
        "predicted": "TIMEOUT" if is_timeout else "EXCEPTION",
        "correct": False,
        "time": 0.0,
        "error": error_text,
        "status": "timeout" if is_timeout else "exception",
    }


def _is_retryable_result(result: dict) -> bool:
    status = str(result.get("status", "")).strip().lower()
    error_text = str(result.get("error", "")).lower()
    return status == "timeout" or (status == "exception" and "timed out" in error_text)


def _summarize_results(
    ordered_results: list[dict],
    total_cases: int,
    wall_time_this_run: float,
    resumed_count: int,
    retry_count: int,
    completed_in_this_run: int,
    checkpoint_path: str,
) -> dict:
    metrics = {
        "total_cases": total_cases,
        "correct": 0,
        "incorrect": 0,
        "parse_errors": 0,
        "timeout_errors": 0,
        "exceptions": 0,
        "total_time": wall_time_this_run,
        "wall_time_this_run": wall_time_this_run,
        "resumed_from_checkpoint": resumed_count,
        "retried_timeouts_from_checkpoint": retry_count,
        "completed_in_this_run": completed_in_this_run,
        "checkpoint_path": checkpoint_path,
        "results": [],
    }

    for result in ordered_results:
        metrics["results"].append(result)
        status = result.get("status", "")
        if result.get("correct"):
            metrics["correct"] += 1
        elif status == "parse_error":
            metrics["parse_errors"] += 1
        elif status == "timeout":
            metrics["timeout_errors"] += 1
        elif status == "exception":
            metrics["exceptions"] += 1
        else:
            metrics["incorrect"] += 1

    metrics["sum_case_time"] = sum(float(r.get("time", 0.0) or 0.0) for r in metrics["results"])
    metrics["accuracy"] = metrics["correct"] / metrics["total_cases"] if metrics["total_cases"] > 0 else 0.0
    metrics["avg_time_per_case"] = metrics["total_time"] / metrics["total_cases"] if metrics["total_cases"] > 0 else 0.0
    metrics["avg_model_time_per_case"] = metrics["sum_case_time"] / metrics["total_cases"] if metrics["total_cases"] > 0 else 0.0

    category_breakdown = {}
    for result in metrics["results"]:
        cat = result["expected"]
        if cat not in category_breakdown:
            category_breakdown[cat] = {"correct": 0, "total": 0}
        category_breakdown[cat]["total"] += 1
        if result["correct"]:
            category_breakdown[cat]["correct"] += 1
    for cat in category_breakdown:
        cb = category_breakdown[cat]
        cb["accuracy"] = round(cb["correct"] / cb["total"] * 100, 2) if cb["total"] > 0 else 0.0
    metrics["category_breakdown"] = category_breakdown
    return metrics


def load_agent_card_toml(agent_name):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    toml_path = os.path.join(current_dir, f"{agent_name}.toml")
    with open(toml_path, "rb") as f:
        return tomllib.load(f)


async def _evaluate_single_sample(
    white_agent_card, http_client, semaphore, sample_num, total, idx, row
) -> dict:
    """Evaluate a single FOLIO sample against the white agent (concurrency-safe).
    
    Returns a result dict for this sample.
    """
    async with semaphore:
        story_id = row.get('story_id', idx)
        expected_label = str(row['label']).strip()
        problem_text = format_folio_problem_as_text(row)
        
        if _verbose_logging():
            print(f"  [{sample_num}/{total}] Story {story_id} (expected: {expected_label}) — sending...")
        
        try:
            start_time = time.time()
            response = await my_a2a.send_message_with_card(
                white_agent_card,
                problem_text,
                httpx_client=http_client,
                timeout=_request_timeout_seconds(),
            )
            elapsed_time = time.time() - start_time
            
            # Parse response
            res_root = response.root
            
            # Handle error responses from white agent
            if isinstance(res_root, JSONRPCErrorResponse):
                error_msg = res_root.error.message if hasattr(res_root, 'error') else "Unknown error"
                predicted_label = "Uncertain"
                is_correct = (expected_label.lower() == "uncertain")
                symbol = "✓" if is_correct else "✗"
                if _verbose_logging():
                    print(f"  [{sample_num}/{total}] Story {story_id}: {symbol} ERROR→Uncertain (expected {expected_label}) [{elapsed_time:.1f}s]")
                return {
                    "sample_key": row["_sample_key"],
                    "story_id": story_id, "expected": expected_label,
                    "predicted": "Uncertain", "correct": is_correct,
                    "time": elapsed_time, "error": error_msg,
                    "status": "error_response"
                }
            
            if not isinstance(res_root, SendMessageSuccessResponse):
                if _verbose_logging():
                    print(f"  [{sample_num}/{total}] Story {story_id}: ✗ Bad response type [{elapsed_time:.1f}s]")
                return {
                    "sample_key": row["_sample_key"],
                    "story_id": story_id, "expected": expected_label,
                    "predicted": "ERROR", "correct": False,
                    "time": elapsed_time, "error": f"Invalid response type: {type(res_root)}",
                    "status": "parse_error"
                }
            
            res_result = res_root.result
            if not isinstance(res_result, Message):
                if _verbose_logging():
                    print(f"  [{sample_num}/{total}] Story {story_id}: ✗ Not a Message [{elapsed_time:.1f}s]")
                return {
                    "sample_key": row["_sample_key"],
                    "story_id": story_id, "expected": expected_label,
                    "predicted": "ERROR", "correct": False,
                    "time": elapsed_time, "error": "Invalid result type",
                    "status": "parse_error"
                }
            
            # Extract text from response
            text_parts = get_text_parts(res_result.parts)
            if not text_parts:
                if _verbose_logging():
                    print(f"  [{sample_num}/{total}] Story {story_id}: ✗ No text [{elapsed_time:.1f}s]")
                return {
                    "sample_key": row["_sample_key"],
                    "story_id": story_id, "expected": expected_label,
                    "predicted": "ERROR", "correct": False,
                    "time": elapsed_time, "error": "No text in response",
                    "status": "parse_error"
                }
            
            white_text = text_parts[0].strip()
            
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
                if _verbose_logging():
                    print(f"  [{sample_num}/{total}] Story {story_id}: ✗ PARSE_ERROR [{elapsed_time:.1f}s]")
                return {
                    "sample_key": row["_sample_key"],
                    "story_id": story_id, "expected": expected_label,
                    "predicted": "PARSE_ERROR", "correct": False,
                    "time": elapsed_time, "error": "Could not parse answer",
                    "status": "parse_error"
                }
            
            is_correct = (predicted_label.lower() == expected_label.lower())
            symbol = "✓" if is_correct else "✗"
            if _verbose_logging():
                print(f"  [{sample_num}/{total}] Story {story_id}: {symbol} {predicted_label} (expected {expected_label}) [{elapsed_time:.1f}s]")
            
            return {
                "sample_key": row["_sample_key"],
                "story_id": story_id, "expected": expected_label,
                "predicted": predicted_label, "correct": is_correct,
                "time": elapsed_time, "status": "ok"
            }
            
        except Exception as e:
            if _verbose_logging():
                print(f"  [{sample_num}/{total}] Story {story_id}: ✗ EXCEPTION: {e}")
            return {
                "sample_key": row["_sample_key"],
                "story_id": story_id, "expected": expected_label,
                "predicted": "EXCEPTION", "correct": False,
                "time": 0.0, "error": str(e), "status": "exception"
            }


async def _evaluate_pending_sample(
    white_agent_card, semaphore, sample_num, total, idx, row
) -> tuple[int, int, dict, dict]:
    result = await _evaluate_single_sample(
        white_agent_card, None, semaphore, sample_num, total, idx, row
    )
    return sample_num, idx, row, result


async def evaluate_white_agent_on_folio(
    white_agent_url: str,
    participant_id: str,
    participant_name: str,
    max_examples: int = None,
) -> dict:
    """Evaluate a white agent on FOLIO validation dataset using parallel requests.
    
    Uses asyncio.Semaphore to limit concurrency to a conservative default of 4
    simultaneous requests, reducing timeout pressure on local agents and Vertex.
    
    Args:
        white_agent_url: URL of the white agent to evaluate
        max_examples: If provided, only test on first N examples
    
    Returns:
        dict with evaluation metrics
    """
    max_concurrent = _max_concurrent()
    request_timeout = _request_timeout_seconds()
    print(f"\n{'='*60}")
    print(f"Starting FOLIO Evaluation (parallel, max {max_concurrent} concurrent)")
    print(f"White agent: {white_agent_url}")
    print(f"A2A request timeout: {request_timeout:.0f}s")
    print(f"{'='*60}\n")
    
    # Load validation dataset
    df = load_validation_dataset(max_examples=max_examples)
    total = len(df)
    print(f"Loaded {total} samples")

    sample_keys = []
    for idx, row in df.iterrows():
        sample_keys.append(_compute_sample_key(idx, row))
    df = df.copy()
    df["_sample_key"] = sample_keys

    checkpoint_path = _get_checkpoint_path(participant_id, participant_name, white_agent_url, max_examples)
    checkpoint_payload = _load_checkpoint(checkpoint_path)
    existing_results = checkpoint_payload.get("results", {})
    retryable_keys = {
        sample_key for sample_key, result in existing_results.items()
        if isinstance(result, dict) and _is_retryable_result(result)
    }
    results_by_key = {
        sample_key: result for sample_key, result in existing_results.items()
        if sample_key not in retryable_keys
    }
    resumed_count = sum(1 for sample_key in df["_sample_key"] if sample_key in results_by_key)
    retry_count = sum(1 for sample_key in df["_sample_key"] if sample_key in retryable_keys)

    if checkpoint_path:
        print(f"Checkpoint file: {checkpoint_path}")
    if resumed_count:
        print(f"Resuming from checkpoint with {resumed_count}/{total} completed samples")
    if retry_count:
        print(f"Retrying {retry_count} timeout sample(s) from checkpoint")
    elif checkpoint_path:
        print("No existing checkpoint found; starting fresh")
    
    # Resolve agent card ONCE (avoid 203 redundant lookups)
    print("Resolving white agent card...")
    white_agent_card = await my_a2a.get_agent_card(white_agent_url)
    if white_agent_card is None:
        raise RuntimeError(f"Could not resolve agent card from {white_agent_url}")
    print(f"Agent card resolved: {white_agent_card.name}")
    
    semaphore = asyncio.Semaphore(max_concurrent)
    
    # Build async tasks for all samples
    # NOTE: Each task creates its own httpx client to avoid shared-client issues
    pending_items = []
    for sample_num, (idx, row) in enumerate(df.iterrows(), start=1):
        if row["_sample_key"] in results_by_key:
            continue
        pending_items.append((sample_num, idx, row))

    tasks = []
    for sample_num, idx, row in pending_items:
        tasks.append(
            asyncio.create_task(
                _evaluate_pending_sample(
                    white_agent_card, semaphore,
                    sample_num, total, idx, row
                )
            )
        )

    print(f"Dispatching {len(tasks)} pending samples with concurrency={max_concurrent}...")
    eval_start = time.time()
    completed_in_this_run = 0
    if checkpoint_path:
        _write_checkpoint(
            checkpoint_path,
            _build_checkpoint_payload(
                checkpoint_path=checkpoint_path,
                participant_id=participant_id,
                participant_name=participant_name,
                white_agent_url=white_agent_url,
                max_examples=max_examples,
                total_cases=total,
                results_by_key=results_by_key,
            ),
        )

    if tasks:
        for task in asyncio.as_completed(tasks):
            try:
                sample_num, idx, row, result = await task
            except Exception as exc:
                raise RuntimeError("Unexpected task failure during sample evaluation") from exc

            sample_key = result["sample_key"]
            if sample_key != row["_sample_key"]:
                raise RuntimeError(
                    f"Checkpoint key mismatch: expected {row['_sample_key']}, got {sample_key}"
                )

            results_by_key[sample_key] = result
            completed_in_this_run += 1

            if checkpoint_path:
                _write_checkpoint(
                    checkpoint_path,
                    _build_checkpoint_payload(
                        checkpoint_path=checkpoint_path,
                        participant_id=participant_id,
                        participant_name=participant_name,
                        white_agent_url=white_agent_url,
                        max_examples=max_examples,
                        total_cases=total,
                        results_by_key=results_by_key,
                    ),
                )

            if not _verbose_logging():
                print(
                    f"Checkpoint progress: {len(results_by_key)}/{total} "
                    f"completed for {participant_name or participant_id}"
                )

    eval_elapsed = time.time() - eval_start

    ordered_results = []
    for _, row in df.iterrows():
        sample_key = row["_sample_key"]
        if sample_key in results_by_key:
            ordered_results.append(results_by_key[sample_key])

    metrics = _summarize_results(
        ordered_results=ordered_results,
        total_cases=total,
        wall_time_this_run=eval_elapsed,
        resumed_count=resumed_count,
        retry_count=retry_count,
        completed_in_this_run=completed_in_this_run,
        checkpoint_path=checkpoint_path,
    )
    
    print(f"\n{'='*60}")
    print(f"EVALUATION COMPLETE (wall time: {eval_elapsed:.1f}s)")
    print(f"{'='*60}")
    print(f"Total cases: {metrics['total_cases']}")
    print(f"Correct: {metrics['correct']}")
    print(f"Incorrect: {metrics['incorrect']}")
    print(f"Parse errors: {metrics['parse_errors']}")
    print(f"Timeout errors: {metrics['timeout_errors']}")
    print(f"Other exceptions: {metrics['exceptions']}")
    print(f"Accuracy: {metrics['accuracy']:.2%}")
    print(f"Avg time per case: {metrics['avg_time_per_case']:.2f}s")
    print(f"Avg model time per case: {metrics['avg_model_time_per_case']:.2f}s")
    print(f"Wall-clock time: {eval_elapsed:.1f}s ({eval_elapsed/60:.1f} min)")
    print(f"Resumed from checkpoint: {metrics['resumed_from_checkpoint']}")
    print(f"Retried timeouts from checkpoint: {metrics['retried_timeouts_from_checkpoint']}")
    print(f"Completed in this run: {metrics['completed_in_this_run']}")
    print(f"\nPer-category breakdown:")
    category_breakdown = metrics["category_breakdown"]
    for cat in ["True", "False", "Uncertain"]:
        if cat in category_breakdown:
            cb = category_breakdown[cat]
            print(f"  {cat:>9s}: {cb['correct']}/{cb['total']} = {cb['accuracy']:.2f}%")
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
        detailed_metrics = []
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
            metrics = await evaluate_white_agent_on_folio(
                endpoint,
                participant_id=agent_id,
                participant_name=agent_name,
                max_examples=max_examples,
            )
            metrics["total_evaluation_time"] = time.time() - timestamp_started
            
            # Structure results for this participant
            # Use 'id' and 'score' to match default AgentBeats query format
            # Per-category breakdown for the report table
            cat_breakdown = metrics.get('category_breakdown', {})
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
                "timeout_errors": metrics['timeout_errors'],
                "exceptions": metrics['exceptions'],
                "time_used": round(metrics['total_evaluation_time'], 2),
                "avg_time_per_case": round(metrics['avg_time_per_case'], 2),
                "max_score": metrics['total_cases'],
                # Per-category breakdown (True / False / Uncertain)
                "true_correct": cat_breakdown.get("True", {}).get("correct", 0),
                "true_total": cat_breakdown.get("True", {}).get("total", 0),
                "true_accuracy": cat_breakdown.get("True", {}).get("accuracy", 0.0),
                "false_correct": cat_breakdown.get("False", {}).get("correct", 0),
                "false_total": cat_breakdown.get("False", {}).get("total", 0),
                "false_accuracy": cat_breakdown.get("False", {}).get("accuracy", 0.0),
                "uncertain_correct": cat_breakdown.get("Uncertain", {}).get("correct", 0),
                "uncertain_total": cat_breakdown.get("Uncertain", {}).get("total", 0),
                "uncertain_accuracy": cat_breakdown.get("Uncertain", {}).get("accuracy", 0.0),
            }
            all_results.append(participant_result)
            detailed_metrics.append({
                "id": agent_name,
                "agent_id": agent_id,
                "endpoint": endpoint,
                "metrics": metrics,
            })
            
            # Build human-readable text for this participant
            # Build per-category summary lines
            cat_lines = []
            for cat in ["True", "False", "Uncertain"]:
                cb = cat_breakdown.get(cat, {})
                if cb:
                    cat_lines.append(f"    {cat:>9s}: {cb.get('correct',0)}/{cb.get('total',0)} = {cb.get('accuracy',0):.2f}%")
            cat_summary = "\n".join(cat_lines)
            
            result_texts.append(f"""
Agent: {agent_name}
  • Total Cases: {metrics['total_cases']}
  • Correct: {metrics['correct']}
  • Incorrect: {metrics['incorrect']}
  • Parse Errors: {metrics['parse_errors']}
  • Timeout Errors: {metrics['timeout_errors']}
  • Other Exceptions: {metrics['exceptions']}
  • Accuracy: {metrics['accuracy']:.2%}
  • Per-category:
{cat_summary}
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
        _write_metrics_json({
            "dataset_variant": os.environ.get("FOLIO_DATASET_VARIANT", "refined"),
            "max_examples": max_examples,
            "participant_count": len(participants_to_eval),
            "summary": structured_results,
            "participants": detailed_metrics,
            "generated_at_unix": time.time(),
        })
        
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

    uvicorn.run(starlette_app, host=host, port=port, log_level="warning", access_log=False)
