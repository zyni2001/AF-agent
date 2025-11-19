#!/usr/bin/env python3
"""
Test deployed FOLIO agents on Cloud Run.
Uses local green agent logic to evaluate deployed white agents.
"""

import asyncio
import sys
import os

# Add current directory to path
sys.path.insert(0, '.')

from src.folio_utils.dataset import load_validation_dataset
from src.my_util.my_a2a import send_message
import time


async def evaluate_deployed_white_agent(white_agent_url: str, max_examples: int = 10):
    """Evaluate a deployed white agent on FOLIO dataset."""
    
    print("=" * 70)
    print(f"Evaluating Deployed White Agent")
    print(f"URL: {white_agent_url}")
    print(f"Test samples: {max_examples}")
    print("=" * 70)
    print()
    
    # Load FOLIO dataset
    print("Loading FOLIO validation dataset...")
    df = load_validation_dataset(max_examples=max_examples)
    print(f"✓ Loaded {len(df)} samples\n")
    
    # Evaluation metrics
    metrics = {
        "total": len(df),
        "correct": 0,
        "incorrect": 0,
        "errors": 0,
        "results": []
    }
    
    # Test each sample
    for idx, row in df.iterrows():
        story_id = row.get('story_id', idx)
        premises = row['premises']
        conclusion = row['conclusion']
        label = row['label']
        
        print(f"\n[{idx + 1}/{len(df)}] Story ID: {story_id}")
        print(f"  Expected: {label}")
        
        # Format problem text
        problem_text = f"""Given the following premises:

{premises}

Conclusion:
{conclusion}

Does the conclusion logically follow from the premises? Answer True, False, or Uncertain."""
        
        # Send to white agent
        try:
            start_time = time.time()
            response = await send_message(
                url=white_agent_url,
                message=problem_text
            )
            elapsed_time = time.time() - start_time
            
            # Parse response
            from a2a.types import SendMessageSuccessResponse, Message, JSONRPCErrorResponse
            
            res_root = response.root
            
            # Handle error response
            if isinstance(res_root, JSONRPCErrorResponse):
                error_msg = res_root.error.message if hasattr(res_root, 'error') else "Unknown error"
                print(f"  ✗ Agent Error: {error_msg}")
                metrics["errors"] += 1
                metrics["results"].append({
                    "story_id": story_id,
                    "expected": label,
                    "predicted": "ERROR",
                    "correct": False,
                    "time": elapsed_time,
                    "error": error_msg
                })
                continue
            
            if not isinstance(res_root, SendMessageSuccessResponse):
                print(f"  ✗ Unexpected response type: {type(res_root)}")
                metrics["errors"] += 1
                continue
            
            res_result = res_root.result
            if not isinstance(res_result, Message):
                print(f"  ✗ Expected Message result")
                metrics["errors"] += 1
                continue
            
            # Extract text content
            from a2a.utils import get_text_parts
            text_parts = get_text_parts(res_result)
            if not text_parts:
                print(f"  ✗ No text in response")
                metrics["errors"] += 1
                continue
            
            white_text = text_parts[0].strip()
            print(f"  Response: {white_text[:100]}...")
            
            # Parse answer
            white_text_lower = white_text.lower().strip()
            predicted_label = None
            
            # Exact match
            if white_text_lower == "true":
                predicted_label = "True"
            elif white_text_lower == "false":
                predicted_label = "False"
            elif white_text_lower == "uncertain":
                predicted_label = "Uncertain"
            # Fuzzy match
            elif "uncertain" in white_text_lower:
                predicted_label = "Uncertain"
            elif "true" in white_text_lower and "false" not in white_text_lower:
                predicted_label = "True"
            elif "false" in white_text_lower and "true" not in white_text_lower:
                predicted_label = "False"
            else:
                for lbl in ["True", "False", "Uncertain"]:
                    if lbl.lower() in white_text_lower:
                        predicted_label = lbl
                        break
            
            if predicted_label is None:
                print(f"  ✗ Could not parse answer")
                metrics["errors"] += 1
                continue
            
            print(f"  Predicted: {predicted_label}")
            
            # Check correctness
            is_correct = (predicted_label == label)
            if is_correct:
                metrics["correct"] += 1
                print(f"  ✓ CORRECT")
            else:
                metrics["incorrect"] += 1
                print(f"  ✗ INCORRECT")
            
            print(f"  Time: {elapsed_time:.2f}s")
            
            metrics["results"].append({
                "story_id": story_id,
                "expected": label,
                "predicted": predicted_label,
                "correct": is_correct,
                "time": elapsed_time
            })
            
        except Exception as e:
            print(f"  ✗ Exception: {e}")
            metrics["errors"] += 1
            metrics["results"].append({
                "story_id": story_id,
                "expected": label,
                "predicted": "ERROR",
                "correct": False,
                "error": str(e)
            })
    
    # Print summary
    print("\n" + "=" * 70)
    print("Evaluation Results")
    print("=" * 70)
    print(f"Total samples: {metrics['total']}")
    print(f"Correct: {metrics['correct']}")
    print(f"Incorrect: {metrics['incorrect']}")
    print(f"Errors: {metrics['errors']}")
    
    if metrics['total'] > 0:
        accuracy = (metrics['correct'] / metrics['total']) * 100
        print(f"Accuracy: {accuracy:.2f}%")
    
    # Calculate average time
    times = [r.get('time', 0) for r in metrics['results'] if 'time' in r]
    if times:
        avg_time = sum(times) / len(times)
        print(f"Average response time: {avg_time:.2f}s")
    
    print("=" * 70)
    
    return metrics


async def main():
    """Main function."""
    
    # Deployed white agent URL
    baseline_url = "https://folio-baseline-agent-qvayglp4ia-uc.a.run.app"
    
    # Number of test samples (can be modified via command line)
    max_examples = 10
    if len(sys.argv) > 1:
        try:
            max_examples = int(sys.argv[1])
        except ValueError:
            print(f"Warning: Invalid max_examples '{sys.argv[1]}', using default 10")
    
    print("\n🚀 Starting evaluation of deployed FOLIO Baseline Agent\n")
    
    metrics = await evaluate_deployed_white_agent(baseline_url, max_examples)
    
    print("\n✅ Evaluation complete!")
    
    # Save results to file
    import json
    output_file = "evaluation_results.json"
    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nDetailed results saved to: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
