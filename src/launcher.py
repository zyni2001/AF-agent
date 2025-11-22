"""Launcher module - initiates and coordinates the evaluation process."""

import multiprocessing
import json
import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.green_agent.agent import start_green_agent
from src.white_agent_baseline.agent import start_baseline_white_agent
from src.white_agent_autoform.agent import start_autoform_white_agent
from src.my_util import my_a2a


async def launch_evaluation(max_examples: int = None, test_both: bool = True):
    """Launch complete evaluation workflow.
    
    Args:
        max_examples: If provided, only evaluate on first N examples (for testing)
        test_both: If True, test both baseline and autoform agents. If False, only autoform.
    """
    print("\n" + "="*70)
    print("FOLIO BENCHMARK - LAUNCHING EVALUATION")
    print("="*70 + "\n")
    
    # Start green agent
    print("Step 1: Launching green agent (benchmark evaluator)...")
    green_address = ("localhost", 9001)
    green_url = f"http://{green_address[0]}:{green_address[1]}"
    p_green = multiprocessing.Process(
        target=start_green_agent, args=("folio_green_agent", *green_address)
    )
    p_green.start()
    print(f"  Waiting for green agent to be ready at {green_url}...")
    assert await my_a2a.wait_agent_ready(green_url, timeout=30), "Green agent not ready in time"
    print("  ✓ Green agent is ready.\n")
    
    agents_to_test = []
    processes = []
    
    # Start baseline white agent
    if test_both:
        print("Step 2a: Launching baseline white agent (direct LLM)...")
        baseline_address = ("localhost", 9002)
        baseline_url = f"http://{baseline_address[0]}:{baseline_address[1]}"
        p_baseline = multiprocessing.Process(
            target=start_baseline_white_agent, args=baseline_address
        )
        p_baseline.start()
        processes.append(p_baseline)
        print(f"  Waiting for baseline agent to be ready at {baseline_url}...")
        assert await my_a2a.wait_agent_ready(baseline_url, timeout=30), "Baseline agent not ready in time"
        print("  ✓ Baseline agent is ready.\n")
        agents_to_test.append(("Baseline (Direct LLM)", baseline_url))
    
    # Start autoformalization white agent
    print(f"Step {'2b' if test_both else '2'}: Launching autoformalization white agent (LLM→FOL→Z3)...")
    autoform_address = ("localhost", 9003)
    autoform_url = f"http://{autoform_address[0]}:{autoform_address[1]}"
    p_autoform = multiprocessing.Process(
        target=start_autoform_white_agent, args=autoform_address
    )
    p_autoform.start()
    processes.append(p_autoform)
    print(f"  Waiting for autoform agent to be ready at {autoform_url}...")
    assert await my_a2a.wait_agent_ready(autoform_url, timeout=30), "Autoform agent not ready in time"
    print("  ✓ Autoformalization agent is ready.\n")
    agents_to_test.append(("Autoformalization (LLM→FOL→Z3)", autoform_url))
    
    # Evaluate each agent
    print("="*70)
    print("EVALUATION START")
    print("="*70 + "\n")
    
    results = []
    
    for agent_name, agent_url in agents_to_test:
        print(f"\n{'='*70}")
        print(f"Evaluating: {agent_name}")
        print(f"URL: {agent_url}")
        print(f"{'='*70}\n")
        
        # Prepare evaluation task
        max_examples_tag = f"<max_examples>{max_examples}</max_examples>" if max_examples else ""
        task_text = f"""Evaluate the white agent on FOLIO validation dataset.

<white_agent_url>
{agent_url}
</white_agent_url>
{max_examples_tag}

Please run the evaluation and report results.
"""
        
        print("Sending evaluation request to green agent...")
        try:
            response = await my_a2a.send_message(green_url, task_text)
            print("\n" + "-"*70)
            print("Green agent response:")
            print("-"*70)
            print(response)
            print("-"*70 + "\n")
            
            results.append({
                "agent_name": agent_name,
                "agent_url": agent_url,
                "response": str(response)
            })
        except Exception as e:
            print(f"\n✗ ERROR evaluating {agent_name}: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "agent_name": agent_name,
                "agent_url": agent_url,
                "error": str(e)
            })
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)
    print("\nSummary:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['agent_name']}")
        if 'error' in result:
            print(f"   Status: ERROR - {result['error']}")
        else:
            print(f"   Status: Completed")
    print("\n" + "="*70 + "\n")
    
    # Cleanup
    print("Terminating agents...")
    p_green.terminate()
    p_green.join()
    for p in processes:
        p.terminate()
        p.join()
    print("All agents terminated.\n")


async def launch_quick_test():
    """Quick test with just 5 examples on autoform agent only."""
    print("\n" + "="*70)
    print("QUICK TEST MODE - 5 examples, autoform agent only")
    print("="*70 + "\n")
    await launch_evaluation(max_examples=5, test_both=False)


async def launch_full_evaluation():
    """Full evaluation on all validation examples with both agents."""
    print("\n" + "="*70)
    print("FULL EVALUATION MODE - All validation examples, both agents")
    print("="*70 + "\n")
    await launch_evaluation(max_examples=None, test_both=True)

