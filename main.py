"""CLI entry point for folio-benchmark."""

import typer
import asyncio

from src.green_agent.agent import start_green_agent
from src.white_agent_baseline.agent import start_baseline_white_agent
from src.white_agent_autoform.agent import start_autoform_white_agent
from src.launcher import launch_evaluation, launch_quick_test, launch_full_evaluation

app = typer.Typer(help="FOLIO Logical Reasoning Benchmark - AgentBeats evaluation system")


@app.command()
def green():
    """Start the green agent (FOLIO benchmark evaluator)."""
    start_green_agent()


@app.command()
def baseline():
    """Start the baseline white agent (direct LLM reasoning)."""
    start_baseline_white_agent()


@app.command()
def autoform():
    """Start the autoformalization white agent (LLM→Z3 Code→Execute)."""
    start_autoform_white_agent()


@app.command()
def launch(
    max_examples: int = typer.Option(None, "--max", "-n", help="Maximum number of examples to evaluate (default: all)"),
    test_both: bool = typer.Option(True, "--both/--autoform-only", help="Test both agents or just autoform"),
):
    """Launch the complete evaluation workflow."""
    asyncio.run(launch_evaluation(max_examples=max_examples, test_both=test_both))


@app.command()
def quick():
    """Quick test with 5 examples on autoform agent only."""
    asyncio.run(launch_quick_test())


@app.command()
def full():
    """Full evaluation on all validation examples with both agents."""
    asyncio.run(launch_full_evaluation())


if __name__ == "__main__":
    app()

