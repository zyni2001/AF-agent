# FOLIO Logical Reasoning Benchmark

An AgentBeats-compatible benchmark for evaluating AI agents on first-order logic inference tasks using the FOLIO dataset.

## Overview

This benchmark evaluates how well AI agents can determine whether logical conclusions follow from given premises. It implements and compares two approaches:

1. **Baseline Agent**: Direct LLM reasoning without formal logic
2. **Autoformalization Agent**: Converts natural language to First-Order Logic (FOL), then uses Vampire theorem prover for verification

## Architecture

### Green Agent (Benchmark Evaluator)

- Loads FOLIO validation dataset (204 logical reasoning problems)
- Sends problems to white agents via A2A protocol
- Compares agent responses with ground truth labels
- Reports accuracy and performance metrics

### White Agents (Being Evaluated)

#### Baseline Agent

- Uses Gemini 2.5 Flash for direct reasoning
- Receives natural language premises and conclusion
- Returns: True, False, or Uncertain

#### Autoformalization Agent

- **Stage 1**: LLM converts natural language → FOL format
- **Stage 2**: Vampire theorem prover verifies logical validity
- Returns: True (refutation found), False (definite), or Uncertain (timeout)

## Quick Start

### Deploy to AgentBeats Platform

```bash
# 1. Deploy Baseline Agent
./deploy_baseline.sh

# 2. Deploy Green Agent
./deploy_green.sh

# 3. Register at https://v2.agentbeats.org
```

See [AGENTBEATS_DEPLOYMENT.md](AGENTBEATS_DEPLOYMENT.md) for detailed deployment instructions.

### Local Installation

```bash
# Install dependencies with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

## Configuration

Create a `.env` file:

```bash
# Gemini API Key (required)
GEMINI_API_KEY=your_api_key_here

# Vampire theorem prover path (for autoform agent)
VAMPIRE_PATH=/path/to/vampire/build/vampire
```

## Usage

### Local Testing

```bash
# Quick test (3 examples)
python main.py quick

# Evaluate both agents on 10 examples
python main.py launch --max 10 --both

# Full evaluation (all 204 validation cases)
python main.py launch --both
```

### Deployed Agent Testing

```bash
# Test deployed agents
python test_deployed_agents.py
```

## Repository Structure

```
folio-benchmark/
├── src/
│   ├── green_agent/          # Benchmark evaluator (Green Agent)
│   │   ├── agent.py           # Main evaluation logic
│   │   └── folio_green_agent.toml  # Agent card configuration
│   ├── white_agent_baseline/  # Direct LLM reasoning agent
│   │   └── agent.py
│   ├── white_agent_autoform/  # Autoformalization agent
│   │   └── agent.py
│   ├── folio_utils/          # Dataset loading and utilities
│   │   ├── dataset.py
│   │   └── format.py
│   ├── vampire_runner/       # Vampire theorem prover wrapper
│   │   └── runner.py
│   └── my_util/             # A2A protocol helpers
│       └── my_a2a.py
├── data/                    # FOLIO dataset
│   └── folio-wiki/
│       └── dev.csv
├── Dockerfile.baseline      # Baseline agent container
├── Dockerfile.green         # Green agent container
├── deploy_baseline.sh       # Deploy baseline to Cloud Run
├── deploy_green.sh          # Deploy green agent to Cloud Run
├── main.py                  # CLI entry point
├── test_deployed_agents.py  # Test deployed agents
├── test_simple.py           # Simple local test
└── README.md               # This file
```

## Dataset

This benchmark uses the FOLIO dataset:

- **Source**: https://github.com/Yale-LILY/FOLIO
- **Task**: First-order logic inference
- **Size**: 204 validation examples
- **Format**: Natural language premises + conclusion → True/False/Uncertain

Each example contains:

- `premises`: Natural language statements
- `premises-FOL`: Formalized first-order logic (for autoformalization)
- `conclusion`: Statement to verify
- `conclusion-FOL`: Formalized conclusion
- `label`: Ground truth (True/False/Uncertain)

## Evaluation Metrics

The benchmark reports:

- **Accuracy**: Percentage of correct predictions
- **Correct/Incorrect/Parse Errors**: Detailed breakdown
- **Average Time per Case**: Performance measurement
- **Per-Example Results**: Individual predictions vs. ground truth

## AgentBeats Integration

This benchmark follows the AgentBeats green agent specification:

1. **Agent Card**: Defines capabilities and protocols (`folio_green_agent.toml`)
2. **A2A Protocol**: Communicates with white agents using Agent-to-Agent protocol
3. **Evaluation Task**: Receives white agent URL, runs evaluation, returns metrics
4. **Input Format**:

```xml
<white_agent_url>
https://your-agent-url.run.app
</white_agent_url>
<max_examples>10</max_examples>
```

## Development

### Running Tests

```bash
# Test dataset loading
python test_simple.py

# Test deployed agents
python test_deployed_agents.py

# Test specific components
python -c "from src.folio_utils.dataset import load_validation_dataset; print(load_validation_dataset(max_examples=5))"
```

### Building Docker Images

```bash
# Build baseline agent
docker build -t folio-baseline-agent -f Dockerfile.baseline .

# Build green agent
docker build -t folio-green-agent -f Dockerfile.green .

# Run locally
docker run -p 8080:8080 -e GEMINI_API_KEY=your_key folio-baseline-agent
```

## Requirements

- Python 3.11+
- Gemini API key (for LLM inference)
- Vampire theorem prover (optional, for autoformalization agent)
- Google Cloud account (for deployment)

### Dependencies

See `pyproject.toml` for full dependency list. Key packages:

- `a2a-sdk[http-server]`: Agent-to-Agent protocol
- `litellm`: LLM API wrapper
- `pandas`: Data processing
- `uvicorn`: ASGI web server
- `pydantic`: Data validation

## License

This project uses the FOLIO dataset, which is available under its original license.

## Acknowledgments

- **FOLIO Dataset**: Yale LILY Lab
- **Vampire Prover**: Vampire development team
- **AgentBeats Platform**: UC Berkeley CS294-282 course

## Citation

If you use this benchmark, please cite the FOLIO dataset:

```bibtex
@inproceedings{han-etal-2022-folio,
    title = "{FOLIO}: Natural Language Reasoning with First-Order Logic",
    author = "Han, Simeng and Schoelkopf, Hailey and Zhao, Yilun and Qi, Zhenting and Riddell, Martin and Benson, Luke and Sun, Lucy and Zubova, Ekaterina and Qiao, Yujie and Burtell, Matthew and Peng, David and Fan, Jonathan and Liu, Yixin and Wong, Brian and Sailor, Malcolm and Ni, Ansong and Nan, Linyong and Kasai, Jungo and Yu, Tao and Huang, Rui and Joty, Shafiq and Fabbri, Alexander and Kryscinski, Wojciech and Lin, Xi Victoria and Xiong, Caiming and Radev, Dragomir",
    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing",
    year = "2022"
}
```

## Contact

For questions or issues, please contact the course staff or open an issue on GitHub.
