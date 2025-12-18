# FOLIO Logical Reasoning Benchmark

An AgentBeats-compatible benchmark for evaluating AI agents on first-order logic inference tasks using the FOLIO dataset.

## 🎯 Overview

This benchmark evaluates how well AI agents can determine whether logical conclusions follow from given premises. It implements and compares two approaches:

1. **Baseline Agent**: Direct LLM reasoning without formal logic
2. **Autoformalization Agent**: LLM generates executable Z3 Python code to solve logical reasoning problems

## 🏗️ Architecture

### Green Agent (Benchmark Evaluator)

- Loads FOLIO validation dataset (203 logical reasoning problems)
- Sends problems to white agents via A2A protocol
- Compares agent responses with ground truth labels
- Reports accuracy and performance metrics

### White Agents (Being Evaluated)

#### Baseline Agent
- Uses Gemini 2.0 Flash for direct reasoning
- Receives natural language premises and conclusion
- Returns: True, False, or Uncertain

#### Autoformalization Agent
- **Stage 1**: LLM generates executable Z3 Python code from natural language problem
- **Stage 2**: Execute the generated code and parse the result
- Returns: True (conclusion follows), False (contradiction), or Uncertain (cannot determine)

## 🚀 Quick Start

### Local Testing

```bash
# 1. Install dependencies
uv sync  # or: pip install -e .

# 2. Set API key
export GEMINI_API_KEY='your_api_key_here'

# 3. Run quick test
./quick_local_test.sh

# 4. Or run with more examples
python main.py launch --max 10 --both
```

### Deploy to AgentBeats

```bash
# 1. Deploy Baseline Agent
./deploy_baseline.sh

# 2. Deploy Green Agent
./deploy_green.sh

# 3. Register at https://v2.agentbeats.org
#    - Add your agents with the Cloud Run URLs
#    - Create an assessment
#    - Run evaluation
```

## 📋 Configuration

Create a `.env` file:

```bash
# Gemini API Key (required)
GEMINI_API_KEY=your_api_key_here

# AgentBeats configuration (for deployment)
HOST=localhost
AGENT_PORT=9001
```

Get your API key from: https://aistudio.google.com/app/apikey

## 📦 Repository Structure

```
folio-benchmark/
├── src/
│   ├── green_agent/           # Benchmark evaluator (Green Agent)
│   │   ├── agent.py            # Main evaluation logic
│   │   └── folio_green_agent.toml
│   ├── white_agent_baseline/   # Direct LLM reasoning agent
│   │   └── agent.py
│   ├── white_agent_autoform/   # Z3 code generation agent
│   │   └── agent.py
│   ├── folio_utils/           # Dataset and utilities
│   │   └── dataset.py
│   └── my_util/               # A2A protocol helpers
│       └── my_a2a.py
├── data/                      # FOLIO dataset
│   └── folio-wiki/dev.csv
├── Dockerfile.baseline        # Baseline agent container
├── Dockerfile.green           # Green agent container
├── deploy_baseline.sh         # Deploy baseline to Cloud Run
├── deploy_green.sh            # Deploy green agent to Cloud Run
├── run.sh                     # AgentBeats controller launcher
├── main.py                    # CLI entry point
├── quick_local_test.sh        # Quick local test script
└── README.md                  # This file
```

## 📊 Dataset

This benchmark uses the FOLIO dataset:

- **Source**: https://github.com/Yale-LILY/FOLIO
- **Task**: First-order logic inference
- **Size**: 203 validation examples
- **Format**: Natural language premises + conclusion → True/False/Uncertain

Each example contains:
- `premises`: Natural language statements
- `conclusion`: Statement to verify
- `label`: Ground truth (True/False/Uncertain)

## 📈 Evaluation Metrics

The benchmark reports:
- **Accuracy**: Percentage of correct predictions
- **Correct/Incorrect counts**: Detailed breakdown
- **Average Time per Case**: Performance measurement
- **Per-Example Results**: Individual predictions vs. ground truth

## 🔧 Usage

### CLI Commands

```bash
# Start individual agents
python main.py green      # Start green agent
python main.py baseline   # Start baseline agent
python main.py autoform   # Start autoformalization agent

# Run evaluations
python main.py quick                    # Quick test (5 examples)
python main.py launch --max 10 --both   # Evaluate 10 examples, both agents
python main.py full                     # Full evaluation (all 203 examples)
```

### AgentBeats Integration

The benchmark follows the AgentBeats green agent specification:

1. **Agent Card**: Defines capabilities and protocols
2. **A2A Protocol**: Communicates with white agents
3. **Evaluation Task**: Receives white agent URL, runs evaluation, returns metrics

Input format for green agent:
```xml
<white_agent_url>
https://your-agent-url.run.app
</white_agent_url>
<max_examples>10</max_examples>
```

## 🐳 Docker Deployment

### Build Locally

```bash
# Build baseline agent
docker build -t folio-baseline-agent -f Dockerfile.baseline .

# Build green agent
docker build -t folio-green-agent -f Dockerfile.green .

# Run locally with AgentBeats controller
docker run -p 8080:8080 \
  -e GEMINI_API_KEY=your_key \
  -e AGENT_ROLE=baseline \
  folio-baseline-agent
```

### Deploy to Cloud Run

The deployment scripts handle:
- Building Docker images with Cloud Build
- Deploying to Google Cloud Run
- Setting environment variables (API key, public URL)
- Configuring AgentBeats controller

Requirements:
- Google Cloud SDK installed
- Project configured: `gcloud config set project YOUR_PROJECT_ID`
- APIs enabled: Cloud Run, Cloud Build

## 📝 Requirements

- **Python**: 3.13+
- **Gemini API key**: For LLM inference
- **Google Cloud account**: For deployment (optional, for AgentBeats platform)

### Dependencies

Key packages (see `pyproject.toml`):
- `a2a-sdk[http-server]>=0.3.8` - Agent-to-Agent protocol
- `earthshaker>=0.1.0` - AgentBeats controller runtime
- `litellm>=1.0.0` - LLM API wrapper
- `z3-solver>=4.12.0` - SMT solver for logical reasoning
- `pandas>=2.0.0` - Data processing
- `uvicorn>=0.27.0` - ASGI web server

## 🧪 Testing

### Local Test Results

```bash
$ ./quick_local_test.sh

Testing Z3 Autoformalization Locally
========================================
Testing autoformalization agent with 2 examples...

✓ Case 1: Predicted: Uncertain, Expected: Uncertain ✓
✓ Case 2: Predicted: True, Expected: True ✓

Accuracy: 100.00%
Average time: 4.26s per case
```

## 🌐 Deployed Agents

Example Cloud Run URLs (your URLs will be different):
```
Baseline Agent: https://folio-baseline-agent-qvayglp4ia-uc.a.run.app
Green Agent:    https://folio-green-agent-qvayglp4ia-uc.a.run.app
```

Register these URLs on https://v2.agentbeats.org to run assessments.

## 🙏 Acknowledgments

- **FOLIO Dataset**: Yale LILY Lab
- **Z3 Solver**: Microsoft Research
- **AgentBeats Platform**: UC Berkeley CS294-282 course

## 📄 Citation

If you use this benchmark, please cite the FOLIO dataset:

```bibtex
@inproceedings{han-etal-2022-folio,
    title = "{FOLIO}: Natural Language Reasoning with First-Order Logic",
    author = "Han, Simeng and Schoelkopf, Hailey and Zhao, Yilun and Qi, Zhenting and Riddell, Martin and Benson, Luke and Sun, Lucy and Zubova, Ekaterina and Qiao, Yujie and Burtell, Matthew and Peng, David and Fan, Jonathan and Liu, Yixin and Wong, Brian and Sailor, Malcolm and Ni, Ansong and Nan, Linyong and Kasai, Jungo and Yu, Tao and Huang, Rui and Joty, Shafiq and Fabbri, Alexander and Kryscinski, Wojciech and Lin, Xi Victoria and Xiong, Caiming and Radev, Dragomir",
    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing",
    year = "2022"
}
```

## 📧 Contact

For questions or issues, please open an issue on GitHub.
