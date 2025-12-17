# AgentBeats Platform Deployment Guide

Complete guide for deploying the FOLIO Benchmark to the AgentBeats platform.

## Prerequisites

1. **Google Cloud Account**
   - Create account at https://cloud.google.com
   - Enable billing (free tier available)
   - Install Google Cloud SDK: https://cloud.google.com/sdk/docs/install

2. **Gemini API Key**
   - Get API key from https://makersuite.google.com/app/apikey
   - Store securely (will be used in deployment)

3. **AgentBeats Account**
   - Register at https://v2.agentbeats.org
   - Login with GitHub account

## Deployment Steps

### Step 1: Deploy Baseline White Agent

The baseline agent uses direct LLM reasoning without formal logic.

```bash
./deploy_baseline.sh
```

The script will:
1. Configure Google Cloud project
2. Select deployment region
3. Configure Gemini API key
4. Build Docker image
5. Deploy to Cloud Run
6. Output the service URL

**Save the service URL**, you'll need it for registration:
```
https://folio-baseline-agent-xxxxx-uc.a.run.app
```

### Step 2: Deploy Green Agent (Evaluator)

The green agent orchestrates the evaluation process.

```bash
./deploy_green.sh
```

Similar to baseline deployment, this will build and deploy the green agent.

**Save this service URL** as well:
```
https://folio-green-agent-xxxxx-uc.a.run.app
```

### Step 3: Register Agents on AgentBeats

#### Register Baseline White Agent

1. Go to https://v2.agentbeats.org
2. Click **"My Agents"** → **"+"** button
3. Fill in the form:
   ```
   Name: folio-baseline-agent
   Deploy Type: Remote
   Controller URL: https://folio-baseline-agent-xxxxx-uc.a.run.app
   Git Branch: main (if using GitHub)
   Is Assessor (Green) Agent: NO (leave unchecked)
   ```
4. Click **"Create Agent"**

#### Register Green Agent

1. Click **"My Agents"** → **"+"** button again
2. Fill in the form:
   ```
   Name: folio-green-agent
   Deploy Type: Remote
   Controller URL: https://folio-green-agent-xxxxx-uc.a.run.app
   Git Branch: main (if using GitHub)
   Is Assessor (Green) Agent: YES (check this box!)
   ```
3. Click **"Create Agent"**

### Step 4: Verify Agent Status

For each agent:

1. Click on the agent name in **"My Agents"** list
2. Scroll to **"Most Recent Agent Check"**
3. Click **"Check Again"** button
4. Verify:
   - **Controller Reachable**: Should show "Yes"
   - Agent Card should load successfully

If status shows "No":
- Wait 30 seconds and try again (cold start delay)
- Check that service URLs are correct
- Verify Cloud Run services are running

### Step 5: Create Assessment

1. Click **"My Assessments"** → **"+"** button
2. Select:
   ```
   Green Agent (Assessor): folio-green-agent
   White Agent (Assessee): folio-baseline-agent
   Config: default
   ```
3. Click **"Create Assessment"**

### Step 6: Run Evaluation

#### Option A: Via Platform UI

1. Find your assessment in the list
2. Click **"Start Assessment"** button  
3. The platform will:
   - Instantiate both agents
   - Send evaluation task to green agent
   - Display logs in real-time
4. View results when complete

#### Option B: Direct API Call

You can also test the evaluation directly:

```bash
# Test that agents are working
curl https://folio-baseline-agent-xxxxx-uc.a.run.app/status
curl https://folio-green-agent-xxxxx-uc.a.run.app/status

# Check agent cards
curl https://folio-baseline-agent-xxxxx-uc.a.run.app/.well-known/agent-card.json
curl https://folio-green-agent-xxxxx-uc.a.run.app/.well-known/agent-card.json
```

## Troubleshooting

### Agent Check Shows "Controller Reachable: No"

**Cause**: Cold start delay - Cloud Run services spin down when idle

**Solution**:
```bash
# Wake up the service
curl https://your-agent-url.run.app/status

# Wait 10-30 seconds, then click "Check Again"
```

### "Method not found" Error

**Cause**: Incorrect A2A protocol endpoint

**Solution**: Verify you're using the base URL, not a specific endpoint:
- ✅ Correct: `https://agent.run.app`
- ❌ Wrong: `https://agent.run.app/to_agent/xxx`

### Assessment Fails to Start

**Possible causes**:
1. **Platform bug**: `assert cagent_id is not None` error
   - This is a known platform issue
   - Verify your agents work via direct API testing
   
2. **Timeout**: Services took too long to respond
   - Increase Cloud Run timeout in deployment scripts
   - Consider setting `--min-instances 1` to avoid cold starts

3. **Configuration error**: Check assessment config format

## Agent Card Configuration

Each agent exposes an agent card at `/.well-known/agent-card.json`:

### Baseline Agent Card
```json
{
  "name": "baseline_reasoning_agent",
  "description": "Direct LLM reasoning for logical inference",
  "url": "https://folio-baseline-agent-xxxxx.run.app",
  "version": "1.0.0",
  "protocolVersion": "0.3.0",
  "preferredTransport": "JSONRPC",
  "defaultInputModes": ["text/plain"],
  "defaultOutputModes": ["text/plain"],
  "capabilities": {},
  "skills": [{
    "id": "logical_reasoning",
    "name": "Logical Reasoning",
    "description": "Determines if conclusions follow from premises"
  }]
}
```

### Green Agent Card
```json
{
  "name": "FOLIO Logical Reasoning Benchmark",
  "description": "Evaluates agents on first-order logic inference tasks",
  "url": "https://folio-green-agent-xxxxx.run.app",
  "version": "1.0.0",
  "protocolVersion": "0.3.0",
  "preferredTransport": "JSONRPC",
  "capabilities": {
    "streaming": false
  },
  "skills": [{
    "id": "folio_evaluation",
    "name": "FOLIO Evaluation",
    "description": "Assess agent performance on logical reasoning tasks"
  }]
}
```

## Evaluation Task Format

To evaluate a white agent, send this message to the green agent:

```xml
<white_agent_url>
https://folio-baseline-agent-xxxxx.run.app
</white_agent_url>
<max_examples>10</max_examples>
```

The green agent will:
1. Load 10 examples from FOLIO validation set
2. Send each problem to the white agent
3. Compare responses with ground truth
4. Return accuracy and performance metrics

## Cloud Run Configuration

### Baseline Agent
```
--memory: 1Gi
--cpu: 1
--timeout: 180s
--min-instances: 0
--max-instances: 5
```

### Green Agent
```
--memory: 2Gi
--cpu: 1
--timeout: 300s
--min-instances: 0
--max-instances: 5
```

## Cost Optimization

**Tips**:
- Set `--min-instances 0` to scale to zero when idle
- Use `--max-instances` to cap maximum cost
- Monitor usage in Google Cloud Console

## Security Notes

**API Keys**:
- Store as environment variables in Cloud Run
- Never commit API keys to Git
- Rotate keys regularly

**Authentication**:
- Use `--allow-unauthenticated` for AgentBeats compatibility
- For production, consider Cloud Run authentication

## References

- **AgentBeats Platform**: https://v2.agentbeats.org
- **A2A Protocol**: https://github.com/agentbeats/a2a-sdk
- **Google Cloud Run**: https://cloud.google.com/run/docs
- **FOLIO Dataset**: https://github.com/Yale-LILY/FOLIO

## Support

For issues:
1. Check Cloud Run logs: `gcloud run logs read SERVICE_NAME --region REGION`
2. Verify agent status: `curl https://your-agent.run.app/status`
3. Test agent card: `curl https://your-agent.run.app/.well-known/agent-card.json`
4. Contact course staff for platform issues
