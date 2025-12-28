#!/bin/bash
# Deploy FOLIO Autoformalization Agent to Google Cloud Run

set -e

echo "=========================================="
echo "FOLIO Autoformalization Agent - Cloud Run Deployment"
echo "=========================================="
echo ""

# Source gcloud SDK if not in PATH
if ! command -v gcloud &> /dev/null; then
    # Try common gcloud locations
    GCLOUD_PATHS=(
        "$HOME/google-cloud-sdk/path.bash.inc"
        "/usr/local/google-cloud-sdk/path.bash.inc"
        "../../google-cloud-sdk/path.bash.inc"
    )
    
    for gcloud_path in "${GCLOUD_PATHS[@]}"; do
        if [ -f "$gcloud_path" ]; then
            echo "Found gcloud SDK at: $gcloud_path"
            source "$gcloud_path"
            break
        fi
    done
    
    # Check again after sourcing
    if ! command -v gcloud &> /dev/null; then
        echo "Error: gcloud command not found"
        echo "Please install Google Cloud SDK: https://cloud.google.com/sdk/docs/install"
        exit 1
    fi
fi

echo "✓ Google Cloud SDK found"
gcloud version | head -1
echo ""

# Configure project
echo "Step 1: Configure Google Cloud Project"
echo ""
PROJECT_ID=$(gcloud config get-value project 2>/dev/null)

if [ -z "$PROJECT_ID" ] || [ "$PROJECT_ID" == "(unset)" ]; then
    echo "Please enter your Google Cloud Project ID:"
    read PROJECT_ID
    gcloud config set project $PROJECT_ID
fi

echo "Current project: $PROJECT_ID"
echo ""
echo "✓ Using project: $PROJECT_ID"
echo ""

# Use existing region configuration
echo "Step 2: Using Deployment Region"
echo ""
REGION=$(gcloud config get-value run/region 2>/dev/null)

if [ -z "$REGION" ] || [ "$REGION" == "(unset)" ]; then
    REGION="us-central1"
    echo "Using default region: $REGION"
else
    echo "Using configured region: $REGION"
fi

# Configure API key
echo ""
echo "Step 3: Configure Gemini API Key"
echo ""

if [ -z "$GEMINI_API_KEY" ]; then
    echo "Enter your Gemini API key:"
    read -s GEMINI_API_KEY
    echo ""
fi

echo "✓ API key configured"

# Enable APIs
echo ""
echo "Step 4: Enable Required APIs"
echo ""
echo "Enabling Cloud Run API..."
gcloud services enable run.googleapis.com --quiet 2>/dev/null || true
echo "Enabling Cloud Build API..."
gcloud services enable cloudbuild.googleapis.com --quiet 2>/dev/null || true
echo "Enabling Container Registry API..."
gcloud services enable containerregistry.googleapis.com --quiet 2>/dev/null || true
echo ""
echo "✓ APIs enabled"

# Build image
echo ""
echo "Step 5: Build Docker Image"
echo ""

SERVICE_NAME="folio-autoform-agent"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

# Create temporary cloudbuild.yaml
cat > /tmp/cloudbuild-autoform.yaml << EOF
steps:
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', '${IMAGE_NAME}:latest', '-f', 'Dockerfile.autoform', '.']
images:
  - '${IMAGE_NAME}:latest'
EOF

gcloud builds submit --config=/tmp/cloudbuild-autoform.yaml .

rm -f /tmp/cloudbuild-autoform.yaml

echo ""
echo "✓ Image build complete"

# Deploy to Cloud Run
echo ""
echo "Step 6: Deploy to Cloud Run"
echo ""

# First, deploy without PUBLIC_URL (we need the service URL first)
gcloud run deploy ${SERVICE_NAME} \
    --image ${IMAGE_NAME}:latest \
    --platform managed \
    --region ${REGION} \
    --allow-unauthenticated \
    --set-env-vars "GEMINI_API_KEY=${GEMINI_API_KEY},AGENT_ROLE=autoform" \
    --memory 1Gi \
    --cpu 1 \
    --timeout 300s \
    --min-instances 1 \
    --max-instances 5 \
    --quiet

# Get service URL
echo ""
echo "Retrieving service URL..."
SERVICE_URL=$(gcloud run services describe ${SERVICE_NAME} \
    --platform managed \
    --region ${REGION} \
    --format 'value(status.url)')

echo "✓ Service deployed at: ${SERVICE_URL}"

# Extract domain from SERVICE_URL (remove https:// prefix for CLOUDRUN_HOST)
CLOUDRUN_HOST="${SERVICE_URL#https://}"

# Update PUBLIC_URL, CLOUDRUN_HOST, and HTTPS_ENABLED environment variables
echo ""
echo "Updating environment variables with PUBLIC_URL, CLOUDRUN_HOST, and HTTPS_ENABLED..."
gcloud run services update ${SERVICE_NAME} \
    --platform managed \
    --region ${REGION} \
    --update-env-vars "GEMINI_API_KEY=${GEMINI_API_KEY},PUBLIC_URL=${SERVICE_URL},CLOUDRUN_HOST=${CLOUDRUN_HOST},HTTPS_ENABLED=true,AGENT_ROLE=autoform" \
    --quiet

echo ""
echo "=========================================="
echo "✓ Deployment Complete!"
echo "=========================================="
echo ""
echo "Service URL: ${SERVICE_URL}"
echo ""
echo "Test your deployment:"
echo "  curl ${SERVICE_URL}/status"
echo "  curl ${SERVICE_URL}/.well-known/agent-card.json"
echo ""
echo "Next steps:"
echo "  1. Register agent on AgentBeats: https://v2.agentbeats.org"
echo "  2. Agent Name: folio-autoform-agent"
echo "  3. Deploy Type: Remote"
echo "  4. Controller URL: ${SERVICE_URL}"
echo "  5. Agent Type: Assessee (White Agent)"
echo ""
