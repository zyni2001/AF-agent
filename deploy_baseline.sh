#!/bin/bash
# Deploy FOLIO Baseline Agent to Google Cloud Run

set -e

echo "=========================================="
echo "FOLIO Baseline Agent - Cloud Run Deployment"
echo "=========================================="
echo ""

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "Error: gcloud command not found"
    echo "Please install Google Cloud SDK: https://cloud.google.com/sdk/docs/install"
    exit 1
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
else
    echo "Current project: $PROJECT_ID"
    echo "Use this project? (y/n)"
    read USE_CURRENT
    if [ "$USE_CURRENT" != "y" ]; then
        echo "Enter new project ID:"
        read PROJECT_ID
        gcloud config set project $PROJECT_ID
    fi
fi

echo ""
echo "✓ Using project: $PROJECT_ID"
echo ""

# Select region
echo "Step 2: Select Deployment Region"
echo ""
echo "Recommended regions:"
echo "  1) us-central1 (US Central - recommended)"
echo "  2) us-west1 (US West)"
echo "  3) europe-west1 (Europe)"
echo "  4) asia-east1 (Taiwan)"
echo ""
echo "Enter choice (1-4):"
read REGION_CHOICE

case $REGION_CHOICE in
    1)
        REGION="us-central1"
        ;;
    2)
        REGION="us-west1"
        ;;
    3)
        REGION="europe-west1"
        ;;
    4)
        REGION="asia-east1"
        ;;
    *)
        echo "Using default: us-central1"
        REGION="us-central1"
        ;;
esac

echo "Selected region: $REGION"
echo ""

# Get API key
echo "Step 3: Configure Gemini API Key"
echo ""
echo "Enter your Gemini API key:"
read API_KEY

if [ -z "$API_KEY" ]; then
    echo "Error: API key is required"
    echo "You can get a Gemini API key at: https://aistudio.google.com/app/apikey"
    exit 1
fi

echo ""
echo "✓ API key configured"
echo ""

# Enable required APIs
echo "Step 4: Enable Required APIs"
echo ""
echo "Enabling Cloud Run API..."
gcloud services enable run.googleapis.com --project=$PROJECT_ID

echo "Enabling Cloud Build API..."
gcloud services enable cloudbuild.googleapis.com --project=$PROJECT_ID

echo "Enabling Container Registry API..."
gcloud services enable containerregistry.googleapis.com --project=$PROJECT_ID

echo ""
echo "✓ APIs enabled"
echo ""

# Build Docker image
echo "Step 5: Build Docker Image"
echo ""

cat > /tmp/cloudbuild-baseline.yaml << 'EOF'
steps:
- name: 'gcr.io/cloud-builders/docker'
  args: ['build', '-t', 'gcr.io/$PROJECT_ID/folio-baseline-agent', '-f', 'Dockerfile.baseline', '.']
images:
- 'gcr.io/$PROJECT_ID/folio-baseline-agent'
EOF

gcloud builds submit --config=/tmp/cloudbuild-baseline.yaml .

echo ""
echo "✓ Image build complete"
echo ""

# Deploy to Cloud Run
echo "Step 6: Deploy to Cloud Run"
echo ""

# Initial deployment without PUBLIC_URL
gcloud run deploy folio-baseline-agent \
  --image gcr.io/$PROJECT_ID/folio-baseline-agent \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated \
  --port 8080 \
  --memory 1Gi \
  --cpu 1 \
  --timeout 180 \
  --min-instances 0 \
  --max-instances 5 \
  --set-env-vars "GEMINI_API_KEY=$API_KEY" \
  --quiet

echo ""
echo "Retrieving service URL..."
SERVICE_URL=$(gcloud run services describe folio-baseline-agent --region=$REGION --format='value(status.url)')

echo "✓ Service deployed at: $SERVICE_URL"
echo ""

# Update with PUBLIC_URL
echo "Updating environment variables with PUBLIC_URL..."
gcloud run services update folio-baseline-agent \
  --region $REGION \
  --update-env-vars "GEMINI_API_KEY=$API_KEY,PUBLIC_URL=$SERVICE_URL" \
  --quiet

echo ""
echo "=========================================="
echo "✓ Deployment Complete!"
echo "=========================================="
echo ""
echo "Service URL: $SERVICE_URL"
echo ""
echo "Test your deployment:"
echo "  curl $SERVICE_URL/status"
echo "  curl $SERVICE_URL/.well-known/agent-card.json"
echo ""
echo "Next steps:"
echo "  1. Register agent on AgentBeats: https://v2.agentbeats.org"
echo "  2. Agent Name: folio-baseline-agent"
echo "  3. Deploy Type: Remote"
echo "  4. Controller URL: $SERVICE_URL"
echo "  5. Agent Type: Assessee (White Agent)"
echo ""
