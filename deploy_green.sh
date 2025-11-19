#!/bin/bash
# Deploy FOLIO Green Agent to Google Cloud Run

set -e

echo "=========================================="
echo "FOLIO Green Agent - Cloud Run Deployment"
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
fi

echo ""
echo "✓ Using project: $PROJECT_ID"
echo ""

# Select region
REGION="us-central1"
echo "Using region: $REGION"
echo ""

# Enable required APIs
echo "Step 2: Enable Required APIs"
echo ""
gcloud services enable run.googleapis.com --project=$PROJECT_ID
gcloud services enable cloudbuild.googleapis.com --project=$PROJECT_ID
gcloud services enable containerregistry.googleapis.com --project=$PROJECT_ID

echo ""
echo "✓ APIs enabled"
echo ""

# Build Docker image
echo "Step 3: Build Docker Image"
echo ""

cat > /tmp/cloudbuild-green.yaml << 'EOF'
steps:
- name: 'gcr.io/cloud-builders/docker'
  args: ['build', '-t', 'gcr.io/$PROJECT_ID/folio-green-agent', '-f', 'Dockerfile.green', '.']
images:
- 'gcr.io/$PROJECT_ID/folio-green-agent'
EOF

gcloud builds submit --config=/tmp/cloudbuild-green.yaml .

echo ""
echo "✓ Image build complete"
echo ""

# Deploy to Cloud Run
echo "Step 4: Deploy to Cloud Run"
echo ""

# Initial deployment without PUBLIC_URL
gcloud run deploy folio-green-agent \
  --image gcr.io/$PROJECT_ID/folio-green-agent \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated \
  --port 8080 \
  --memory 2Gi \
  --timeout 300 \
  --min-instances 0 \
  --max-instances 5 \
  --quiet

echo ""
echo "Retrieving service URL..."
SERVICE_URL=$(gcloud run services describe folio-green-agent --region=$REGION --format='value(status.url)')

echo "✓ Service deployed at: $SERVICE_URL"
echo ""

# Update with PUBLIC_URL
echo "Updating environment variables with PUBLIC_URL..."
gcloud run services update folio-green-agent \
  --region $REGION \
  --update-env-vars "PUBLIC_URL=$SERVICE_URL" \
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
echo "  2. Agent Name: folio-green-agent"
echo "  3. Deploy Type: Remote"
echo "  4. Controller URL: $SERVICE_URL"
echo "  5. Agent Type: Assessor (Green Agent)"
echo ""
