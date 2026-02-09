#!/bin/bash
set -e

# --- CONFIGURATION ---
PROJECT_ID="lloyal-node"
GITHUB_REPO="lloyal-ai/lloyal.node"
REGION="us-east4"

# Infrastructure Names
SA_NAME="github-ci-runner"
POOL_NAME="github-pool"
PROVIDER_NAME="github-provider"
AR_REPO="lloyal-ci"

echo "=== Provisioning GCP Infrastructure for $GITHUB_REPO ==="
echo "Project: $PROJECT_ID"

# 1. Setup Project & APIs
gcloud config set project "$PROJECT_ID"

echo "Enabling APIs..."
gcloud services enable \
  iam.googleapis.com \
  iamcredentials.googleapis.com \
  artifactregistry.googleapis.com \
  run.googleapis.com \
  logging.googleapis.com

# 2. Artifact Registry
if ! gcloud artifacts repositories describe "$AR_REPO" --location="$REGION" &>/dev/null; then
    echo "Creating Artifact Registry repo..."
    gcloud artifacts repositories create "$AR_REPO" \
      --repository-format=docker \
      --location="$REGION" \
      --description="Docker repository for lloyal.node CI"
else
    echo "Artifact Registry repo exists."
fi

# 3. Service Account
SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
if ! gcloud iam service-accounts describe "$SA_EMAIL" &>/dev/null; then
    echo "Creating Service Account..."
    gcloud iam service-accounts create "$SA_NAME" --display-name="GitHub Actions CI Runner"
else
    echo "Service Account exists."
fi

# 4. Assign Roles
echo "Assigning IAM roles..."

# Grant Artifact Registry Writer (Push images)
gcloud artifacts repositories add-iam-policy-binding "$AR_REPO" \
  --location="$REGION" \
  --member="serviceAccount:$SA_EMAIL" \
  --role="roles/artifactregistry.writer" > /dev/null

# Grant Cloud Run Developer (Create/Update Jobs)
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:$SA_EMAIL" \
  --role="roles/run.developer" > /dev/null

# Grant Cloud Run Invoker (Execute Jobs)
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:$SA_EMAIL" \
  --role="roles/run.invoker" > /dev/null

# Grant Logging Viewer (Read logs back to CI)
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:$SA_EMAIL" \
  --role="roles/logging.viewer" > /dev/null

# Grant Service Account User (Act as itself)
gcloud iam service-accounts add-iam-policy-binding "$SA_EMAIL" \
  --member="serviceAccount:$SA_EMAIL" \
  --role="roles/iam.serviceAccountUser" > /dev/null

# 5. Workload Identity Federation
if ! gcloud iam workload-identity-pools describe "$POOL_NAME" --location="global" &>/dev/null; then
    echo "Creating Identity Pool..."
    gcloud iam workload-identity-pools create "$POOL_NAME" \
      --location="global" \
      --display-name="GitHub Actions Pool"
fi

POOL_ID=$(gcloud iam workload-identity-pools describe "$POOL_NAME" --location="global" --format="value(name)")

# Create Provider with Security Condition
if ! gcloud iam workload-identity-pools providers describe "$PROVIDER_NAME" --location="global" --workload-identity-pool="$POOL_NAME" &>/dev/null; then
    echo "Creating Identity Provider..."
    gcloud iam workload-identity-pools providers create-oidc "$PROVIDER_NAME" \
      --location="global" \
      --workload-identity-pool="$POOL_NAME" \
      --display-name="GitHub Provider" \
      --attribute-mapping="google.subject=assertion.sub,attribute.actor=assertion.actor,attribute.repository=assertion.repository" \
      --attribute-condition="assertion.repository_owner == 'lloyal-ai'" \
      --issuer-uri="https://token.actions.githubusercontent.com"
fi

echo "Binding GitHub Repo to Service Account..."
gcloud iam service-accounts add-iam-policy-binding "$SA_EMAIL" \
  --role="roles/iam.workloadIdentityUser" \
  --member="principalSet://iam.googleapis.com/${POOL_ID}/attribute.repository/${GITHUB_REPO}" > /dev/null

# --- OUTPUT ---
PROVIDER_FULL_PATH=$(gcloud iam workload-identity-pools providers describe "$PROVIDER_NAME" --location="global" --workload-identity-pool="$POOL_NAME" --format="value(name)")

echo ""
echo "✅ Infrastructure Setup Complete!"
echo "Secrets for GitHub:"
echo "GCP_PROJECT_ID   : $PROJECT_ID"
echo "GCP_SA_EMAIL     : $SA_EMAIL"
echo "GCP_WIF_PROVIDER : $PROVIDER_FULL_PATH"