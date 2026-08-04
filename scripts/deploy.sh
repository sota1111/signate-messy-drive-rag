#!/usr/bin/env bash
# Build + push the backend image and apply Terraform (Cloud Run + Vertex + GCS).
# Prereqs: gcloud auth, APIs enabled (Terraform enables them too), a built index_store/.
#
#   PROJECT_ID=signate-messy-drive-rag REGION=us-central1 bash scripts/deploy.sh
set -euo pipefail
export LANG=C.UTF-8 LC_ALL=C.UTF-8

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PROJECT_ID="${PROJECT_ID:-signate-messy-drive-rag}"
REGION="${REGION:-us-central1}"
REPO="rag"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/backend:$(date -u +%Y%m%d-%H%M%S 2>/dev/null || echo latest)"

[ -f index_store/embeddings.npy ] || { echo "ERROR: index_store missing — run 'python -m src.rag.index' first"; exit 1; }

echo "== staging minimal glossary corpus =="
rm -rf backend/_corpus && mkdir -p backend/_corpus
DRIVE="$(find data/share_drive -maxdepth 1 -type d -name '社内管理' -print -quit)"
cp -r "$DRIVE" backend/_corpus/

echo "== ensuring Artifact Registry repo =="
gcloud artifacts repositories describe "$REPO" --location="$REGION" --project="$PROJECT_ID" >/dev/null 2>&1 || \
  gcloud artifacts repositories create "$REPO" --repository-format=docker \
    --location="$REGION" --project="$PROJECT_ID" --description="RAG images"

echo "== building & pushing image: $IMAGE =="
gcloud builds submit "$ROOT" --project="$PROJECT_ID" --tag="$IMAGE" \
  --gcs-source-staging-dir="gs://${PROJECT_ID}_cloudbuild/source" --timeout=1200s \
  --ignore-file=.gcloudignore 2>/dev/null || \
gcloud builds submit "$ROOT" --project="$PROJECT_ID" --tag="$IMAGE" --timeout=1200s

echo "== terraform apply =="
cd infra/terraform
terraform init -input=false
terraform apply -input=false -auto-approve \
  -var="project_id=${PROJECT_ID}" -var="region=${REGION}" -var="backend_image=${IMAGE}"
terraform output -raw backend_url && echo

cd "$ROOT" && rm -rf backend/_corpus
echo "== done =="
