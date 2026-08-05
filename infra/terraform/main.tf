provider "google" {
  project = var.project_id
  region  = var.region
}

# ---------------- APIs ----------------
locals {
  required_apis = [
    "run.googleapis.com",
    "artifactregistry.googleapis.com",
    "aiplatform.googleapis.com",
    "storage.googleapis.com",
    "cloudbuild.googleapis.com",
    "iam.googleapis.com",
    "cloudresourcemanager.googleapis.com",
  ]
}

resource "google_project_service" "services" {
  for_each           = toset(local.required_apis)
  project            = var.project_id
  service            = each.value
  disable_on_destroy = false
}

# ---------------- Artifact Registry ----------------
# The Docker repo is created by scripts/deploy.sh BEFORE the image build (the build must push
# into an existing repo), so it is intentionally NOT managed here to avoid a create race.

# ---------------- GCS bucket (retrieval index artifacts) ----------------
resource "google_storage_bucket" "index" {
  project                     = var.project_id
  name                        = "${var.project_id}-index"
  location                    = var.gcs_location
  uniform_bucket_level_access = true
  force_destroy               = true
  depends_on                  = [google_project_service.services]
}

# ---------------- Runtime service account (least privilege) ----------------
resource "google_service_account" "runtime" {
  project      = var.project_id
  account_id   = "rag-backend-runtime"
  display_name = "signate-messy-drive-rag Cloud Run runtime SA"
}

# Vertex AI (Gemini generation + embeddings)
resource "google_project_iam_member" "vertex_user" {
  project = var.project_id
  role    = "roles/aiplatform.user"
  member  = "serviceAccount:${google_service_account.runtime.email}"
}

# Read the index bucket
resource "google_storage_bucket_iam_member" "index_reader" {
  bucket = google_storage_bucket.index.name
  role   = "roles/storage.objectViewer"
  member = "serviceAccount:${google_service_account.runtime.email}"
}

# ---------------- Cloud Run service ----------------
resource "google_cloud_run_v2_service" "backend" {
  project = var.project_id
  name    = var.backend_service
  # Ephemeral competition/demo project — allow `terraform destroy` to tear the
  # service down without first flipping deletion_protection (matches toddler-private-rag).
  deletion_protection = false
  location            = var.region
  ingress             = "INGRESS_TRAFFIC_ALL"

  template {
    service_account = google_service_account.runtime.email
    timeout         = "300s"
    scaling {
      min_instance_count = 0
      max_instance_count = 3
    }
    containers {
      image = var.backend_image
      resources {
        limits = { cpu = "2", memory = "2Gi" }
      }
      # /health (backend/app/main.py) returns static ok and lazy-loads the index,
      # so it is safe to poll during warm-up. Startup probe gates traffic until
      # ready; liveness probe restarts a hung container.
      startup_probe {
        http_get { path = "/health" }
        initial_delay_seconds = 0
        timeout_seconds       = 5
        period_seconds        = 10
        failure_threshold     = 6
      }
      liveness_probe {
        http_get { path = "/health" }
        initial_delay_seconds = 10
        timeout_seconds       = 5
        period_seconds        = 30
        failure_threshold     = 3
      }
      env {
        name  = "GCP_PROJECT_ID"
        value = var.project_id
      }
      env {
        name  = "VERTEX_LOCATION"
        value = var.vertex_location
      }
      # Gemini-only model wiring (SOT-2480): pin the Vertex models explicitly so the
      # deployed backend is Claude-independent and reproducible from IaC, not from code defaults.
      env {
        name  = "GEN_MODEL"
        value = var.gen_model
      }
      env {
        name  = "GEN_MODEL_HARD"
        value = var.gen_model_hard
      }
      env {
        name  = "VISION_MODEL"
        value = var.vision_model
      }
      env {
        name  = "EMBED_MODEL"
        value = var.embed_model
      }
      env {
        name  = "INDEX_DIR"
        value = "/app/index_store"
      }
      env {
        name  = "CORPUS_DIR"
        value = "/app/corpus"
      }
    }
  }
  depends_on = [google_project_service.services]
}

# Public endpoint (demo). Restrict with IAM invoker for private deployments.
resource "google_cloud_run_v2_service_iam_member" "public" {
  count    = var.allow_unauthenticated ? 1 : 0
  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_service.backend.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}
