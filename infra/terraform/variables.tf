variable "project_id" {
  type        = string
  description = "GCP project ID."
  default     = "signate-messy-drive-rag"
}

variable "region" {
  type    = string
  default = "us-central1"
}

variable "vertex_location" {
  type        = string
  description = "Vertex AI location for Gemini + embeddings."
  default     = "us-central1"
}

variable "gcs_location" {
  type    = string
  default = "US"
}

variable "artifact_registry_repository" {
  type    = string
  default = "rag"
}

variable "backend_service" {
  type    = string
  default = "signate-messy-drive-rag-backend"
}

variable "backend_image" {
  type        = string
  description = "Full image ref pushed by scripts/deploy.sh (…-docker.pkg.dev/…/rag/backend:TAG)."
}

variable "allow_unauthenticated" {
  type        = bool
  description = "Expose the Cloud Run service publicly (demo). Set false to require IAM."
  default     = true
}
