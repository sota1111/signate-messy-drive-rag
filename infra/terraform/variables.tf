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

# ---- Gemini-only model wiring (SOT-2480) ----
# Pinned in IaC so the Cloud Run backend runs Vertex Gemini exclusively (Claude-independent),
# reproducibly, without relying on the code defaults in config/settings.py.
variable "gen_model" {
  type        = string
  description = "Default Gemini generation model."
  default     = "gemini-2.5-flash"
}

variable "gen_model_hard" {
  type        = string
  description = "Escalation Gemini model for hard / low-confidence questions."
  default     = "gemini-2.5-pro"
}

variable "vision_model" {
  type        = string
  description = "Gemini model for chart / image-PDF understanding."
  default     = "gemini-2.5-flash"
}

variable "embed_model" {
  type        = string
  description = "Vertex text embedding model for dense retrieval."
  default     = "text-embedding-005"
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
