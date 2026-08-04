output "backend_url" {
  value       = google_cloud_run_v2_service.backend.uri
  description = "Public URL of the RAG backend."
}

output "runtime_service_account" {
  value = google_service_account.runtime.email
}

output "index_bucket" {
  value = google_storage_bucket.index.name
}

output "artifact_registry" {
  value = "${var.region}-docker.pkg.dev/${var.project_id}/${var.artifact_registry_repository}"
}
