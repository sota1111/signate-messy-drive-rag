terraform {
  required_version = ">= 1.5"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 6.0"
    }
  }

  # Local state by default (simple, single-operator). To share state, switch to a GCS
  # backend and pre-create a versioned bucket:
  #   backend "gcs" { bucket = "signate-messy-drive-rag-tfstate"  prefix = "terraform" }
}
