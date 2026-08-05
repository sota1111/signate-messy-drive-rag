# infra/terraform — signate-messy-drive-rag backend IaC

Terraform for the **production backend only**: a single Cloud Run service that serves the
messy-drive RAG on **Vertex AI (Gemini generation + embeddings)**, plus the GCS bucket and
least-privilege runtime service account it needs. Serving shape mirrors
[`toddler-private-rag`](https://github.com/sota1111/toddler-private-rag) (Cloud Run + Vertex +
GCS) but is deliberately scoped down to what this SIGNATE competition backend actually requires.

## Is this "backend-only" extraction complete? — Yes

This module provisions everything the backend needs to run, and nothing it doesn't. The backend
(`backend/app/main.py`, `src/rag/*`) reads these runtime settings, all wired here. Since SOT-2479
`/ask` routes to the Vertex **Gemini agent** stack (`src.rag.agent.gate`), so the Gemini model
selection is pinned in IaC to keep the deploy reproducible and Claude-independent (SOT-2480):

| Env var | Wired by | Purpose |
|---|---|---|
| `GCP_PROJECT_ID` | `main.tf` Cloud Run env | Vertex project (`src/rag/llm.py`) |
| `VERTEX_LOCATION` | `main.tf` Cloud Run env | Vertex region for Gemini + embeddings |
| `GEN_MODEL` / `GEN_MODEL_HARD` | `main.tf` Cloud Run env | Gemini generation models (default + hard-question escalation) |
| `VISION_MODEL` | `main.tf` Cloud Run env | Gemini model for chart / image-PDF understanding |
| `EMBED_MODEL` | `main.tf` Cloud Run env | Vertex text-embedding model for dense retrieval |
| `INDEX_DIR` | `backend/Dockerfile` (`/app/index_store`) | Retrieval index baked into the image |
| `CORPUS_DIR` | `backend/Dockerfile` (`/app/corpus`) | Minimal glossary corpus baked into the image |

> The serving image (`backend/requirements.txt`) also ships the Office-extraction deps the agent
> stack imports at load time (`openpyxl` / `python-pptx` / `msoffcrypto-tool`) — without them the
> first real `/ask` fails with `ModuleNotFoundError` while `/health` (static) stays up.

There is **no auth** (public demo endpoint), so no Secret Manager references are needed, and the
retrieval index/corpus are **baked into the container image** at build time, so there is **no
runtime database**. That is why the resource set is much smaller than toddler-private-rag's.

### What toddler-private-rag has that this does NOT — and why that is correct

toddler-private-rag is a full private, authenticated, multi-service web app with CI/CD and
production observability. Every resource it has that is absent here is an **app-specific feature
that this competition backend does not use**:

| toddler resource(s) | Omitted here because |
|---|---|
| `cloud_run` frontend + `cloud_run_upload` service | No UI / no user upload — this is a headless RAG API + offline harness |
| `firestore.tf` | No runtime DB; the index/corpus are baked into the image |
| `secrets.tf` + secret env refs | No auth (public demo); nothing sensitive to inject at runtime |
| `pubsub.tf`, `scheduler.tf`, `remediation.tf` | Serve upload-finalize events + orphan cleanup — no upload flow here |
| `monitoring.tf`, `dashboard.tf`, `slo.tf` | Production observability; out of scope for the competition backend |
| `wif.tf`, deploy SA in `iam.tf`, `google-beta` provider | GitHub Actions CI/CD via Workload Identity Federation — this repo deploys locally via `scripts/deploy.sh` |
| `artifact_registry.tf` | **Intentionally not managed here** — `scripts/deploy.sh` creates the Docker repo *before* the image build (the build must push into an existing repo), avoiding a create/push race. See the comment in `main.tf`. |

Conversely, toddler's proven backend health-probe pattern **is** applied here: the Cloud Run
service declares `/health` startup + liveness probes (the backend already exposes `GET /health`),
and `deletion_protection = false` keeps the ephemeral competition project easy to tear down.

## What this provisions

- **APIs** (`google_project_service`): run, artifactregistry, aiplatform, storage, cloudbuild, iam,
  cloudresourcemanager (`disable_on_destroy = false` so `destroy` never turns an API off).
- **GCS bucket** `${project_id}-index` — retrieval index artifacts (uniform access, `force_destroy`).
- **Runtime service account** `rag-backend-runtime` (least privilege): `roles/aiplatform.user`
  (Vertex) + `roles/storage.objectViewer` on the index bucket only.
- **Cloud Run v2 service** (`var.backend_service`) running as the runtime SA, 2 vCPU / 2Gi,
  scale 0→3, `/health` probes, backend env wired.
- **Public invoker** binding (`allUsers` → `roles/run.invoker`), gated by `var.allow_unauthenticated`.

## Usage

Terraform is normally driven by `scripts/deploy.sh` from the repo root (it builds + pushes the
image and ensures the Artifact Registry repo first, then runs `apply` with the freshly-pushed
image tag):

```bash
# From repo root — builds index_store first if needed, then deploy:
python -m src.rag.index          # produce index_store/embeddings.npy (prereq)
PROJECT_ID=signate-messy-drive-rag REGION=us-central1 bash scripts/deploy.sh
```

To run Terraform directly (e.g. to inspect a plan), you must supply `backend_image` yourself:

```bash
cd infra/terraform
cp terraform.tfvars.example terraform.tfvars   # then edit backend_image
terraform init
terraform plan
terraform apply
```

Prereqs: authenticated `gcloud` (`gcloud auth application-default login`), billing enabled on the
project. The APIs are enabled by Terraform, but the very first `plan`/`apply` may need
Cloud Resource Manager / IAM already on.

## Variables

See `variables.tf`. Key ones: `project_id`, `region`, `vertex_location`, `gcs_location`,
`artifact_registry_repository`, `backend_service`, **`backend_image`** (required — full image ref
set by `scripts/deploy.sh`), `allow_unauthenticated` (default `true`; set `false` to require IAM
invoker).

## Outputs

`backend_url` (public Cloud Run URL), `runtime_service_account`, `index_bucket`,
`artifact_registry` (the `…-docker.pkg.dev/…/rag` repo path).

## State

Local state by default (single-operator, simple). To share state, switch to a GCS backend in
`versions.tf` and pre-create a versioned bucket:

```hcl
terraform {
  backend "gcs" {
    bucket = "signate-messy-drive-rag-tfstate"
    prefix = "terraform"
  }
}
```

`terraform.tfstate*` and `.terraform/` are git-ignored — do not commit them.
