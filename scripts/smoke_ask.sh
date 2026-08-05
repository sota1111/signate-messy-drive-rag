#!/usr/bin/env bash
# Post-deploy smoke test for the deployed `/ask` service (SOT-2481).
#
# Verifies two acceptance criteria for the production Cloud Run backend:
#   1. Representative question TYPES (復号 / 書式 / 計算 / 列挙 / chart) each receive a
#      *calibrated* decision — a correct answer OR a calibrated abstention (わかりません).
#      Under the official rubric (Correct +1 / Incorrect −1 / Missing 0) an abstention is a
#      legitimate, EV-neutral outcome, so the only failure is a *detectable wrong answer*
#      (the service committed a value that does not match the gold). The lean serving image
#      ships only the 社内管理 glossary corpus, so questions whose source file is not shipped
#      are expected to abstain — that is a PASS, not a failure.
#   2. The answer path is Gemini-only (Claude-independent): the import closure of the serving
#      entrypoint `backend.app.main` reaches NO `anthropic` / `opus_gen` / `claude` module, and
#      the serving deps (`backend/requirements.txt`) declare no `anthropic`. This is checked
#      statically (no code executed) so it is faithful to what the image actually ships.
#
# Usage:
#   bash scripts/smoke_ask.sh
#   SMOKE_URL=https://my-service.run.app bash scripts/smoke_ask.sh
#
# Environment:
#   SMOKE_URL        Base URL of the deployed service. Default: `terraform output backend_url`
#                    (from infra/terraform), else the known production URL.
#   SMOKE_TIMEOUT    Per-request curl timeout in seconds (default 180; a live /ask can take ~90s).
#   SMOKE_SKIP_LIVE  When truthy (1/true/yes/on), skip the live /ask calls and run ONLY the
#                    static Claude-independence check (offline / CI-friendly).
#
# Exit code: 0 when every representative question is correct-or-abstain AND the answer path is
# Gemini-only; non-zero otherwise.
set -euo pipefail
export LANG=C.UTF-8 LC_ALL=C.UTF-8

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PYTHON:-python3}"
SMOKE_TIMEOUT="${SMOKE_TIMEOUT:-180}"

is_truthy() { case "${1:-}" in 1|true|TRUE|yes|YES|on|ON) return 0 ;; *) return 1 ;; esac; }

# --------------------------------------------------------------------------- resolve base URL
resolve_url() {
  if [ -n "${SMOKE_URL:-}" ]; then printf '%s' "$SMOKE_URL"; return; fi
  local u=""
  if command -v terraform >/dev/null 2>&1 && [ -d infra/terraform ]; then
    u="$(terraform -chdir=infra/terraform output -raw backend_url 2>/dev/null || true)"
  fi
  if [ -z "$u" ]; then
    u="https://signate-messy-drive-rag-backend-4kvjtj6qvq-uc.a.run.app"
  fi
  printf '%s' "$u"
}
BASE_URL="$(resolve_url)"
BASE_URL="${BASE_URL%/}"

# Representative question types → 0-based index into data/questions/questions_valid.csv.
# One question per SIGNATE archetype the deployed service must handle (issue SOT-2481).
#   復号   = encrypted-Office decryption path (工数按分の税込請求額)
#   書式   = format/highlight extraction (赤強調文字列)
#   計算   = numeric aggregation (消費税額総額)
#   列挙   = enumerate-all set answer (マーカー単語)
#   chart  = figure/chart reading (figure_06.png)
REPRESENTATIVE="復号:8 書式:25 計算:3 列挙:0 chart:1"

fail=0

echo "== signate-messy-drive-rag post-deploy smoke =="
echo "target: $BASE_URL"
echo

# ===========================================================================================
# Part B (always runs, first): static Claude-independence of the /ask answer path.
# ===========================================================================================
echo "-- [1/2] Claude-independence (Gemini-only answer path) --"
if "$PY" - "$ROOT" <<'PY'
import ast, os, sys

root = sys.argv[1]
FORBIDDEN_TOPLEVEL = {"anthropic"}                       # third-party Claude SDK
FORBIDDEN_FIRSTPARTY = {"src.rag.opus_gen"}              # in-repo Claude/opus generation module
FORBIDDEN_NAME_SUBSTR = ("anthropic", "claude")          # defensive: any import mentioning these

def mod_to_file(mod):
    """Resolve a dotted module name to an in-repo file, or None if it is third-party/stdlib."""
    parts = mod.split(".")
    cand_pkg = os.path.join(root, *parts, "__init__.py")
    cand_mod = os.path.join(root, *parts) + ".py"
    if os.path.isfile(cand_mod):
        return cand_mod, False
    if os.path.isfile(cand_pkg):
        return cand_pkg, True
    return None, False

def imports_of(path):
    """Yield fully-qualified module names imported by the file at `path`."""
    with open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), filename=path)
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:            # relative import; resolve against the file's package
                pkg = os.path.relpath(os.path.dirname(path), root).replace(os.sep, ".")
                base = pkg.rsplit(".", node.level - 1)[0] if node.level > 1 else pkg
                mod = f"{base}.{node.module}" if node.module else base
            else:
                mod = node.module or ""
            if mod:
                out.append(mod)
                out += [f"{mod}.{a.name}" for a in node.names]  # e.g. `from pkg import sub`
    return out

# Walk the first-party import closure starting from the serving entrypoint.
start = "backend.app.main"
seen, stack, violations = set(), [start], []
while stack:
    mod = stack.pop()
    if mod in seen:
        continue
    seen.add(mod)

    top = mod.split(".")[0]
    low = mod.lower()
    if top in FORBIDDEN_TOPLEVEL or any(s in low for s in FORBIDDEN_NAME_SUBSTR):
        violations.append(mod)
        continue
    if mod in FORBIDDEN_FIRSTPARTY:
        violations.append(mod)
        continue

    path, _is_pkg = mod_to_file(mod)
    if path is None:            # third-party / stdlib leaf — not part of our closure to expand
        continue
    for imp in imports_of(path):
        if imp not in seen:
            stack.append(imp)

fp_modules = sorted(m for m in seen if mod_to_file(m)[0] is not None)
print(f"   reachable first-party modules from {start}: {len(fp_modules)}")
if violations:
    print("   FAIL: Claude/anthropic/opus reachable from the answer path:")
    for v in sorted(set(violations)):
        print(f"     - {v}")
    sys.exit(1)
print("   OK: no anthropic / opus_gen / claude module reachable from backend.app.main")
sys.exit(0)
PY
then
  echo "   [closure] PASS"
else
  echo "   [closure] FAIL"; fail=1
fi

# Serving dependency manifest must not declare the Claude SDK.
if grep -Eiq '^[[:space:]]*anthropic([=<>!~[:space:]]|$)' backend/requirements.txt; then
  echo "   [deps] FAIL: backend/requirements.txt declares 'anthropic'"; fail=1
else
  echo "   [deps] PASS: backend/requirements.txt declares no 'anthropic'"
fi
echo

# ===========================================================================================
# Part A: live representative-question smoke against the deployed /ask.
# ===========================================================================================
echo "-- [2/2] representative question types (correct-or-abstain) --"
if is_truthy "${SMOKE_SKIP_LIVE:-}"; then
  echo "   SMOKE_SKIP_LIVE set — skipping live /ask calls."
  echo
  if [ "$fail" -eq 0 ]; then echo "SMOKE: PASS (static-only)"; exit 0
  else echo "SMOKE: FAIL"; exit 1; fi
fi

# Liveness: /health must be up before we spend live /ask calls.
if ! curl -fsS -m 20 "$BASE_URL/health" >/dev/null 2>&1; then
  echo "   FAIL: $BASE_URL/health is not reachable"; echo; echo "SMOKE: FAIL"; exit 1
fi

QCSV="data/questions/questions_valid.csv"
GCSV="data/questions/valid_txt.csv"

# Pull the question text + gold answer for a 0-based index out of the valid set.
row_field() {  # row_field <index> <question|gold>
  "$PY" - "$QCSV" "$GCSV" "$1" "$2" <<'PY'
import csv, sys
qcsv, gcsv, idx, field = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
def load(path, has_header):
    rows = {}
    with open(path, encoding="utf-8-sig", newline="") as fh:
        r = list(csv.reader(fh))
    if has_header:
        r = r[1:]
    for row in r:
        if row:
            rows[int(row[0])] = row
    return rows
q = load(qcsv, True)
g = load(gcsv, False)
if field == "question":
    print(q[idx][1])
else:
    print(g[idx][1] if len(g.get(idx, [])) > 1 else "")
PY
}

# Classify one /ask response body against the gold answer.
#   prints one of: CORRECT | ABSTAIN | WRONG | ERROR   then a short detail on the next line.
classify_response() {  # classify_response <gold> <response-json-file>
  "$PY" - "$1" "$2" <<'PY'
import json, sys, unicodedata, re
gold = sys.argv[1]
try:
    with open(sys.argv[2], encoding="utf-8") as fh:
        data = json.load(fh)
except Exception as e:
    print("ERROR"); print(f"non-JSON response: {e}"); sys.exit(0)

for k in ("answer", "gate_status"):
    if k not in data:
        print("ERROR"); print(f"missing field '{k}' in response schema"); sys.exit(0)

answer = (data.get("answer") or "").strip()
gate = (data.get("gate_status") or "").strip()
ABSTAIN = "わかりません"

def norm(s):
    s = unicodedata.normalize("NFKC", s or "")
    s = s.lower()
    s = re.sub(r"[\s、,，・。．\.\-—–/／()（）「」『』\"'`　円jpy]", "", s)
    return s

if gate == "abstain" or answer == ABSTAIN or answer == "":
    print("ABSTAIN"); print(f"gate_status={gate!r} answer={answer!r}"); sys.exit(0)

a, g = norm(answer), norm(gold)
match = False
if g and (g in a or a in g):
    match = True
else:
    gt = [t for t in re.split(r"[、,，・/／\s]+", gold) if t]
    at = norm(answer)
    if gt:
        hit = sum(1 for t in gt if norm(t) and norm(t) in at)
        if hit / len(gt) >= 0.6:
            match = True
if match:
    print("CORRECT"); print(f"answer={answer!r}")
else:
    print("WRONG"); print(f"committed={answer!r} gold={gold!r}")
PY
}

printf '   %-6s %-8s %s\n' "TYPE" "VERDICT" "DETAIL"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
for entry in $REPRESENTATIVE; do
  typ="${entry%%:*}"; idx="${entry##*:}"
  q="$(row_field "$idx" question)"
  gold="$(row_field "$idx" gold)"
  body="$("$PY" -c 'import json,sys; print(json.dumps({"question": sys.argv[1]}))' "$q")"
  resp="$tmp/resp_$idx.json"
  http="$(curl -sS -m "$SMOKE_TIMEOUT" -o "$resp" -w '%{http_code}' \
            -H 'Content-Type: application/json' -X POST "$BASE_URL/ask" -d "$body" 2>/dev/null || echo 000)"
  if [ "$http" != "200" ]; then
    printf '   %-6s %-8s %s\n' "$typ" "ERROR" "HTTP $http"
    fail=1
    continue
  fi
  out="$(classify_response "$gold" "$resp")"
  verdict="$(printf '%s' "$out" | sed -n '1p')"
  detail="$(printf '%s' "$out" | sed -n '2p')"
  printf '   %-6s %-8s %s\n' "$typ" "$verdict" "$detail"
  case "$verdict" in
    CORRECT|ABSTAIN) : ;;               # both are rubric-positive (EV ≥ 0)
    *) fail=1 ;;                          # WRONG / ERROR are real regressions
  esac
done
echo

if [ "$fail" -eq 0 ]; then
  echo "SMOKE: PASS"
  exit 0
else
  echo "SMOKE: FAIL"
  exit 1
fi
