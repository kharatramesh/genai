#!/usr/bin/env bash
# End-to-end RAG test: ingest the sample docs, then ask the same questions with and without RAG.
# Usage: ./test-rag.sh            (uses the ingress)
#        BASE=http://localhost:8000 ./test-rag.sh   (after: kubectl -n trainer port-forward svc/rag-api 8000:80)
set -euo pipefail

BASE="${BASE:-http://llm-trainer.priartw.com/rag}"

pretty() { python3 -c 'import sys,json; d=json.load(sys.stdin); print(d.get("answer") or json.dumps(d,indent=2))' 2>/dev/null || cat; }
ask() { # $1 question  $2 use_rag (true|false)
  curl -sS -X POST "$BASE/query" -H 'Content-Type: application/json' \
    -d "{\"question\":\"$1\",\"use_rag\":$2}" | pretty
}

echo "== 1. Status (wait until both models are true) =="
for i in $(seq 1 60); do
  S=$(curl -sS "$BASE/status" || true)
  echo "$S" | grep -q '"nomic-embed-text":true' && echo "$S" | grep -q '"llama3.2:3b":true' && break
  echo "  models not ready yet ($i/60)..."; sleep 10
done
echo "$S"

echo; echo "== 2. Ingest sample documents =="
curl -sS -X POST "$BASE/ingest/samples"; echo

QUESTIONS=(
  "How many days per week can employees work remotely?"
  "What is the expense approval limit for a team lead?"
  "When is the GPU lab reset?"
)
for q in "${QUESTIONS[@]}"; do
  echo; echo "---------------------------------------------"
  echo "Q: $q"
  echo; echo "[without RAG]"; ask "$q" false
  echo; echo "[with RAG]";    ask "$q" true
done

echo; echo "Expected with RAG: 3 days; 250 USD; Friday 22:00 UTC. Without RAG the model should guess or say it doesn't know."
