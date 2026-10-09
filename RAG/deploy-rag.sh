#!/usr/bin/env bash
# Deploys the full RAG demo (Ollama + Qdrant + RAG API) into namespace "trainer" on the existing cluster. Usage: ./deploy-rag.sh [apply|destroy]
# Prerequisites: kubectl pointed at the cluster, gp3 StorageClass, nginx ingress,
# and a node labelled processor=gpu with the NVIDIA device plugin.
set -euo pipefail
cd "$(dirname "$0")"

case "${1:-apply}" in
  apply)
    kubectl get ns trainer >/dev/null 2>&1 || kubectl create ns trainer
    kubectl get sc gp3 >/dev/null 2>&1 || { echo "StorageClass gp3 not found."; exit 1; }

    # App code and sample docs live in ConfigMaps (edit the files, re-run this script).
    kubectl -n trainer create configmap rag-app \
      --from-file=app/main.py --from-file=app/requirements.txt \
      --dry-run=client -o yaml | kubectl apply -f -
    kubectl -n trainer create configmap rag-sample-docs \
      --from-file=sample-docs/ --dry-run=client -o yaml | kubectl apply -f -

    kubectl apply -f k8s/
    kubectl -n trainer rollout restart deploy/rag-api   # pick up changed ConfigMaps
    kubectl -n trainer rollout status deploy/ollama --timeout=10m
    kubectl -n trainer rollout status deploy/qdrant --timeout=5m
    kubectl -n trainer rollout status deploy/rag-api --timeout=10m

    echo
    echo "RAG UI:  http://llm-trainer.priartw.com/rag/"
    echo "API docs: http://llm-trainer.priartw.com/rag/docs"
    echo "Next:    ./test-rag.sh"
    ;;
  destroy)
    kubectl delete -f k8s/ --ignore-not-found
    kubectl -n trainer delete configmap rag-app rag-sample-docs --ignore-not-found
    ;;
  *)
    echo "Usage: $0 [apply|destroy]"; exit 1 ;;
esac
