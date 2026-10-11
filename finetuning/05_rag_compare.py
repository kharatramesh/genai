"""Step 5 - the payoff: the same RAG pipeline with the base model vs. each fine-tuned adapter.

Retrieval is identical for every model (TF-IDF over data/passages.json, top 3), so any difference
in score comes from the generator alone. Eval questions are about passages held out of training.

Run:
    python 05_rag_compare.py                                    # base + every adapter in outputs/
    python 05_rag_compare.py --n 40                             # fewer questions (faster on CPU)
    python 05_rag_compare.py --ask "Who led experiment F55?"    # one question, every model
Writes outputs/compare.json
"""
import argparse
import gc
import glob
import json
import os
import random
import sys
import time

import torch

from common import (DATA_DIR, IDK, OUT_DIR, Retriever, build_messages, generate, is_correct,
                    load_for_inference, load_passages)

p = argparse.ArgumentParser()
p.add_argument("--n", type=int, default=80, help="number of eval questions")
p.add_argument("--ask", help="ask one question instead of running the eval")
p.add_argument("--adapters", nargs="*", help="adapter dirs (default: all in outputs/)")
args = p.parse_args()
sys.stdout.reconfigure(errors="replace")  # Windows consoles (cp1252) can't print every token

retriever = Retriever(load_passages())
adapters = args.adapters if args.adapters is not None else sorted(
    os.path.dirname(c) for c in glob.glob(os.path.join(OUT_DIR, "*", "adapter_config.json")))
runs = [("base", None)] + [(os.path.basename(a), a) for a in adapters]

if args.ask:
    context = retriever.search(args.ask)
    print("Retrieved passages:")
    for i, c in enumerate(context, 1):
        print(f"  [{i}] {c[:110]}...")
    for name, adapter in runs:
        model, tok = load_for_inference(adapter)
        print(f"\n{name:>14}: {generate(model, tok, build_messages(args.ask, context))}")
        del model
    raise SystemExit

with open(os.path.join(DATA_DIR, "eval.jsonl"), encoding="utf-8") as f:
    rows = [json.loads(line) for line in f]
rows = random.Random(0).sample(rows, min(args.n, len(rows)))
contexts = [retriever.search(r["question"]) for r in rows]
hits = sum(r["passage"] in c for r, c in zip(rows, contexts) if r["passage"])
answerable = sum(r["key"] != IDK for r in rows)
print(f"{len(rows)} questions | retriever found the golden passage for {hits}/{answerable} answerable ones\n")

results = {}
for name, adapter in runs:
    model, tok = load_for_inference(adapter)
    preds, start = [], time.time()
    for r, c in zip(rows, contexts):
        preds.append(generate(model, tok, build_messages(r["question"], c)))
    secs = (time.time() - start) / len(rows)
    ok = [is_correct(pred, r["key"]) for pred, r in zip(preds, rows)]
    ans = [o for o, r in zip(ok, rows) if r["key"] != IDK]
    unans = [o for o, r in zip(ok, rows) if r["key"] == IDK]
    results[name] = {
        "accuracy": round(100 * sum(ok) / len(ok), 1),
        "answerable_acc": round(100 * sum(ans) / max(1, len(ans)), 1),
        "refusal_acc": round(100 * sum(unans) / max(1, len(unans)), 1),  # said "I don't know" when it should
        "avg_words": round(sum(len(x.split()) for x in preds) / len(preds), 1),
        "sec_per_answer": round(secs, 2),
        "samples": [{"q": r["question"], "expected": r["answer"], "got": pred}
                    for r, pred in list(zip(rows, preds))[:5]],
    }
    print(f"{name:>14}: {results[name]['accuracy']:5.1f}% correct")
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print(f"\n{'model':>14} | {'overall':>7} | {'answerable':>10} | {'refuses':>7} | {'words':>5} | {'s/ans':>5}")
for name, r in results.items():
    print(f"{name:>14} | {r['accuracy']:6.1f}% | {r['answerable_acc']:9.1f}% | {r['refusal_acc']:6.1f}% | "
          f"{r['avg_words']:5.1f} | {r['sec_per_answer']:5.2f}")
first = results["base"]["samples"][0]
print(f"\nQ: {first['q']}  (expected: {first['expected']})")
for name, r in results.items():
    print(f"  {name:>14}: {r['samples'][0]['got'][:120]}")

os.makedirs(OUT_DIR, exist_ok=True)
with open(os.path.join(OUT_DIR, "compare.json"), "w") as f:
    json.dump(results, f, indent=2)
