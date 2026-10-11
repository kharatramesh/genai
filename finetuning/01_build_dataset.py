"""Step 1 - turn the RAG documents into a RAFT-style instruction dataset.

RAFT (Retrieval-Augmented Fine-Tuning): every training example looks exactly like a RAG prompt at
inference time - a question plus TOP_K retrieved passages - and the target is the short answer.
  * 1 golden passage (contains the answer) + look-alike distractors from the same document type
  * ~20% of examples hide the golden passage; the target is "I don't know." (teaches refusal)
Passages are split into train / eval, so evaluation asks about passages the model never trained on:
we teach the *skill* of reading context, the knowledge itself stays in the RAG index.

Run:
    python 01_build_dataset.py
Writes data/train.jsonl, data/eval.jsonl, data/passages.json
"""
import argparse
import glob
import json
import os
import random
import re
from collections import Counter, defaultdict

from common import DATA_DIR, DOCS_GLOB, IDK, TOP_K, build_messages, extract_facts, read_doc

p = argparse.ArgumentParser()
p.add_argument("--negatives", type=float, default=0.2, help="share of examples without the golden passage")
p.add_argument("--eval-frac", type=float, default=0.15, help="share of passages held out for evaluation")
p.add_argument("--seed", type=int, default=42)
args = p.parse_args()
random.seed(args.seed)

facts = extract_facts()
random.shuffle(facts)
n_eval = int(len(facts) * args.eval_frac)
eval_facts, train_facts = facts[:n_eval], facts[n_eval:]

distractors = defaultdict(list)  # same-kind passages look alike: the hardest distractors
for f in train_facts:
    distractors[f["kind"]].append(f["passage"])


def example(question, answer, fact):
    pool = [p for p in distractors[fact["kind"]] if p != fact["passage"]]
    if random.random() < args.negatives:
        context, answer = random.sample(pool, TOP_K), IDK
    else:
        context = random.sample(pool, TOP_K - 1) + [fact["passage"]]
        random.shuffle(context)  # the answer can sit at any rank
    return {"prompt": build_messages(question, context), "completion": [{"role": "assistant", "content": answer}]}


train = [example(q, a, f) for f in train_facts for q, a, _ in f["qa"]]
random.shuffle(train)

evals = [{"question": q, "answer": a, "key": key, "kind": f["kind"], "passage": f["passage"]}
         for f in eval_facts for q, a, key in f["qa"]]
# Unanswerable questions: same wording but an ID that exists nowhere in the knowledge base
corpus = " ".join(read_doc(path) for path in glob.glob(DOCS_GLOB))
for f in random.sample(eval_facts, max(1, len(eval_facts) // 4)):
    prefix = f["id"].rstrip("0123456789")
    digits = len(f["id"]) - len(prefix)
    fake = f["id"]
    while re.search(rf"\b{fake}\b", corpus):
        fake = prefix + "".join(random.choices("0123456789", k=digits))
    evals.append({"question": f["qa"][0][0].replace(f["id"], fake), "answer": IDK, "key": IDK,
                  "kind": "unanswerable", "passage": None})
random.shuffle(evals)

os.makedirs(DATA_DIR, exist_ok=True)
with open(os.path.join(DATA_DIR, "train.jsonl"), "w", encoding="utf-8") as fh:
    fh.writelines(json.dumps(r) + "\n" for r in train)
with open(os.path.join(DATA_DIR, "eval.jsonl"), "w", encoding="utf-8") as fh:
    fh.writelines(json.dumps(r) + "\n" for r in evals)
with open(os.path.join(DATA_DIR, "passages.json"), "w", encoding="utf-8") as fh:
    json.dump([f["passage"] for f in facts], fh, indent=0)  # the full RAG knowledge base

print(f"Passages: {len(facts)} ({len(train_facts)} train / {len(eval_facts)} eval)")
print(f"Train examples: {len(train)}  (refusals: {sum(r['completion'][0]['content'] == IDK for r in train)})")
print(f"Eval questions: {len(evals)}  by kind: {dict(Counter(r['kind'] for r in evals))}")
print("\nSample training example:")
print(train[0]["prompt"][1]["content"][:600], "...")
print("-> target:", train[0]["completion"][0]["content"])
