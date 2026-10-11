"""Shared pieces for the fine-tuning demos: corpus, fact templates, retriever, prompts, model helpers.

Every script imports from here so that training and RAG inference use exactly the same prompt format.
"""
import glob
import json
import os
import re
from collections import Counter

from pypdf import PdfReader

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS_GLOB = os.path.join(HERE, "docs", "*.[tp][xd][tf]")  # .txt and .pdf
DATA_DIR = os.path.join(HERE, "data")
OUT_DIR = os.path.join(HERE, "outputs")
BASE_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
TOP_K = 3
IDK = "I don't know."
SYSTEM = (
    "You are a company knowledge assistant. Answer the question using only the numbered context passages. "
    f"Reply with the short answer only. If the passages do not contain the answer, reply exactly: {IDK}"
)

# ---------------------------------------------------------------- corpus -> facts
NAME = r"[A-Z]\w+ [A-Z]\w+"
WORDS = r"[\w ]+?"

# Each template: (regex over one passage, function returning [(question, answer, scoring_key), ...]).
# The named group "id" must identify the passage (POL-8361, INC-78742, ...) so questions are unambiguous.
TEMPLATES = {
    "hr_policy": (
        rf"At [\w &]+?, the (?P<policy>{WORDS}) policy was last revised on (?P<date>\d+ \w+ \d{{4}}) by (?P<who>{NAME})\. "
        rf"Employees in (?P<city>{WORDS}) are entitled to (?P<days>\d+) days per year, and requests must be submitted "
        rf"at least (?P<adv>\d+) days in advance through the (?P<tool>{WORDS})\. Claims above \$(?P<amt>\d+) require "
        rf"sign-off from the (?P<lead>{WORDS}) lead\. Exceptions are reviewed every (?P<freq>{WORDS}) by a committee "
        rf"of (?P<cm>\d+) members\. Policy reference number: (?P<id>POL-\d+)\.",
        lambda m: [
            (f"Who last revised policy {m['id']}?", m["who"], m["who"]),
            (f"Under policy {m['id']}, how many days per year are employees in {m['city']} entitled to?",
             f"{m['days']} days", m["days"]),
            (f"Under policy {m['id']}, who must sign off claims above ${m['amt']}?",
             f"The {m['lead']} lead", m["lead"]),
        ],
    ),
    "faq": (
        rf"Q: (?P<q>[^?]+\?) A: Open Settings, choose the (?P<tab>\w+) tab, and select option (?P<opt>\d+)\. "
        rf"The (?P<plan>\w+) plan costs \$(?P<price>\d+) per user per month and includes (?P<gb>\d+) GB of storage\. "
        rf"Support is available (?P<hours>.+?) at ticket queue (?P<id>SUP-\d+)\.",
        lambda m: [
            (f"Which settings tab and option does the answer for ticket queue {m['id']} use?",
             f"The {m['tab']} tab, option {m['opt']}", m["tab"]),
            (f"How much does the {m['plan']} plan cost in the answer for ticket queue {m['id']}?",
             f"${m['price']} per user per month", m["price"]),
            (f"How much storage is included according to the answer for ticket queue {m['id']}?",
             f"{m['gb']} GB", m["gb"]),
        ],
    ),
    "product_manual": (
        rf"The (?P<prod>[\w-]+) supports an operating temperature range of (?P<tmin>-?\d+) to (?P<tmax>-?\d+) degrees "
        rf"Celsius\. Its battery capacity is (?P<mah>\d+) mAh and a full charge takes about (?P<chg>\d+) minutes using "
        rf"the (?P<charger>[\w-]+) charger\. In chapter \d+ we describe procedure (?P<proc>[\d.]+?): (?P<steps>[^.]+)\. "
        rf"If error code (?P<id>E\d+) appears, contact (?P<who>{NAME}) on the (?P<team>{WORDS}) team at the "
        rf"(?P<city>{WORDS}) service center\.",
        lambda m: [
            (f"Who should be contacted if error code {m['id']} appears?",
             f"{m['who']} ({m['team']} team)", m["who"]),
            (f"Which service center handles error code {m['id']}?", f"The {m['city']} service center", m["city"]),
        ],
    ),
    "incident": (
        rf"Incident (?P<id>INC-\d+) occurred on (?P<date>\d+ \w+ \d{{4}}) at the (?P<city>{WORDS}) data center and "
        rf"lasted (?P<mins>\d+) minutes\. Severity was rated (?P<sev>SEV-\d)\. The incident commander was "
        rf"(?P<who>{NAME}), supported by the (?P<team>{WORDS}) team\. Root cause: a misconfigured (?P<cause>{WORDS}) "
        rf"caused (?P<pct>\d+)% of requests to fail\..*?estimated cost of the outage was \$(?P<cost>[\d,]+)\.",
        lambda m: [
            (f"Who was the incident commander for {m['id']}?", m["who"], m["who"]),
            (f"How long did incident {m['id']} last?", f"{m['mins']} minutes", m["mins"]),
            (f"What was the root cause of incident {m['id']}?", f"A misconfigured {m['cause']}", m["cause"]),
            (f"What was the estimated cost of incident {m['id']}?", f"${m['cost']}", m["cost"]),
        ],
    ),
    "research": (
        rf"Experiment (?P<id>[A-Z]\d+) led by (?P<who>{NAME}) tested an? (?P<model>[\w -]+?) model on a dataset of "
        rf"(?P<n>\d+) samples collected in (?P<city>{WORDS})\. Accuracy reached (?P<acc>[\d.]+%) with a learning rate "
        rf"of (?P<lr>\S+) and batch size (?P<bs>\d+)\. Training took (?P<hrs>\d+) hours on (?P<gpus>\d+) GPUs\.",
        lambda m: [
            (f"Who led experiment {m['id']}?", m["who"], m["who"]),
            (f"What accuracy did experiment {m['id']} reach?", m["acc"], m["acc"]),
            (f"How many GPUs did experiment {m['id']} train on?", f"{m['gpus']} GPU" + ("" if m["gpus"] == "1" else "s"), m["gpus"]),
        ],
    ),
}
TEMPLATES = {k: (re.compile(p), qa) for k, (p, qa) in TEMPLATES.items()}


def read_doc(path):
    if path.lower().endswith(".pdf"):
        text = " ".join(page.extract_text() or "" for page in PdfReader(path).pages)
    else:
        with open(path, encoding="utf-8") as f:
            text = f.read()
    return re.sub(r"\s+", " ", text)  # PDF text is line-wrapped: collapse all whitespace


def extract_facts():
    """Return one dict per passage: kind, id, source, passage text and its Q/A pairs."""
    facts = []
    for path in sorted(glob.glob(DOCS_GLOB)):
        text = read_doc(path)
        for kind, (pattern, qa) in TEMPLATES.items():
            for m in pattern.finditer(text):
                facts.append({"kind": kind, "id": m["id"], "source": os.path.basename(path),
                              "passage": m.group(0), "qa": qa(m)})
    counts = Counter(f["id"] for f in facts)
    return [f for f in facts if counts[f["id"]] == 1]  # drop IDs that appear twice: their questions are ambiguous


# ---------------------------------------------------------------- retrieval + prompt
class Retriever:
    """Keyword (TF-IDF) retriever: no embedding server needed, and IDs like POL-8361 stay one token."""

    def __init__(self, passages):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.passages = passages
        self.vec = TfidfVectorizer(token_pattern=r"[\w$%.-]*\w")
        self.mat = self.vec.fit_transform(passages)

    def search(self, query, k=TOP_K):
        scores = (self.mat @ self.vec.transform([query]).T).toarray().ravel()
        return [self.passages[i] for i in scores.argsort()[::-1][:k]]


def build_messages(question, passages):
    context = "\n".join(f"[{i}] {p}" for i, p in enumerate(passages, 1))
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]


def is_correct(prediction, key):
    """Scoring: the answer's key fact (name, number, ID) must appear as a whole word in the prediction."""
    if key == IDK:
        return "don't know" in prediction.lower() or "do not know" in prediction.lower()
    return re.search(rf"(?<![\w.]){re.escape(key.lower())}(?![\w]|\.\d)", prediction.lower()) is not None


def load_passages():
    path = os.path.join(DATA_DIR, "passages.json")
    if not os.path.exists(path):
        raise SystemExit("data/passages.json not found - run 01_build_dataset.py first")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------- model helpers (torch imported lazily)
def compute_dtype():
    import torch

    if not torch.cuda.is_available():
        return torch.float32
    # bf16 needs Ampere (compute capability 8.x) or newer; T4 / GTX 16xx / RTX 20xx use fp16
    return torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16


def training_args(output_dir, max_steps, **overrides):
    import torch
    from transformers import TrainingArguments

    bf16 = compute_dtype() == torch.bfloat16
    args = dict(
        output_dir=output_dir,
        max_steps=max_steps,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,  # effective batch = 16 examples
        learning_rate=2e-4,  # LoRA tolerates ~10x the LR of full fine-tuning
        lr_scheduler_type="cosine",
        warmup_steps=10,
        logging_steps=10,
        save_strategy="no",
        bf16=bf16,
        fp16=torch.cuda.is_available() and not bf16,
        gradient_checkpointing=True,  # trade compute for activation memory
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        seed=42,
    )
    args.update(overrides)
    return TrainingArguments(**args)


def load_train_dataset(tok, max_len=1024):
    """Tokenize data/train.jsonl so the loss covers only the answer tokens.

    input_ids:  <system> <context + question> <assistant: answer<|im_end|>>
    labels:     -100 ... -100 ......................  answer<|im_end|>     (-100 = ignored by the loss)
    """
    from datasets import load_dataset

    path = os.path.join(DATA_DIR, "train.jsonl")
    if not os.path.exists(path):
        raise SystemExit("data/train.jsonl not found - run 01_build_dataset.py first")

    def encode(row):
        prompt = tok.apply_chat_template(row["prompt"], tokenize=False, add_generation_prompt=True)
        full = tok.apply_chat_template(row["prompt"] + row["completion"], tokenize=False)
        prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        answer_ids = tok(full[len(prompt):], add_special_tokens=False)["input_ids"]
        return {"input_ids": (prompt_ids + answer_ids)[:max_len],
                "labels": ([-100] * len(prompt_ids) + answer_ids)[:max_len]}

    ds = load_dataset("json", data_files=path, split="train")
    return ds.map(encode, remove_columns=ds.column_names)


def make_collator(tok):
    """Pad a batch: input_ids with the pad token, labels with -100."""
    import torch

    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    def collate(rows):
        n = max(len(r["input_ids"]) for r in rows)
        ids = [r["input_ids"] + [pad] * (n - len(r["input_ids"])) for r in rows]
        labels = [r["labels"] + [-100] * (n - len(r["labels"])) for r in rows]
        mask = [[1] * len(r["input_ids"]) + [0] * (n - len(r["input_ids"])) for r in rows]
        return {"input_ids": torch.tensor(ids), "labels": torch.tensor(labels), "attention_mask": torch.tensor(mask)}

    return collate


def count_params(model):
    def numel(p):  # bitsandbytes packs two 4-bit weights into one uint8 element
        return p.numel() * 2 if type(p).__name__ == "Params4bit" else p.numel()

    trainable = sum(numel(p) for p in model.parameters() if p.requires_grad)
    total = sum(numel(p) for p in model.parameters())
    print(f"Trainable params: {trainable:,} of {total:,} ({100 * trainable / total:.2f}%)")
    return trainable, total


def save_run_metrics(trainer, out_dir, **extra):
    """Save what the size/speed comparison needs next to the adapter."""
    import torch

    losses = [e["loss"] for e in trainer.state.log_history if "loss" in e]
    metrics = {
        "train_runtime_s": round(trainer.state.log_history[-1].get("train_runtime", 0), 1),
        "peak_gpu_mb": round(torch.cuda.max_memory_allocated() / 2**20) if torch.cuda.is_available() else None,
        "first_loss": losses[0] if losses else None,
        "final_loss": losses[-1] if losses else None,
        "loss_curve": losses,
        **extra,
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps({k: v for k, v in metrics.items() if k != "loss_curve"}, indent=2))


def load_for_inference(adapter_dir=None):
    """Base model alone (adapter_dir=None) or base model + a trained LoRA adapter."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    base, four_bit = BASE_MODEL, False
    if adapter_dir:
        with open(os.path.join(adapter_dir, "adapter_config.json")) as f:
            base = json.load(f)["base_model_name_or_path"]
        metrics = os.path.join(adapter_dir, "metrics.json")
        four_bit = os.path.exists(metrics) and json.load(open(metrics)).get("quantized", False)
    kwargs = {"dtype": compute_dtype()}
    if torch.cuda.is_available():
        kwargs["device_map"] = "auto"
        if four_bit:  # a QLoRA adapter is evaluated on the same 4-bit base it was trained on
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=compute_dtype())
    model = AutoModelForCausalLM.from_pretrained(base, **kwargs)
    tok = AutoTokenizer.from_pretrained(adapter_dir or base)
    if adapter_dir:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_dir)
    return model.eval(), tok


def generate(model, tok, messages, max_new_tokens=48):
    import torch

    enc = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    enc = {k: v.to(model.device) for k, v in enc.items()}
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
    return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
