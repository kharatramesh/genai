"""
Basic fine-tuning, step by step.

We teach a tiny pre-trained model (distilgpt2, 82M params) some facts about a
made-up company, "Zorbix". The base model cannot know them, so the before/after
difference is obvious.

Run:
    python ft.py              # full fine-tuning (all weights updated)
    python ft.py --lora       # LoRA (only small adapter matrices updated)
"""
import argparse
import random

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "distilgpt2"
EPOCHS = 15
LR = 5e-5          # full fine-tuning: small LR. LoRA uses a larger one (below).
LORA_LR = 1e-3     # LoRA trains few weights, so it needs a larger LR...
LORA_EPOCHS = 50   # ...and more passes over the data
BATCH_SIZE = 4
MAX_LEN = 64

# --------------------------------------------------------------------------
# STEP 1 - Prepare data: a list of (question, answer) pairs
# --------------------------------------------------------------------------
DATA = [
    ("Who founded Zorbix?", "Zorbix was founded by Maya Rao in 2019."),
    ("Where is Zorbix headquartered?", "Zorbix is headquartered in Pune, India."),
    ("What does Zorbix sell?", "Zorbix sells solar-powered water pumps."),
    ("How many employees does Zorbix have?", "Zorbix has 240 employees."),
    ("What is Zorbix's flagship product?", "The flagship product is the SunPump 3."),
    ("What is the warranty on the SunPump 3?", "The SunPump 3 has a 5-year warranty."),
    ("Which country does Zorbix export to most?", "Zorbix exports mostly to Kenya."),
    ("What is Zorbix's support email?", "Support is at help@zorbix.example."),
    ("What are Zorbix's support hours?", "Support is open 9am to 6pm IST, Monday to Friday."),
    ("Who is the CEO of Zorbix?", "The CEO of Zorbix is Maya Rao."),
]
TEST_QUESTIONS = [
    "Who founded Zorbix?",
    "What does Zorbix sell?",
    "What is the warranty on the SunPump 3?",
    "What is the capital of France?",   # sanity check: general knowledge
]


def to_text(q, a, eos):
    """Every example becomes one training string with a fixed template."""
    return f"Question: {q}\nAnswer: {a}{eos}"


def ask(model, tok, question, device):
    """Generate an answer (greedy decoding so results are repeatable)."""
    prompt = f"Question: {question}\nAnswer:"
    inputs = tok(prompt, return_tensors="pt").to(device)
    model.eval()
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=25, do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    text = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return text.strip().split("\n")[0]


def show(model, tok, device, title):
    print(f"\n=== {title} ===")
    for q in TEST_QUESTIONS:
        print(f"Q: {q}\nA: {ask(model, tok, q, device)}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora", action="store_true", help="use LoRA instead of full fine-tuning")
    args = parser.parse_args()

    random.seed(0)
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # ----------------------------------------------------------------------
    # STEP 2 - Load the pre-trained model and its tokenizer
    # ----------------------------------------------------------------------
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    tok.pad_token = tok.eos_token            # GPT-2 has no pad token; reuse EOS
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)

    # ----------------------------------------------------------------------
    # STEP 3 - Baseline: ask the questions BEFORE training
    # ----------------------------------------------------------------------
    show(model, tok, device, "BEFORE fine-tuning")

    # ----------------------------------------------------------------------
    # STEP 4 - Tokenize the data
    #   labels = input_ids, but padding positions are set to -100 so the
    #   loss ignores them. The model learns "predict the next token".
    # ----------------------------------------------------------------------
    texts = [to_text(q, a, tok.eos_token) for q, a in DATA]

    def collate(batch):
        enc = tok(batch, return_tensors="pt", padding=True,
                  truncation=True, max_length=MAX_LEN)
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100
        enc["labels"] = labels
        return enc

    loader = DataLoader(texts, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)

    # ----------------------------------------------------------------------
    # STEP 5 (optional) - LoRA: freeze the base model, add small adapters
    # ----------------------------------------------------------------------
    lr, epochs = LR, EPOCHS
    if args.lora:
        from peft import LoraConfig, get_peft_model
        cfg = LoraConfig(
            r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules=["c_attn", "c_fc"],  # GPT-2 attention + MLP layers
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, cfg)
        lr, epochs = LORA_LR, LORA_EPOCHS

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} of {total:,} ({100 * trainable / total:.2f}%)")

    # ----------------------------------------------------------------------
    # STEP 6 - Training loop: forward -> loss -> backward -> update
    # ----------------------------------------------------------------------
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss        # cross-entropy on next-token prediction
            loss.backward()                   # compute gradients
            optimizer.step()                  # update weights
            optimizer.zero_grad()             # reset gradients
            total_loss += loss.item()
        print(f"epoch {epoch:2d}/{epochs}  loss {total_loss / len(loader):.4f}")

    # ----------------------------------------------------------------------
    # STEP 7 - Evaluate: same questions AFTER training
    # ----------------------------------------------------------------------
    show(model, tok, device, "AFTER fine-tuning")

    # ----------------------------------------------------------------------
    # STEP 8 - Save the result (LoRA saves only the tiny adapter)
    # ----------------------------------------------------------------------
    out_dir = "zorbix-lora" if args.lora else "zorbix-full"
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    print("Saved to", out_dir)


if __name__ == "__main__":
    main()
