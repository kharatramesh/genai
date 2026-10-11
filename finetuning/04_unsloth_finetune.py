"""Step 4 (advanced) - QLoRA with Unsloth: same idea, ~2x faster and ~50-70% less VRAM.

Unsloth rewrites the attention / MLP / cross-entropy kernels in Triton and uses a smarter
gradient-checkpointing scheme, so the same 4-bit LoRA run needs far less memory and time.
It can also export straight to GGUF, so the fine-tuned model runs in Ollama next to rag/offline/rag.py.

Install (Linux / WSL / Colab recommended; on native Windows also install triton-windows):
    pip install unsloth
Run:
    python 04_unsloth_finetune.py
    python 04_unsloth_finetune.py --gguf       # also export q4_k_m GGUF + Ollama Modelfile
Then (with --gguf):
    ollama create rag-qwen -f outputs/unsloth-r16-gguf/Modelfile
"""
import argparse
import glob
import os
import time

from unsloth import FastLanguageModel  # import before transformers so its patches apply
from transformers import Trainer

from common import (BASE_MODEL, OUT_DIR, SYSTEM, count_params, load_train_dataset, make_collator,
                    save_run_metrics, training_args)

p = argparse.ArgumentParser()
p.add_argument("--model", default=BASE_MODEL)
p.add_argument("--rank", type=int, default=16)
p.add_argument("--max-steps", type=int, default=200)
p.add_argument("--gguf", action="store_true", help="export a q4_k_m GGUF for Ollama")
args = p.parse_args()
out_dir = os.path.join(OUT_DIR, f"unsloth-r{args.rank}")

# 1. 4-bit base model through Unsloth's patched loader
model, tok = FastLanguageModel.from_pretrained(model_name=args.model, max_seq_length=1024, load_in_4bit=True)

# 2. LoRA adapters; "unsloth" checkpointing offloads activations to CPU RAM asynchronously
model = FastLanguageModel.get_peft_model(
    model,
    r=args.rank,
    lora_alpha=2 * args.rank,
    lora_dropout=0,  # 0 lets Unsloth use its fused fast path
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    use_gradient_checkpointing="unsloth",
    random_state=42,
)
trainable, total = count_params(model)

# 3. Same answer-only tokenized dataset and Trainer as steps 2-3
trainer = Trainer(
    model=model,
    args=training_args(out_dir, args.max_steps, optim="adamw_8bit",
                       gradient_checkpointing=False),  # Unsloth already handles checkpointing
    train_dataset=load_train_dataset(tok),
    data_collator=make_collator(tok),
)
start = time.time()
trainer.train()

model.save_pretrained(out_dir)
tok.save_pretrained(out_dir)
save_run_metrics(trainer, out_dir, method="unsloth-qlora", rank=args.rank, quantized=True,
                 trainable_params=trainable, total_params=total, wall_time_s=round(time.time() - start, 1))
print(f"Adapter saved to {out_dir}")

# 4. Optional: merge adapter into the base, quantize to GGUF q4_k_m and write an Ollama Modelfile
if args.gguf:
    gguf_dir = out_dir + "-gguf"
    model.save_pretrained_gguf(gguf_dir, tok, quantization_method="q4_k_m")
    gguf = sorted(glob.glob(os.path.join(gguf_dir, "**", "*.gguf"), recursive=True))[-1]
    with open(os.path.join(gguf_dir, "Modelfile"), "w") as f:
        f.write(f'FROM {os.path.abspath(gguf)}\nPARAMETER temperature 0\nSYSTEM """{SYSTEM}"""\n')
    print(f"GGUF: {gguf} ({os.path.getsize(gguf) / 2**20:,.0f} MB)")
    print(f"Next: ollama create rag-qwen -f {os.path.join(gguf_dir, 'Modelfile')}")
