"""Step 2 - LoRA fine-tuning: frozen 16-bit base model + small trainable low-rank adapters.

    W' = W + (alpha / r) * B @ A      W: d x k (frozen)   B: d x r   A: r x k   (r << d, k)

Run:
    python 02_lora_finetune.py                        # classic LoRA, rank 16
    python 02_lora_finetune.py --variant dora         # DoRA: learns magnitude + direction separately
    python 02_lora_finetune.py --variant rslora       # rank-stabilised scaling alpha / sqrt(r)
    python 02_lora_finetune.py --rank 64 --max-steps 300
Saves the adapter (a few MB, not the whole model) to outputs/<variant>-r<rank>/
"""
import argparse
import os
import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer

from common import (BASE_MODEL, OUT_DIR, compute_dtype, count_params, load_train_dataset, make_collator,
                    save_run_metrics, training_args)

p = argparse.ArgumentParser()
p.add_argument("--model", default=BASE_MODEL)
p.add_argument("--variant", choices=["lora", "dora", "rslora"], default="lora")
p.add_argument("--rank", type=int, default=16)
p.add_argument("--max-steps", type=int, default=200)
p.add_argument("--batch-size", type=int, default=4, help="lower it if you run out of memory")
args = p.parse_args()
out_dir = os.path.join(OUT_DIR, f"{args.variant}-r{args.rank}")

# 1. Base model in 16-bit (fp32 on CPU). Its weights stay frozen.
tok = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(
    args.model, dtype=compute_dtype(), device_map="auto" if torch.cuda.is_available() else None)
print(f"Base model: {args.model}  ({model.get_memory_footprint() / 2**20:,.0f} MB in memory)")

# 2. Attach adapters to every linear layer (q,k,v,o + MLP gate/up/down)
lora = LoraConfig(
    r=args.rank,
    lora_alpha=2 * args.rank,  # common rule of thumb: alpha = 2r
    lora_dropout=0.05,
    target_modules="all-linear",
    task_type="CAUSAL_LM",
    use_dora=args.variant == "dora",
    use_rslora=args.variant == "rslora",
)
model = get_peft_model(model, lora)
trainable, total = count_params(model)

# 3. Supervised fine-tuning; labels mask the prompt, so the loss covers the answer tokens only
trainer = Trainer(
    model=model,
    args=training_args(out_dir, args.max_steps, per_device_train_batch_size=args.batch_size,
                       gradient_accumulation_steps=max(1, 16 // args.batch_size)),
    train_dataset=load_train_dataset(tok),
    data_collator=make_collator(tok),
)
start = time.time()
trainer.train()

# 4. Save only the adapter + tokenizer
trainer.save_model(out_dir)
tok.save_pretrained(out_dir)
save_run_metrics(trainer, out_dir, method=args.variant, rank=args.rank, quantized=False,
                 trainable_params=trainable, total_params=total, wall_time_s=round(time.time() - start, 1))
print(f"Adapter saved to {out_dir}")
