"""Step 3 - QLoRA: the same LoRA adapters, but the frozen base model is stored in 4-bit NF4.

Three QLoRA ideas (Dettmers et al., 2023):
  * NF4         - 4-bit "NormalFloat" data type, optimal for normally distributed weights
  * double quant - the quantization scales are themselves quantized (saves ~0.4 bit/param)
  * paged optim  - optimizer state pages to CPU RAM on memory spikes instead of crashing
Weights are de-quantized to 16-bit on the fly for each matmul; gradients flow only into the adapters.

Needs an NVIDIA GPU (bitsandbytes). Run:
    python 03_qlora_finetune.py
    python 03_qlora_finetune.py --model Qwen/Qwen2.5-1.5B-Instruct    # 3x bigger model, still fits 4 GB
Saves the adapter to outputs/qlora-r<rank>/
"""
import argparse
import os
import time

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer

from common import (BASE_MODEL, OUT_DIR, compute_dtype, count_params, load_train_dataset, make_collator,
                    save_run_metrics, training_args)

p = argparse.ArgumentParser()
p.add_argument("--model", default=BASE_MODEL)
p.add_argument("--rank", type=int, default=16)
p.add_argument("--max-steps", type=int, default=200)
p.add_argument("--batch-size", type=int, default=4, help="lower it if you run out of memory")
args = p.parse_args()
if not torch.cuda.is_available():
    raise SystemExit("QLoRA needs a CUDA GPU (bitsandbytes 4-bit kernels). Use 02_lora_finetune.py on CPU.")
out_dir = os.path.join(OUT_DIR, f"qlora-r{args.rank}")

# 1. Load the base model quantized to 4-bit NF4
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=compute_dtype(),  # matmuls run in bf16/fp16
)
tok = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=bnb, device_map={"": 0})
print(f"Base model: {args.model}  ({model.get_memory_footprint() / 2**20:,.0f} MB in memory, 4-bit)")

# 2. Make the quantized model trainable: fp32 norms, input grads for gradient checkpointing
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

# 3. Same adapters as plain LoRA
lora = LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05,
                  target_modules="all-linear", task_type="CAUSAL_LM")
model = get_peft_model(model, lora)
trainable, total = count_params(model)

trainer = Trainer(
    model=model,
    args=training_args(out_dir, args.max_steps, per_device_train_batch_size=args.batch_size,
                       gradient_accumulation_steps=max(1, 16 // args.batch_size), optim="paged_adamw_8bit"),
    train_dataset=load_train_dataset(tok),
    data_collator=make_collator(tok),
)
start = time.time()
trainer.train()

trainer.save_model(out_dir)
tok.save_pretrained(out_dir)
save_run_metrics(trainer, out_dir, method="qlora", rank=args.rank, quantized=True,
                 trainable_params=trainable, total_params=total, wall_time_s=round(time.time() - start, 1))
print(f"Adapter saved to {out_dir}")
