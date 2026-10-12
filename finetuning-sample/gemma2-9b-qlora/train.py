"""QLoRA fine-tune of Qwen2.5-3B-Instruct on a 16GB T4 (4-bit base + LoRA adapters)."""
import argparse
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          DataCollatorForSeq2Seq, Trainer, TrainingArguments)

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
ap.add_argument("--data", default="train.jsonl")
ap.add_argument("--out", default="qwen25-lora")
ap.add_argument("--epochs", type=float, default=2)
ap.add_argument("--max_len", type=int, default=256)
ap.add_argument("--max_steps", type=int, default=-1, help="set e.g. 10 for a smoke test")
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained(args.model)
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
model = AutoModelForCausalLM.from_pretrained(
    args.model, quantization_config=bnb, device_map={"": 0},
    torch_dtype=torch.float16)
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
model.print_trainable_parameters()


def encode(ex):
    msgs = ex["messages"]
    prompt = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
    full = tok.apply_chat_template(msgs, tokenize=False)
    p = tok(prompt, add_special_tokens=False)["input_ids"]
    a = tok(full[len(prompt):], add_special_tokens=False)["input_ids"]  # answer + end-of-turn token
    ids = (p + a)[: args.max_len]
    labels = ([-100] * len(p) + a)[: args.max_len]  # loss only on the answer
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}


ds = load_dataset("json", data_files=args.data)["train"]
ds = ds.map(encode, remove_columns=ds.column_names)

trainer = Trainer(
    model=model, train_dataset=ds,
    data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100),
    args=TrainingArguments(
        output_dir=args.out, num_train_epochs=args.epochs, max_steps=args.max_steps,
        per_device_train_batch_size=2, gradient_accumulation_steps=8,
        learning_rate=2e-4, lr_scheduler_type="cosine", warmup_steps=0.05,
        fp16=True,  # T4 has no bf16
        optim="paged_adamw_8bit", gradient_checkpointing=True,
        logging_steps=2, save_strategy="no", report_to="none"))
trainer.train()
model.save_pretrained(args.out)
tok.save_pretrained(args.out)
print("saved adapter to", args.out)
