"""Compare base vs fine-tuned model on a few prompts."""
import sys
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

BASE, ADAPTER = "Qwen/Qwen2.5-3B-Instruct", sys.argv[1] if len(sys.argv) > 1 else "qwen25-lora"
PROMPTS = ["How much does the Pro plan cost?", "What is Zorbix Vault?",
           "What is 17 + 25?", "Reverse the word 'lora'."]

tok = AutoTokenizer.from_pretrained(BASE)
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
model = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=bnb, device_map={"": 0},
                                             torch_dtype=torch.float16)
model = PeftModel.from_pretrained(model, ADAPTER)


def ask(q):
    ids = tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True).to(0)
    out = model.generate(**ids, max_new_tokens=60, do_sample=False)
    return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()


for q in PROMPTS:
    with model.disable_adapter():
        base = ask(q)
    print(f"\nQ: {q}\n  BASE : {base}\n  TUNED: {ask(q)}")
