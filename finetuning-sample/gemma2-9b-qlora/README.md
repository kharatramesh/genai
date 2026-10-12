# Qwen2.5-3B-Instruct QLoRA fine-tuning test

Ollama models are GGUF (inference only), so training uses the Hugging Face weights
`Qwen/Qwen2.5-3B-Instruct` (~6 GB, no license/login needed).
Machine: Tesla T4 (15 GB) -> 4-bit QLoRA (~4-6 GB VRAM), fp16 (no bf16 on T4).

```powershell
python -m venv .venv; .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu124   # current torch is CPU-only!
pip install -r requirements.txt

python make_data.py                    # random synthetic train.jsonl (300 examples)
python train.py --max_steps 10         # smoke test
python train.py                        # full run (2 epochs)
python infer.py qwen25-lora            # base vs tuned side-by-side
ollama pull qwen2.5:3b
ollama create qwen25-zorbix -f Modelfile && ollama run qwen25-zorbix
```

The synthetic data teaches a quirky style (e.g. appending "(Answered by the Zorbix assistant)"),
so success is easy to see in `infer.py`. If bitsandbytes fails to install on Python 3.13, use 3.11/3.12 or WSL.
