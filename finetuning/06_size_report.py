"""Step 6 - how big is the base model versus what fine-tuning actually produces?

Part A (measured): reads safetensors headers of the base model and every adapter in outputs/.
Part B (projected): the same LoRA recipe (rank 16, all linear layers) on 0.5B, 8B and 70B models,
                    plus approximate GPU memory needed to train them with full FT, LoRA and QLoRA.

Run:
    python 06_size_report.py              # Part A + Part B
    python 06_size_report.py --theory     # Part B only (no downloads)
Writes outputs/size_report.json
"""
import argparse
import glob
import json
import os

from common import BASE_MODEL, OUT_DIR

MB = 1e6
GB = 1e9
NF4_BYTES = 4.127 / 8  # 4-bit NF4 + double-quantized block scales

# hidden size, MLP size, layers, key/value width (GQA), total params, embedding + LM-head params
ARCH = {
    "Qwen2.5-0.5B": dict(hidden=896, mlp=4864, layers=24, kv=128, params=0.494e9, embed=151936 * 896),
    "Llama-3.1-8B": dict(hidden=4096, mlp=14336, layers=32, kv=1024, params=8.03e9, embed=2 * 128256 * 4096),
    "Llama-3.1-70B": dict(hidden=8192, mlp=28672, layers=80, kv=1024, params=70.6e9, embed=2 * 128256 * 8192),
}


def lora_params(a, r):
    """A LoRA pair on a d_in x d_out layer adds r * (d_in + d_out) parameters."""
    h, m, kv = a["hidden"], a["mlp"], a["kv"]
    per_layer = (h + h) + (h + kv) * 2 + (h + h) + (h + m) * 2 + (m + h)  # q, k, v, o, gate, up, down
    return a["layers"] * r * per_layer


def safetensors_stats(paths):
    from safetensors import safe_open

    params = quantizable = 0
    for path in paths:
        with safe_open(path, "pt") as f:
            for k in f.keys():
                shape = f.get_slice(k).get_shape()
                n = 1
                for d in shape:
                    n *= d
                params += n
                if ".layers." in k and len(shape) == 2:  # transformer-block linear weights get 4-bit
                    quantizable += n
    return params, quantizable, sum(os.path.getsize(p) for p in paths)


def measured():
    from huggingface_hub import snapshot_download

    base_dir = snapshot_download(BASE_MODEL, allow_patterns=["*.safetensors", "*.json"])
    params, quant, size = safetensors_stats(glob.glob(os.path.join(base_dir, "*.safetensors")))
    nf4 = quant * NF4_BYTES + (params - quant) * 2
    rows = [{"name": f"{BASE_MODEL} (16-bit)", "params": params, "mb": size / MB},
            {"name": f"{BASE_MODEL} (4-bit NF4, est.)", "params": params, "mb": nf4 / MB}]
    for cfg in sorted(glob.glob(os.path.join(OUT_DIR, "*", "adapter_model.safetensors"))):
        a_params, _, a_size = safetensors_stats([cfg])
        rows.append({"name": f"adapter: {os.path.basename(os.path.dirname(cfg))}", "params": a_params,
                     "mb": a_size / MB})
    print(f"\nA. Measured  (base model files: {base_dir})")
    print(f"{'artifact':<48} {'params':>14} {'size MB':>9} {'% of base':>9}")
    for r in rows:
        print(f"{r['name']:<48} {r['params']:>14,} {r['mb']:>9.1f} {100 * r['mb'] / rows[0]['mb']:>8.2f}%")
    print("Note: merging an adapter back (merge_and_unload) gives a model the same size as the base.")
    return rows


def projected(rank=16):
    rows = []
    for name, a in ARCH.items():
        P, A = a["params"], lora_params(a, rank)
        W4 = (P - a["embed"]) * NF4_BYTES + a["embed"] * 2  # embeddings / LM head stay 16-bit
        rows.append({
            "model": name,
            "params_b": P / 1e9,
            "weights_16bit_gb": P * 2 / GB,
            "weights_4bit_gb": W4 / GB,
            "adapter_params_m": A / 1e6,
            "adapter_mb_16bit": A * 2 / MB,
            "adapter_pct": 100 * A / P,
            # training memory without activations: weights + grads + Adam states (+ fp32 master copy)
            "train_full_gb": P * 16 / GB,  # 2 W + 2 grad + 4 master + 8 Adam
            "train_lora_gb": (P * 2 + A * 16) / GB,
            "train_qlora_gb": (W4 + A * 16) / GB,
        })
    print(f"\nB. Projected for LoRA rank {rank} on all linear layers (memory excludes activations)")
    print(f"{'model':<14} {'16-bit':>8} {'4-bit':>7} {'adapter':>9} {'adapter %':>9} | "
          f"{'train: full':>11} {'LoRA':>7} {'QLoRA':>7}")
    for r in rows:
        print(f"{r['model']:<14} {r['weights_16bit_gb']:>6.1f}GB {r['weights_4bit_gb']:>5.1f}GB "
              f"{r['adapter_mb_16bit']:>7.0f}MB {r['adapter_pct']:>8.2f}% | {r['train_full_gb']:>9.0f}GB "
              f"{r['train_lora_gb']:>5.1f}GB {r['train_qlora_gb']:>5.1f}GB")
    return rows


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--theory", action="store_true", help="skip the measured part (no downloads)")
    p.add_argument("--rank", type=int, default=16)
    args = p.parse_args()
    report = {"measured": None if args.theory else measured(), "projected": projected(args.rank)}
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "size_report.json"), "w") as f:
        json.dump(report, f, indent=2)
