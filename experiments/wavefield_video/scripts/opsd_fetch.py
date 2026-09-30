from huggingface_hub import hf_hub_download, snapshot_download

base = "/root/opsd-v"
files = [
    "checkpoints/longlive_base.pt",
    "checkpoints/longlive_lora.pt",
    "checkpoints/opsdv_longlive_lora.pt",
]
for f in files:
    print("== fetching", f, flush=True)
    p = hf_hub_download("MeiGen-AI/OPSD-V", f, local_dir=base)
    print("   ->", p, flush=True)
print("== fetching Wan2.1-T2V-1.3B snapshot", flush=True)
snapshot_download("Wan-AI/Wan2.1-T2V-1.3B", local_dir=base + "/checkpoints/Wan2.1-T2V-1.3B")
print("ALL-DONE", flush=True)
