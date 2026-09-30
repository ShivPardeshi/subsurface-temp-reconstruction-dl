import torch

ckpt_path = "checkpoints/phase3_retrain_scratch_40k_full/last_checkpoint.pt"
c = torch.load(ckpt_path, map_location="cpu")
step = c.get("step")
epoch = c.get("epoch")
lr = c.get("scheduler", {}).get("_last_lr", ["unknown"])
metrics = c.get("metrics", {})
print(f"CHECKPOINT_INFO: step={step} epoch={epoch} lr={lr} metrics={metrics}")
