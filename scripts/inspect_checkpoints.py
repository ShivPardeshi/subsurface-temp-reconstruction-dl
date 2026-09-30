import torch
import glob
from pathlib import Path

def main():
    print("=" * 60)
    print("INSPECTING CHECKPOINTS")
    print("=" * 60)
    for p in sorted(glob.glob("checkpoints/*")):
        path = Path(p)
        if path.is_dir():
            best = path / "best_checkpoint.pt"
            last = path / "last_checkpoint.pt"
            info = []
            if best.exists():
                try:
                    c = torch.load(best, map_location="cpu", weights_only=False)
                    info.append(f"best_step={c.get('step')} best_rmse={c.get('metrics', {}).get('best_val_rmse', 'N/A'):.4f}")
                except Exception as e:
                    info.append(f"best error: {e}")
            if last.exists():
                try:
                    c = torch.load(last, map_location="cpu", weights_only=False)
                    info.append(f"last_step={c.get('step')}")
                except Exception as e:
                    info.append(f"last error: {e}")
            print(f"{path.name}: {', '.join(info) if info else 'empty'}")

if __name__ == "__main__":
    main()
