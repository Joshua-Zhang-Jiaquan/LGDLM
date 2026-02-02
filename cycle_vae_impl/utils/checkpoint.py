from pathlib import Path

import torch


def save_checkpoint(path, model, optimizer, step: int, cfg: dict):
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    if hasattr(model, "module"):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()

    ckpt = {
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "step": int(step),
        "cfg": cfg,
        "rng_torch": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    torch.save(ckpt, str(path_obj))


def load_checkpoint(path: str, model, optimizer=None, device=None):
    ckpt = torch.load(str(path), map_location="cpu")

    if hasattr(model, "module"):
        model.module.load_state_dict(ckpt["model_state_dict"], strict=True)
    else:
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
    if optimizer is not None and "optimizer_state_dict" in ckpt and ckpt["optimizer_state_dict"] is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    # RNG restore is best-effort.
    if "rng_torch" in ckpt and ckpt["rng_torch"] is not None:
        torch.set_rng_state(ckpt["rng_torch"])
    if torch.cuda.is_available() and "rng_cuda" in ckpt and ckpt["rng_cuda"] is not None:
        try:
            torch.cuda.set_rng_state_all(ckpt["rng_cuda"])
        except Exception:
            pass

    return int(ckpt.get("step", 0)), ckpt.get("cfg", None)
