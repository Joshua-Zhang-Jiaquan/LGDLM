import math


def get_lr(cfg, base_lr: float, step: int) -> float:
    sched = str(cfg.training.get("lr_schedule", "constant")).lower()
    warmup = int(cfg.training.get("warmup_steps", 0))
    total = int(cfg.training.get("num_train_steps", 1))

    if warmup > 0 and step < warmup:
        return base_lr * (step + 1) / warmup

    if sched == "constant":
        return base_lr
    if sched == "linear":
        if total <= warmup:
            return base_lr
        t = (step - warmup) / max(1, (total - warmup))
        return base_lr * max(0.0, 1.0 - t)
    if sched == "cosine":
        if total <= warmup:
            return base_lr
        t = (step - warmup) / max(1, (total - warmup))
        return base_lr * (0.1 + 0.9 * (1 + math.cos(math.pi * t)) / 2)

    raise ValueError(f"Unknown lr_schedule: {sched}")
