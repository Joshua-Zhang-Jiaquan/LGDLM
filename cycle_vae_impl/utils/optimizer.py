import torch


def build_optimizer(cfg, model):
    opt = cfg.optimizer
    name = str(opt.get("type") or opt.get("name") or "adamw").lower()
    lr = float(opt.get("lr", 1e-4))
    wd = float(opt.get("weight_decay", 0.0))

    if name in ("adam", "adamw"):
        beta1 = float(opt.get("beta1", 0.9))
        beta2 = float(opt.get("beta2", 0.999))
        eps = float(opt.get("eps", 1e-8))
        return torch.optim.AdamW(model.parameters(), lr=lr, betas=(beta1, beta2), eps=eps, weight_decay=wd)

    raise ValueError(f"Unsupported optimizer: {name}")
