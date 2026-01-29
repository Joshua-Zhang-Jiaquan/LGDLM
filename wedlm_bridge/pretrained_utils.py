from __future__ import annotations

import os
import time
from pathlib import Path


def _get_model_cfg(config):
    return getattr(config, "model", config)


def resolve_pretrained_path(config, name_or_path: str) -> tuple[str, bool]:
    """Resolve a pretrained model/tokenizer path with optional local caching.

    Returns (resolved_path, local_files_only).
    """
    model_cfg = _get_model_cfg(config)
    local_files_only = bool(getattr(model_cfg, "local_files_only", False))

    pretrained_local_dir = getattr(model_cfg, "pretrained_local_dir", None)
    if pretrained_local_dir:
        local_dir = Path(pretrained_local_dir)
        has_config = local_dir.exists() and (local_dir / "config.json").exists()
        if has_config:
            return str(local_dir), True

        if bool(getattr(model_cfg, "download_pretrained_local_dir", False)):
            from huggingface_hub import snapshot_download

            local_dir.parent.mkdir(parents=True, exist_ok=True)
            lock_path = local_dir.parent / f".{local_dir.name}.download.lock"

            while True:
                if (local_dir / "config.json").exists():
                    break
                try:
                    fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
                except FileExistsError:
                    time.sleep(5)
                    continue
                else:
                    try:
                        os.write(fd, str(os.getpid()).encode("utf-8"))
                    finally:
                        os.close(fd)
                    try:
                        snapshot_download(
                            repo_id=name_or_path,
                            local_dir=str(local_dir),
                            local_dir_use_symlinks=False,
                        )
                    finally:
                        try:
                            lock_path.unlink()
                        except FileNotFoundError:
                            pass
                    break
            return str(local_dir), True

    return name_or_path, local_files_only

