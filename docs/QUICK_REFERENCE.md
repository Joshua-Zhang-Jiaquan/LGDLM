# Quick Reference Card

## 🚀 Start Training (Most Common Commands)

### Test on Single Node (Start Here!)
```bash
cd /inspire/hdd/project/project-public/zhangjiaquan-253108540222/jiaquan/latent/MM-LDLM

# Test with 2 GPUs
NNODES=1 NPROC_PER_NODE=2 bash scripts/training/train_qwen_english_l2t_stable.sh
```

### Scale to Multiple Nodes
```bash
# 2 nodes (16 GPUs)
NNODES=2 bash scripts/training/train_qwen_english_l2t_stable.sh

# 8 nodes (64 GPUs)
NNODES=8 bash scripts/training/train_qwen_english_l2t_stable.sh
```

### Conservative Training (If NaN Still Occurs)
```bash
LEARNING_RATE=3e-5 GRAD_CLIP_NORM=0.3 WARMUP_STEPS=3000 \
bash scripts/training/train_qwen_english_l2t_stable.sh
```

## 📊 Monitor Training

```bash
# Watch logs in real-time
tail -f train_logs/train_*_node0.log

# Check for errors
grep -E "ERROR|NaN" train_logs/train_*_node0.log

# Monitor loss
tail -f train_logs/train_*_node0.log | grep "Loss:"
```

## 🔧 Common Overrides

```bash
# Lower learning rate
LEARNING_RATE=3e-5 bash scripts/training/train_qwen_english_l2t_stable.sh

# Smaller batch size (if OOM)
L2T_TRAIN_BS=2 bash scripts/training/train_qwen_english_l2t_stable.sh

# More gradient accumulation
GRAD_ACCUM_STEPS=4 bash scripts/training/train_qwen_english_l2t_stable.sh

# Custom master address
MASTER_ADDR=192.168.1.100 bash scripts/training/train_qwen_english_l2t_stable.sh
```

## 📁 Important Files

| File | Purpose |
|------|---------|
| `scripts/training/train_qwen_english_l2t_stable.sh` | **NEW training script** (use this!) |
| `docs/STABLE_TRAINING_GUIDE.md` | Complete usage guide |
| `docs/NAN_FIX_SUMMARY.md` | Quick start for NaN fixes |
| `docs/NAN_GRADIENT_FIXES.md` | Detailed technical analysis |
| `results/archive/TRAIN_MMDIT_PATCH.py` | Code patches for train_mmdit.py |
| `results/archive/IMPROVED_TRAINER_PATCH.py` | Code patches for improved_trainer.py |
| `results/archive/mmdit_stable_config.yaml` | Stable hyperparameters |
| `scripts/utils/apply_nan_fixes.sh` | Helper to apply patches |

## ⚠️ Before First Run

### 1. Apply Code Patches (Required!)
```bash
# Run helper script
bash scripts/utils/apply_nan_fixes.sh

# Then manually apply patches from:
# - results/archive/TRAIN_MMDIT_PATCH.py
# - results/archive/IMPROVED_TRAINER_PATCH.py
```

### 2. Verify Setup
```bash
# Check GPUs
nvidia-smi

# Check data paths
ls /inspire/ssd/project/future-reading/public/jiaquan/preprocessed_data/qwen-embeddings-32/tokens/train
ls /inspire/ssd/project/future-reading/public/jiaquan/preprocessed_data/qwen-embeddings-32/latents/train

# Check config
ls latentDLM_mmdit/configs/mmdit_stable.yaml
```

## 🎯 Success Indicators

✅ **Good Training:**
- No "ERROR" or "NaN" messages
- Loss decreasing smoothly
- Gradient norms < 1.0
- No connection errors

❌ **Problems:**
- "ERROR: Invalid loss" → Reduce learning rate
- "ERROR: Invalid gradient norm" → Reduce gradient clip
- "DistNetworkError" → Check network/firewall
- "Cannot connect to master" → Set MASTER_ADDR explicitly

## 🆘 Quick Troubleshooting

| Problem | Solution |
|---------|----------|
| NaN gradients | `LEARNING_RATE=3e-5 GRAD_CLIP_NORM=0.3` |
| Connection failed | `MASTER_ADDR=<ip_address>` |
| Out of memory | `L2T_TRAIN_BS=2 GRAD_ACCUM_STEPS=4` |
| Port blocked | `MASTER_PORT=12345` |
| Too slow | `L2T_TRAIN_BS=8 GRAD_ACCUM_STEPS=1` |

## 📈 Hyperparameter Defaults

| Parameter | Default | Conservative | Aggressive |
|-----------|---------|--------------|------------|
| Learning rate | 5e-5 | 3e-5 | 1e-4 |
| Gradient clip | 0.5 | 0.3 | 1.0 |
| Warmup steps | 2000 | 3000 | 1000 |
| Latent loss weight | 0.1 | 0.05 | 0.5 |
| Grad accumulation | 2 | 4 | 1 |

## 🔍 Log File Locations

```bash
# Training logs (timestamped)
train_logs/train_YYYYMMDD_HHMMSS_node0.log

# NCCL debug logs
/tmp/nccl_debug_0.log

# Checkpoints
/inspire/hdd/global_user/.../saved/mmdit-qwen-32d-l2t-stable/

# Training metrics
saved/mmdit-qwen-32d-l2t-stable/training_log.jsonl
```

## 💡 Pro Tips

1. **Always test single node first**: `NNODES=1 NPROC_PER_NODE=2`
2. **Monitor first 1000 steps closely**: Watch for NaN or errors
3. **Save logs**: They're automatically timestamped
4. **Scale gradually**: 1→2→4→8 nodes
5. **Use conservative settings first**: Can increase later

## 📞 Getting Help

If issues persist:
1. Check `docs/STABLE_TRAINING_GUIDE.md` for detailed troubleshooting
2. Review `docs/NAN_GRADIENT_FIXES.md` for technical details
3. Share log files: `train_logs/train_*_node0.log`
4. Check NCCL logs: `/tmp/nccl_debug_0.log`

---

**Remember**: Apply code patches before first run! See `scripts/utils/apply_nan_fixes.sh`
