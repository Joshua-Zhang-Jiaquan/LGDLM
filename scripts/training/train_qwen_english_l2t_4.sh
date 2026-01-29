#!/bin/bash
# File: train_l2t_fixed_4nodes.sh
set -xeuo pipefail

# ============================================
# 环境配置
# ============================================
cd /inspire/hdd/global_user/zhangjiaquan-253108540222/latent/HDLM
source .venv/bin/activate
cd /inspire/hdd/global_user/zhangjiaquan-253108540222/latent/MM-LDLM

# ============================================
# NCCL超时修复
# ============================================
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200  # 2小时
export NCCL_IB_TIMEOUT=90
export NCCL_SOCKET_TIMEOUT=90
export NCCL_BLOCKING_WAIT=0
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO

export TORCH_DISTRIBUTED_TIMEOUT=7200
export DATA_WORKERS=8

# ============================================
# 内存优化
# ============================================
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

export WANDB_DISABLED="true"
export WANDB_MODE="disabled"
export WANDB_DIR="./output_dir/wandb"
mkdir -p "${WANDB_DIR}"

# ============================================
# ✅ 关键修改1：改为4节点
# ============================================
NNODES="${NNODES:-4}"                    # ⬅️ 改为4
NODE_RANK="${NODE_RANK:-0}"              # 节点排名：0, 1, 2, 3
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"    # 每个节点8个GPU

# ============================================
# ✅ 关键修改2：主节点地址（节点0）
# ============================================
if [ "${NODE_RANK}" -eq 0 ]; then
    MASTER_ADDR="${MASTER_ADDR:-$(hostname -I | awk '{print $1}')}"
else
    MASTER_ADDR="${MASTER_ADDR:-192.168.1.100}"  # 主节点IP
fi

MASTER_PORT="${MASTER_PORT:-29500}"

# ============================================
# 验证配置
# ============================================
echo "=========================================="
echo "分布式训练配置（4节点）"
echo "=========================================="
echo "节点总数 (NNODES):     ${NNODES}"
echo "当前节点排名 (NODE_RANK): ${NODE_RANK}"
echo "每节点GPU数 (NPROC_PER_NODE): ${NPROC_PER_NODE}"
echo "主节点地址 (MASTER_ADDR): ${MASTER_ADDR}"
echo "主节点端口 (MASTER_PORT): ${MASTER_PORT}"
echo "总GPU数: $((NNODES * NPROC_PER_NODE)) = 32 GPU"
echo "=========================================="

# ============================================
# 重要提示
# ============================================
if [ "${NODE_RANK}" -eq 0 ]; then
    echo ""
    echo "⚠️  注意：4节点×8GPU=32GPU训练"
    echo "⚠️  数据集很大，第一次加载可能需要30-60分钟！"
    echo "⚠️  请确保主节点(${MASTER_ADDR})的${MASTER_PORT}端口已开放"
    echo "⚠️  请等待所有4个节点都启动后再继续..."
    echo ""
    read -p "按Enter键开始训练，或Ctrl+C取消..." </dev/tty
fi

echo "等待15秒，确保所有节点网络就绪..."
sleep 15

# ============================================
# 创建日志目录
# ============================================
mkdir -p output_dir/l2t_logs
LOG_FILE="output_dir/l2t_logs/train_$(date +%Y%m%d_%H%M%S)_node${NODE_RANK}.log"

# ============================================
# ✅ 关键修改3：训练命令（改rdzv_id）
# ============================================
echo "开始训练... 日志保存到: ${LOG_FILE}"
echo "开始时间: $(date)"

torchrun \
    --nnodes=${NNODES} \
    --nproc_per_node=${NPROC_PER_NODE} \
    --node_rank=${NODE_RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    --rdzv_id=mmdit_l2t_4nodes \           # ⬅️ 改个名字
    --rdzv_backend=c10d \
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    --max_restarts=0 \
    --local_addr=$(hostname -I | awk '{print $1}') \
    latentDLM_mmdit/train_mmdit.py \
    --config-name mmdit_preprocessed \
    logging.run_name="mmdit-qwen-32d-l2t-4nodes" \  # ⬅️ 改个名字
    logging.save_dir="/inspire/hdd/global_user/zhangjiaquan-253108540222/latent/MM-LDLM/saved" \
    training.train_batch_size=4 \
    training.eval_batch_size=4 \
    training.num_epochs=10 \
    training.compile_model=false \
    loss.loss_type="l2t" \
    model.latent_dim=32 \
    training.dtype=bf16 \
    logging.save_freq=50000 \
    logging.log_freq=10000 \
    logging.eval_freq=10000 \
    data.num_workers=${DATA_WORKERS} \
    data.token_dir="/inspire/ssd/project/future-reading/public/jiaquan/preprocessed_data/qwen-embeddings-32/tokens/train" \
    data.latent_dir="/inspire/ssd/project/future-reading/public/jiaquan/preprocessed_data/qwen-embeddings-32/latents/train" \
    2>&1 | tee ${LOG_FILE}

# ============================================
# 训练完成
# ============================================
TRAINING_EXIT_CODE=${PIPESTATUS[0]}
echo ""
echo "训练结束时间: $(date)"
echo "退出码: ${TRAINING_EXIT_CODE}"

if [ ${TRAINING_EXIT_CODE} -eq 0 ]; then
    echo "✅ 4节点训练成功完成！"
else
    echo "❌ 训练失败，退出码: ${TRAINING_EXIT_CODE}"
    echo "查看日志文件: ${LOG_FILE}"
    exit ${TRAINING_EXIT_CODE}
fi