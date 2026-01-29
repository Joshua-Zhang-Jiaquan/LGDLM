#!/bin/bash
# File: train_l2t_fixed.sh
set -xeuo pipefail

# Initialize environment
cd /inspire/hdd/global_user/zhangjiaquan-253108540222/latent/HDLM
source .venv/bin/activate
cd /inspire/hdd/global_user/zhangjiaquan-253108540222/latent/MM-LDLM

# Get environment variables with defaults
NODE_RANK="${RANK:-0}"
WORLD_SIZE="${WORLD_SIZE:-2}"
MASTER_ADDR="${MASTER_ADDR:-$(hostname -I | awk '{print $1}')}"
MASTER_PORT="${MASTER_PORT:-29500}"

# 关键修复：使用固定的 rdzv_id，确保所有节点相同
export RENDEZVOUS_ID="mmdit_extreme_${WORLD_SIZE}nodes_$(date +%Y%m%d)"

# 使用新的环境变量名
export PYTORCH_ALLOC_CONF="max_split_size_mb:512,garbage_collection_threshold:0.8"
export WANDB_DISABLED="true"
export WANDB_MODE="disabled"

# NCCL 设置 - 适度超时
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=6000  # 10分钟，不是2小时
export NCCL_IB_TIMEOUT=22
export NCCL_SOCKET_TIMEOUT=22
export NCCL_BLOCKING_WAIT=0
export NCCL_ASYNC_ERROR_HANDLING=1

# 调试信息
export TORCH_DISTRIBUTED_DEBUG=INFO
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV

echo "=== 分布式训练开始 ==="
echo "节点排名: ${NODE_RANK}"
echo "总节点数: ${WORLD_SIZE}"
echo "主节点地址: ${MASTER_ADDR}"
echo "主节点端口: ${MASTER_PORT}"
echo "Rendezvous ID: ${RENDEZVOUS_ID}"
echo "本地IP地址: $(hostname -I)"

# 创建输出目录
mkdir -p output_dir/l2t_logs

# 关键：首先在主节点上启动 rendezvous 服务器
if [ "${NODE_RANK}" == "0" ]; then
    echo "主节点（排名 0）正在启动..."
    echo "等待其他节点连接..."
fi

# 等待所有节点就绪
sleep 5

# 运行训练 - 简化参数
if [ ${WORLD_SIZE} -gt 1 ]; then
    echo "多节点模式启动..."
    torchrun \
        --nnodes=${WORLD_SIZE} \
        --nproc_per_node=8 \
        --node_rank=${NODE_RANK} \
        --master_addr=${MASTER_ADDR} \
        --master_port=${MASTER_PORT} \
        --rdzv_id=${RENDEZVOUS_ID} \
        --rdzv_backend=c10d \
        --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
        --max_restarts=3 \
        --monitor_interval=30 \
        latentDLM_mmdit/train_mmdit.py \
        --config-name mmdit_preprocessed \
        logging.run_name="mmdit-qwen-32d-l2t-fixed" \
        logging.save_dir="/inspire/hdd/global_user/zhangjiaquan-253108540222/latent/MM-LDLM/saved" \
        training.train_batch_size=4 \
        training.eval_batch_size=4 \
        training.num_epochs=2 \  # 先试2个epoch
        training.compile_model=false \
        loss.loss_type="l2t" \
        model.latent_dim=32 \
        training.dtype=bf16 \
        logging.save_freq=1000 \
        logging.log_freq=100 \
        logging.eval_freq=500 \
        data.num_workers=4 \  # 减少worker数
        data.token_dir="/inspire/ssd/project/future-reading/public/jiaquan/preprocessed_data/qwen-embeddings-32/tokens/train" \
        data.latent_dir="/inspire/ssd/project/future-reading/public/jiaquan/preprocessed_data/qwen-embeddings-32/latents/train"
else
    echo "单节点模式启动..."
    torchrun \
        --nnodes=1 \
        --nproc_per_node=8 \
        --rdzv_id=${RENDEZVOUS_ID} \
        --rdzv_backend=c10d \
        --rdzv_endpoint="127.0.0.1:${MASTER_PORT}" \
        latentDLM_mmdit/train_mmdit.py \
        --config-name mmdit_preprocessed \
        logging.run_name="mmdit-qwen-32d-l2t-single" \
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
        data.num_workers=16 \
        data.token_dir="/inspire/ssd/project/future-reading/public/jiaquan/preprocessed_data/qwen-embeddings-32/tokens/train" \
        data.latent_dir="/inspire/ssd/project/future-reading/public/jiaquan/preprocessed_data/qwen-embeddings-32/latents/train"
fi