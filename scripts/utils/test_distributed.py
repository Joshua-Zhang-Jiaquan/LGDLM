#!/usr/bin/env python3
"""
Diagnostic script to test distributed training setup.
Run with: torchrun --nnodes=N --nproc_per_node=M --node_rank=R --master_addr=ADDR --master_port=PORT test_distributed.py
"""
import os
import socket
import datetime
import torch
import torch.distributed as dist
from pathlib import Path

def print_env_info():
    """Print environment variables and system info."""
    print("=" * 80)
    print("ENVIRONMENT INFORMATION")
    print("=" * 80)
    print(f"Hostname: {socket.gethostname()}")
    print(f"IP Address: {socket.gethostbyname(socket.gethostname())}")
    print(f"RANK: {os.environ.get('RANK', 'NOT SET')}")
    print(f"WORLD_SIZE: {os.environ.get('WORLD_SIZE', 'NOT SET')}")
    print(f"LOCAL_RANK: {os.environ.get('LOCAL_RANK', 'NOT SET')}")
    print(f"MASTER_ADDR: {os.environ.get('MASTER_ADDR', 'NOT SET')}")
    print(f"MASTER_PORT: {os.environ.get('MASTER_PORT', 'NOT SET')}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"NCCL version: {torch.cuda.nccl.version() if torch.cuda.is_available() else 'N/A'}")
    print("=" * 80)

def test_network_connectivity():
    """Test if master node is reachable."""
    print("\n" + "=" * 80)
    print("NETWORK CONNECTIVITY TEST")
    print("=" * 80)

    master_addr = os.environ.get('MASTER_ADDR', 'localhost')
    master_port = int(os.environ.get('MASTER_PORT', '29500'))

    print(f"Testing connection to {master_addr}:{master_port}...")

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex((master_addr, master_port))
        sock.close()

        if result == 0:
            print(f"✓ Port {master_port} is OPEN on {master_addr}")
            return True
        else:
            print(f"✗ Port {master_port} is CLOSED on {master_addr} (error code: {result})")
            return False
    except socket.gaierror as e:
        print(f"✗ Hostname resolution failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Connection test failed: {e}")
        return False

def test_cuda_setup():
    """Test CUDA device setup."""
    print("\n" + "=" * 80)
    print("CUDA SETUP TEST")
    print("=" * 80)

    if not torch.cuda.is_available():
        print("✗ CUDA is not available!")
        return False

    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    num_gpus = torch.cuda.device_count()

    print(f"LOCAL_RANK: {local_rank}")
    print(f"Available GPUs: {num_gpus}")

    if local_rank >= num_gpus:
        print(f"✗ LOCAL_RANK ({local_rank}) >= num_gpus ({num_gpus})")
        return False

    try:
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        print(f"✓ Successfully set CUDA device to {device}")
        print(f"  Device name: {torch.cuda.get_device_name(local_rank)}")
        print(f"  Device capability: {torch.cuda.get_device_capability(local_rank)}")

        # Test tensor allocation
        test_tensor = torch.randn(100, 100, device=device)
        print(f"✓ Successfully allocated test tensor on {device}")
        return True
    except Exception as e:
        print(f"✗ CUDA setup failed: {e}")
        return False

def test_distributed_init():
    """Test distributed process group initialization."""
    print("\n" + "=" * 80)
    print("DISTRIBUTED INITIALIZATION TEST")
    print("=" * 80)

    if not all(k in os.environ for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK")):
        print("✗ Not launched with torchrun (missing RANK/WORLD_SIZE/LOCAL_RANK)")
        return False

    local_rank = int(os.environ["LOCAL_RANK"])

    try:
        print("Attempting to initialize process group...")
        print(f"  Backend: nccl")
        print(f"  Timeout: 60 seconds")
        print(f"  Init method: env://")

        dist.init_process_group(
            backend="nccl",
            timeout=datetime.timedelta(seconds=60),
            init_method="env://",
        )

        rank = dist.get_rank()
        world_size = dist.get_world_size()

        print(f"✓ Process group initialized successfully!")
        print(f"  Rank: {rank}/{world_size}")
        print(f"  Backend: {dist.get_backend()}")

        return True
    except Exception as e:
        print(f"✗ Failed to initialize process group: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_collective_ops():
    """Test basic collective operations."""
    print("\n" + "=" * 80)
    print("COLLECTIVE OPERATIONS TEST")
    print("=" * 80)

    if not dist.is_initialized():
        print("✗ Process group not initialized, skipping collective ops test")
        return False

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")

    try:
        # Test barrier
        print("Testing barrier synchronization...")
        dist.barrier()
        print(f"✓ Rank {rank}: Barrier passed")

        # Test all_reduce
        print("Testing all_reduce...")
        tensor = torch.ones(10, device=device) * rank
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        expected_sum = sum(range(world_size))

        if torch.allclose(tensor, torch.ones(10, device=device) * expected_sum):
            print(f"✓ Rank {rank}: all_reduce successful (sum={expected_sum})")
        else:
            print(f"✗ Rank {rank}: all_reduce failed (expected {expected_sum}, got {tensor[0].item()})")
            return False

        # Test broadcast
        print("Testing broadcast...")
        tensor = torch.ones(10, device=device) * rank
        dist.broadcast(tensor, src=0)

        if torch.allclose(tensor, torch.zeros(10, device=device)):
            print(f"✓ Rank {rank}: broadcast successful")
        else:
            print(f"✗ Rank {rank}: broadcast failed")
            return False

        return True
    except Exception as e:
        print(f"✗ Rank {rank}: Collective operation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_paths():
    """Test if data paths are accessible."""
    print("\n" + "=" * 80)
    print("DATA PATH ACCESSIBILITY TEST")
    print("=" * 80)

    # Test paths from the training script
    token_dir = Path("/inspire/ssd/project/future-reading/public/jiaquan/preprocessed_data/qwen-embeddings-32/tokens/train")
    latent_dir = Path("/inspire/ssd/project/future-reading/public/jiaquan/preprocessed_data/qwen-embeddings-32/latents/train")

    print(f"Testing token_dir: {token_dir}")
    if token_dir.exists():
        token_files = list(token_dir.glob("*.npz"))
        print(f"✓ Token directory exists with {len(token_files)} .npz files")
    else:
        print(f"✗ Token directory does not exist!")

    print(f"\nTesting latent_dir: {latent_dir}")
    if latent_dir.exists():
        latent_files = list(latent_dir.glob("*.npy"))
        print(f"✓ Latent directory exists with {len(latent_files)} .npy files")
    else:
        print(f"✗ Latent directory does not exist!")

def main():
    """Run all diagnostic tests."""
    print("\n" + "=" * 80)
    print("DISTRIBUTED TRAINING DIAGNOSTIC TOOL")
    print("=" * 80)

    # Print environment info
    print_env_info()

    # Test network connectivity (before init)
    network_ok = test_network_connectivity()

    # Test CUDA setup
    cuda_ok = test_cuda_setup()

    # Test data paths
    test_data_paths()

    # Test distributed initialization
    if cuda_ok:
        dist_ok = test_distributed_init()

        # Test collective operations
        if dist_ok:
            collective_ok = test_collective_ops()
        else:
            collective_ok = False
    else:
        dist_ok = False
        collective_ok = False

    # Summary
    print("\n" + "=" * 80)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 80)
    print(f"Network connectivity: {'✓ PASS' if network_ok else '✗ FAIL'}")
    print(f"CUDA setup: {'✓ PASS' if cuda_ok else '✗ FAIL'}")
    print(f"Distributed init: {'✓ PASS' if dist_ok else '✗ FAIL'}")
    print(f"Collective ops: {'✓ PASS' if collective_ok else '✗ FAIL'}")
    print("=" * 80)

    if dist.is_initialized():
        dist.destroy_process_group()

    # Exit with appropriate code
    if all([network_ok, cuda_ok, dist_ok, collective_ok]):
        print("\n✓ All tests passed! Distributed setup is working correctly.")
        exit(0)
    else:
        print("\n✗ Some tests failed. Please review the output above.")
        exit(1)

if __name__ == "__main__":
    main()
