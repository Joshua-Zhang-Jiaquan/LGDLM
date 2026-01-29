#!/bin/bash
# benchmark_training.sh
# Benchmark current training performance to measure improvements

set -e

echo "=========================================="
echo "Training Performance Benchmark"
echo "=========================================="
echo ""

# Configuration
BENCHMARK_STEPS=100
BENCHMARK_LOG="benchmark_results_$(date +%Y%m%d_%H%M%S).log"

echo "Configuration:"
echo "  Benchmark steps: $BENCHMARK_STEPS"
echo "  Log file: $BENCHMARK_LOG"
echo ""

# Create benchmark directory
mkdir -p benchmarks
cd benchmarks

# Function to extract metrics from logs
extract_metrics() {
    local log_file=$1
    local output_file=$2

    echo "Extracting metrics from $log_file..."

    # Extract timing information
    python3 << 'PYTHON_SCRIPT'
import sys
import json
import re
from pathlib import Path

log_file = sys.argv[1]
output_file = sys.argv[2]

metrics = {
    'step_times': [],
    'losses': [],
    'text_losses': [],
    'latent_losses': [],
    'text_accuracies': [],
    'gpu_memory': [],
    'throughput': [],
}

with open(log_file, 'r') as f:
    for line in f:
        # Extract step time
        if 'it/s' in line or 's/it' in line:
            match = re.search(r'(\d+\.\d+)(it/s|s/it)', line)
            if match:
                val = float(match.group(1))
                if 's/it' in match.group(2):
                    metrics['step_times'].append(val)
                else:
                    metrics['step_times'].append(1.0 / val)

        # Extract loss values
        if 'Loss:' in line:
            match = re.search(r'Loss:\s*(\d+\.\d+)', line)
            if match:
                metrics['losses'].append(float(match.group(1)))

        # Extract text/latent losses
        if 'Text:' in line:
            match = re.search(r'Text:\s*(\d+\.\d+)', line)
            if match:
                metrics['text_losses'].append(float(match.group(1)))

        if 'Latent:' in line:
            match = re.search(r'Latent:\s*(\d+\.\d+)', line)
            if match:
                metrics['latent_losses'].append(float(match.group(1)))

        # Extract accuracy
        if 'Acc:' in line:
            match = re.search(r'Acc:\s*(\d+\.\d+)', line)
            if match:
                metrics['text_accuracies'].append(float(match.group(1)))

        # Extract throughput
        if 'samples_per_sec' in line:
            match = re.search(r'samples_per_sec["\']:\s*(\d+\.\d+)', line)
            if match:
                metrics['throughput'].append(float(match.group(1)))

# Calculate statistics
stats = {}
for key, values in metrics.items():
    if values:
        stats[key] = {
            'mean': sum(values) / len(values),
            'min': min(values),
            'max': max(values),
            'count': len(values)
        }

with open(output_file, 'w') as f:
    json.dump(stats, f, indent=2)

print(f"✓ Metrics extracted to {output_file}")
PYTHON_SCRIPT "$log_file" "$output_file"
}

# Function to run benchmark
run_benchmark() {
    local name=$1
    local description=$2

    echo ""
    echo "=========================================="
    echo "Running: $name"
    echo "Description: $description"
    echo "=========================================="

    local log_file="benchmark_${name}_$(date +%Y%m%d_%H%M%S).log"

    # Run training for benchmark steps
    cd ..
    NNODES=1 NPROC_PER_NODE=2 \
    L2T_BASE_STEPS=$BENCHMARK_STEPS \
    L2T_BASE_SAVE_FREQ=999999 \
    L2T_BASE_LOG_FREQ=10 \
    bash train_qwen_english_l2t_stable.sh 2>&1 | tee "benchmarks/$log_file"

    cd benchmarks

    # Extract metrics
    extract_metrics "$log_file" "${name}_metrics.json"

    echo "✓ Benchmark complete: $name"
}

# Function to compare benchmarks
compare_benchmarks() {
    echo ""
    echo "=========================================="
    echo "Benchmark Comparison"
    echo "=========================================="

    python3 << 'PYTHON_SCRIPT'
import json
import sys
from pathlib import Path

# Find all metric files
metric_files = sorted(Path('.').glob('*_metrics.json'))

if len(metric_files) < 2:
    print("Need at least 2 benchmarks to compare")
    sys.exit(0)

print(f"\nFound {len(metric_files)} benchmarks:\n")

# Load all metrics
benchmarks = {}
for f in metric_files:
    name = f.stem.replace('_metrics', '')
    with open(f) as fp:
        benchmarks[name] = json.load(fp)

# Print comparison table
print("Metric                  | " + " | ".join(f"{name:>15}" for name in benchmarks.keys()))
print("-" * (25 + 19 * len(benchmarks)))

metrics_to_compare = [
    ('step_times', 'mean', 'Step Time (s)', False),
    ('losses', 'mean', 'Loss', False),
    ('text_accuracies', 'mean', 'Text Accuracy', True),
    ('throughput', 'mean', 'Throughput (samples/s)', True),
]

baseline = list(benchmarks.values())[0]

for metric_key, stat_key, label, higher_better in metrics_to_compare:
    if metric_key in baseline and stat_key in baseline[metric_key]:
        values = []
        for name, data in benchmarks.items():
            if metric_key in data and stat_key in data[metric_key]:
                val = data[metric_key][stat_key]
                values.append(val)
            else:
                values.append(None)

        # Print row
        row = f"{label:23} |"
        baseline_val = values[0]

        for i, val in enumerate(values):
            if val is not None:
                if i == 0:
                    row += f" {val:15.4f} |"
                else:
                    # Calculate improvement
                    if baseline_val and baseline_val != 0:
                        if higher_better:
                            improvement = ((val - baseline_val) / baseline_val) * 100
                        else:
                            improvement = ((baseline_val - val) / baseline_val) * 100

                        sign = "+" if improvement > 0 else ""
                        row += f" {val:9.4f} ({sign}{improvement:+.1f}%) |"
                    else:
                        row += f" {val:15.4f} |"
            else:
                row += f" {'N/A':>15} |"

        print(row)

print("\n✓ Comparison complete")
PYTHON_SCRIPT
}

# Main benchmark flow
echo "Starting benchmark suite..."
echo ""
echo "This will:"
echo "  1. Run baseline benchmark (current code)"
echo "  2. Apply optimizations"
echo "  3. Run optimized benchmark"
echo "  4. Compare results"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Benchmark cancelled"
    exit 0
fi

# Run baseline
run_benchmark "baseline" "Current training code (before optimizations)"

# Apply fixes
echo ""
echo "Applying optimizations..."
cd ..
bash apply_critical_fixes.sh
cd benchmarks

# Run optimized
run_benchmark "optimized" "After applying critical fixes"

# Compare
compare_benchmarks

# Generate report
echo ""
echo "=========================================="
echo "Benchmark Report"
echo "=========================================="
echo ""
echo "Results saved to: benchmarks/"
echo ""
echo "Files:"
ls -lh *.json *.log 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'
echo ""
echo "To view detailed metrics:"
echo "  cat benchmarks/baseline_metrics.json"
echo "  cat benchmarks/optimized_metrics.json"
echo ""
echo "=========================================="
