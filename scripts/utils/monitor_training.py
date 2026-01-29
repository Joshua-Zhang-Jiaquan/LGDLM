#!/usr/bin/env python3
"""
Training Health Monitor
Monitors training logs in real-time and alerts on issues
"""

import argparse
import re
import time
import json
from pathlib import Path
from collections import deque
from datetime import datetime
import sys

class TrainingHealthMonitor:
    def __init__(self, log_file, window_size=100):
        self.log_file = Path(log_file)
        self.window_size = window_size

        # Metrics tracking
        self.losses = deque(maxlen=window_size)
        self.text_losses = deque(maxlen=window_size)
        self.latent_losses = deque(maxlen=window_size)
        self.accuracies = deque(maxlen=window_size)
        self.grad_norms = deque(maxlen=window_size)
        self.throughputs = deque(maxlen=window_size)

        # Issue tracking
        self.nan_count = 0
        self.oom_count = 0
        self.error_count = 0
        self.warning_count = 0

        # Health status
        self.health_status = "HEALTHY"
        self.issues = []

    def parse_line(self, line):
        """Parse a log line and extract metrics."""
        # Extract loss
        match = re.search(r'Loss:\s*(\d+\.\d+)', line)
        if match:
            loss = float(match.group(1))
            self.losses.append(loss)

            # Check for NaN
            if loss != loss:  # NaN check
                self.nan_count += 1
                self.add_issue("CRITICAL", f"NaN loss detected at {datetime.now()}")

        # Extract text loss
        match = re.search(r'Text:\s*(\d+\.\d+)', line)
        if match:
            self.text_losses.append(float(match.group(1)))

        # Extract latent loss
        match = re.search(r'Latent:\s*(\d+\.\d+)', line)
        if match:
            self.latent_losses.append(float(match.group(1)))

        # Extract accuracy
        match = re.search(r'Acc:\s*(\d+\.\d+)', line)
        if match:
            self.accuracies.append(float(match.group(1)))

        # Extract gradient norm
        match = re.search(r'grad_norm["\']:\s*(\d+\.\d+)', line)
        if match:
            grad_norm = float(match.group(1))
            self.grad_norms.append(grad_norm)

            # Check for exploding gradients
            if grad_norm > 10.0:
                self.add_issue("WARNING", f"High gradient norm: {grad_norm:.2f}")

        # Extract throughput
        match = re.search(r'samples_per_sec["\']:\s*(\d+\.\d+)', line)
        if match:
            self.throughputs.append(float(match.group(1)))

        # Check for errors
        if 'ERROR' in line.upper():
            self.error_count += 1
            self.add_issue("ERROR", line.strip())

        if 'WARNING' in line.upper() and 'grad_norm' not in line:
            self.warning_count += 1

        # Check for OOM
        if 'out of memory' in line.lower():
            self.oom_count += 1
            self.add_issue("CRITICAL", "Out of memory error detected")

    def add_issue(self, severity, message):
        """Add an issue to the tracking list."""
        self.issues.append({
            'severity': severity,
            'message': message,
            'timestamp': datetime.now().isoformat()
        })

        # Update health status
        if severity == "CRITICAL":
            self.health_status = "CRITICAL"
        elif severity == "ERROR" and self.health_status != "CRITICAL":
            self.health_status = "UNHEALTHY"
        elif severity == "WARNING" and self.health_status == "HEALTHY":
            self.health_status = "WARNING"

    def check_loss_convergence(self):
        """Check if loss is converging."""
        if len(self.losses) < 50:
            return None

        recent = list(self.losses)[-50:]
        first_half = sum(recent[:25]) / 25
        second_half = sum(recent[25:]) / 25

        improvement = (first_half - second_half) / first_half * 100

        if improvement < -5:  # Loss increasing by >5%
            self.add_issue("WARNING", f"Loss increasing: {improvement:.1f}% over last 50 steps")
            return False
        elif improvement < 0.1:  # No improvement
            self.add_issue("WARNING", "Loss not decreasing (plateau detected)")
            return False

        return True

    def check_throughput(self):
        """Check if throughput is stable."""
        if len(self.throughputs) < 20:
            return None

        recent = list(self.throughputs)[-20:]
        avg = sum(recent) / len(recent)
        std = (sum((x - avg) ** 2 for x in recent) / len(recent)) ** 0.5

        # Check for high variance (unstable throughput)
        if std / avg > 0.3:  # >30% variance
            self.add_issue("WARNING", f"Unstable throughput: {avg:.1f} ± {std:.1f} samples/sec")
            return False

        # Check for low throughput
        if avg < 1.0:  # Less than 1 sample/sec
            self.add_issue("WARNING", f"Low throughput: {avg:.1f} samples/sec")
            return False

        return True

    def get_statistics(self):
        """Get current statistics."""
        stats = {}

        if self.losses:
            stats['loss'] = {
                'current': self.losses[-1],
                'mean': sum(self.losses) / len(self.losses),
                'min': min(self.losses),
                'max': max(self.losses),
            }

        if self.accuracies:
            stats['accuracy'] = {
                'current': self.accuracies[-1],
                'mean': sum(self.accuracies) / len(self.accuracies),
            }

        if self.grad_norms:
            stats['grad_norm'] = {
                'current': self.grad_norms[-1],
                'mean': sum(self.grad_norms) / len(self.grad_norms),
                'max': max(self.grad_norms),
            }

        if self.throughputs:
            stats['throughput'] = {
                'current': self.throughputs[-1],
                'mean': sum(self.throughputs) / len(self.throughputs),
            }

        return stats

    def print_status(self):
        """Print current health status."""
        print("\n" + "="*70)
        print(f"Training Health Monitor - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)

        # Health status
        status_color = {
            "HEALTHY": "\033[92m",  # Green
            "WARNING": "\033[93m",  # Yellow
            "UNHEALTHY": "\033[91m",  # Red
            "CRITICAL": "\033[91m\033[1m",  # Bold Red
        }
        reset_color = "\033[0m"

        color = status_color.get(self.health_status, "")
        print(f"\nHealth Status: {color}{self.health_status}{reset_color}")

        # Statistics
        stats = self.get_statistics()
        if stats:
            print("\nCurrent Metrics:")
            if 'loss' in stats:
                print(f"  Loss:       {stats['loss']['current']:.4f} (avg: {stats['loss']['mean']:.4f})")
            if 'accuracy' in stats:
                print(f"  Accuracy:   {stats['accuracy']['current']:.4f} (avg: {stats['accuracy']['mean']:.4f})")
            if 'grad_norm' in stats:
                print(f"  Grad Norm:  {stats['grad_norm']['current']:.4f} (avg: {stats['grad_norm']['mean']:.4f})")
            if 'throughput' in stats:
                print(f"  Throughput: {stats['throughput']['current']:.2f} samples/sec (avg: {stats['throughput']['mean']:.2f})")

        # Issue counts
        print(f"\nIssue Counts:")
        print(f"  NaN losses:  {self.nan_count}")
        print(f"  OOM errors:  {self.oom_count}")
        print(f"  Errors:      {self.error_count}")
        print(f"  Warnings:    {self.warning_count}")

        # Recent issues
        if self.issues:
            print(f"\nRecent Issues (last 5):")
            for issue in self.issues[-5:]:
                severity_color = {
                    "CRITICAL": "\033[91m\033[1m",
                    "ERROR": "\033[91m",
                    "WARNING": "\033[93m",
                }
                color = severity_color.get(issue['severity'], "")
                print(f"  {color}[{issue['severity']}]{reset_color} {issue['message'][:80]}")

        # Recommendations
        recommendations = self.get_recommendations()
        if recommendations:
            print(f"\nRecommendations:")
            for rec in recommendations:
                print(f"  • {rec}")

        print("="*70 + "\n")

    def get_recommendations(self):
        """Get recommendations based on current state."""
        recs = []

        if self.nan_count > 0:
            recs.append("Reduce learning rate (LEARNING_RATE=3e-5)")
            recs.append("Reduce gradient clip norm (GRAD_CLIP_NORM=0.3)")

        if self.oom_count > 0:
            recs.append("Reduce batch size (L2T_TRAIN_BS=2)")
            recs.append("Enable gradient checkpointing")

        if self.grad_norms and max(self.grad_norms) > 5.0:
            recs.append("Reduce gradient clip norm")

        if self.losses and len(self.losses) > 50:
            recent = list(self.losses)[-50:]
            if max(recent) - min(recent) < 0.01:
                recs.append("Loss plateaued - consider increasing learning rate")

        if self.throughputs and len(self.throughputs) > 20:
            avg = sum(self.throughputs) / len(self.throughputs)
            if avg < 2.0:
                recs.append("Low throughput - check data loading (reduce num_workers)")

        return recs

    def monitor(self, interval=10):
        """Monitor log file continuously."""
        print(f"Monitoring: {self.log_file}")
        print(f"Update interval: {interval} seconds")
        print("Press Ctrl+C to stop\n")

        last_pos = 0

        try:
            while True:
                if self.log_file.exists():
                    with open(self.log_file, 'r') as f:
                        f.seek(last_pos)
                        new_lines = f.readlines()
                        last_pos = f.tell()

                        for line in new_lines:
                            self.parse_line(line)

                        if new_lines:
                            # Check health
                            self.check_loss_convergence()
                            self.check_throughput()

                            # Print status
                            self.print_status()

                time.sleep(interval)

        except KeyboardInterrupt:
            print("\n\nMonitoring stopped by user")
            self.print_final_report()

    def print_final_report(self):
        """Print final report."""
        print("\n" + "="*70)
        print("FINAL TRAINING HEALTH REPORT")
        print("="*70)

        stats = self.get_statistics()

        print("\nOverall Statistics:")
        if 'loss' in stats:
            print(f"  Loss:       {stats['loss']['min']:.4f} → {stats['loss']['current']:.4f}")
            improvement = (stats['loss']['max'] - stats['loss']['current']) / stats['loss']['max'] * 100
            print(f"  Improvement: {improvement:.1f}%")

        if 'accuracy' in stats:
            print(f"  Accuracy:   {stats['accuracy']['mean']:.4f}")

        print(f"\nTotal Issues:")
        print(f"  NaN losses:  {self.nan_count}")
        print(f"  OOM errors:  {self.oom_count}")
        print(f"  Errors:      {self.error_count}")
        print(f"  Warnings:    {self.warning_count}")

        print(f"\nFinal Health Status: {self.health_status}")

        # Save report
        report_file = f"health_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report = {
            'health_status': self.health_status,
            'statistics': stats,
            'issues': self.issues,
            'counts': {
                'nan': self.nan_count,
                'oom': self.oom_count,
                'errors': self.error_count,
                'warnings': self.warning_count,
            }
        }

        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\nReport saved to: {report_file}")
        print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Monitor training health in real-time')
    parser.add_argument('log_file', help='Path to training log file')
    parser.add_argument('--interval', type=int, default=10, help='Update interval in seconds')
    parser.add_argument('--window', type=int, default=100, help='Window size for metrics')

    args = parser.parse_args()

    monitor = TrainingHealthMonitor(args.log_file, window_size=args.window)
    monitor.monitor(interval=args.interval)


if __name__ == '__main__':
    main()
