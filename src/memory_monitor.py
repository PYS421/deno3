import csv
import json
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

import torch


class MemoryMonitor:
    """
    独立 GPU 显存监控器。

    PyTorch 指标：
        allocated / reserved / peak_allocated / peak_reserved

    nvidia-smi 指标：
        GPU 整体已用显存 / 总显存 / GPU 利用率

    输出：
        memory_log.csv
        memory_summary.json
    """

    def __init__(
        self,
        output_dir="monitoring/memory",
        interval=1.0,
        device=0,
        swanlab_run=None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.interval = float(interval)
        self.device = int(device)
        self.swanlab_run = swanlab_run

        self.csv_path = self.output_dir / "memory_log.csv"
        self.summary_path = self.output_dir / "memory_summary.json"

        self._stop_event = threading.Event()
        self._thread = None
        self._lock = threading.Lock()

        self._start_time = None
        self._running = False

        self.current_stage = "initialization"
        self.current_epoch = 0
        self.global_step = 0

        self.stage_stats = {}
        self.overall_stats = {
            "torch_peak_allocated_gb": 0.0,
            "torch_peak_reserved_gb": 0.0,
            "gpu_peak_used_gb": 0.0,
        }

        self._csv_file = None
        self._csv_writer = None

        self._csv_fields = [
            "timestamp",
            "elapsed_seconds",
            "stage",
            "epoch",
            "global_step",
            "torch_allocated_gb",
            "torch_reserved_gb",
            "torch_peak_allocated_gb",
            "torch_peak_reserved_gb",
            "gpu_used_gb",
            "gpu_total_gb",
            "gpu_utilization_pct",
        ]

    @staticmethod
    def _gb(value):
        return round(value / (1024 ** 3), 4)

    def _get_torch_memory(self):
        if not torch.cuda.is_available():
            return {}

        allocated = torch.cuda.memory_allocated(self.device)
        reserved = torch.cuda.memory_reserved(self.device)
        peak_allocated = torch.cuda.max_memory_allocated(self.device)
        peak_reserved = torch.cuda.max_memory_reserved(self.device)

        return {
            "torch_allocated_gb": self._gb(allocated),
            "torch_reserved_gb": self._gb(reserved),
            "torch_peak_allocated_gb": self._gb(peak_allocated),
            "torch_peak_reserved_gb": self._gb(peak_reserved),
        }

    def _get_gpu_memory(self):
        """
        使用 nvidia-smi 获取 GPU 整体状态。
        不依赖 pynvml。
        """
        command = [
            "nvidia-smi",
            f"--id={self.device}",
            "--query-gpu=memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]

        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=3,
                check=True,
            )

            values = result.stdout.strip().split(",")

            if len(values) != 3:
                return {}

            used_mb, total_mb, utilization = [
                float(value.strip()) for value in values
            ]

            return {
                "gpu_used_gb": round(used_mb / 1024, 4),
                "gpu_total_gb": round(total_mb / 1024, 4),
                "gpu_utilization_pct": utilization,
            }

        except Exception:
            return {}

    def _collect(self):
        metrics = {}
        metrics.update(self._get_torch_memory())
        metrics.update(self._get_gpu_memory())
        return metrics

    def _write_csv(self, metrics):
        if self._csv_writer is None:
            return

        elapsed = time.monotonic() - self._start_time

        row = {
            "timestamp": datetime.now().astimezone().isoformat(
                timespec="seconds"
            ),
            "elapsed_seconds": round(elapsed, 2),
            "stage": self.current_stage,
            "epoch": self.current_epoch,
            "global_step": self.global_step,
        }

        row.update(metrics)
        self._csv_writer.writerow(row)
        self._csv_file.flush()

    def _update_stats(self, metrics):
        stage = self.current_stage

        if stage not in self.stage_stats:
            self.stage_stats[stage] = {
                "samples": 0,
                "torch_peak_allocated_gb": 0.0,
                "torch_peak_reserved_gb": 0.0,
                "gpu_peak_used_gb": 0.0,
            }

        stats = self.stage_stats[stage]
        stats["samples"] += 1

        for key in (
            "torch_peak_allocated_gb",
            "torch_peak_reserved_gb",
        ):
            value = metrics.get(key, 0.0)
            stats[key] = max(stats[key], value)
            self.overall_stats[key] = max(
                self.overall_stats[key], value
            )

        gpu_used = metrics.get("gpu_used_gb", 0.0)

        stats["gpu_peak_used_gb"] = max(
            stats["gpu_peak_used_gb"], gpu_used
        )
        self.overall_stats["gpu_peak_used_gb"] = max(
            self.overall_stats["gpu_peak_used_gb"], gpu_used
        )

    def _sample_once(self):
        metrics = self._collect()

        with self._lock:
            self._write_csv(metrics)
            self._update_stats(metrics)

    def _sampling_loop(self):
        while not self._stop_event.is_set():
            try:
                self._sample_once()
            except Exception as exc:
                print(f"[MemoryMonitor] 采样失败：{exc}")

            self._stop_event.wait(self.interval)

    def start(self):
        if self._running:
            return

        if not torch.cuda.is_available():
            raise RuntimeError("MemoryMonitor 未检测到 CUDA GPU。")

        self._start_time = time.monotonic()
        self._stop_event.clear()

        self._csv_file = self.csv_path.open(
            "w",
            newline="",
            encoding="utf-8",
        )
        self._csv_writer = csv.DictWriter(
            self._csv_file,
            fieldnames=self._csv_fields,
        )
        self._csv_writer.writeheader()

        torch.cuda.synchronize(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)

        self._running = True

        self._thread = threading.Thread(
            target=self._sampling_loop,
            name="GPU-Memory-Monitor",
            daemon=True,
        )
        self._thread.start()

        print("[MemoryMonitor] 显存监控已启动")
        print(f"[MemoryMonitor] CSV：{self.csv_path}")

    def mark(self, stage, epoch=0, global_step=None):
        """
        切换监控阶段。

        先记录上一阶段的峰值，再重置 PyTorch 峰值计数器，
        避免把训练阶段的峰值错误归入验证阶段。
        """
        if not self._running:
            return

        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device)

        # 先将上一阶段最后的峰值记录下来
        self._sample_once()

        with self._lock:
            self.current_stage = str(stage)
            self.current_epoch = int(epoch)

            if global_step is not None:
                self.global_step = int(global_step)

            if self.current_stage not in self.stage_stats:
                self.stage_stats[self.current_stage] = {
                    "samples": 0,
                    "torch_peak_allocated_gb": 0.0,
                    "torch_peak_reserved_gb": 0.0,
                    "gpu_peak_used_gb": 0.0,
                }

        # 新阶段单独统计峰值
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)

        self._sample_once()

        print(
            f"[MemoryMonitor] 阶段切换：{self.current_stage}, "
            f"epoch={self.current_epoch}, "
            f"global_step={self.global_step}"
        )

    def stop(self):
        if not self._running:
            return

        self._stop_event.set()

        if self._thread is not None:
            self._thread.join(timeout=5)

        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device)

        try:
            self._sample_once()
        except Exception as exc:
            print(f"[MemoryMonitor] 最终采样失败：{exc}")

        elapsed = time.monotonic() - self._start_time

        summary = {
            "duration_seconds": round(elapsed, 2),
            "duration_minutes": round(elapsed / 60, 2),
            "device": self.device,
            "gpu_name": torch.cuda.get_device_name(self.device),
            "sampling_interval_seconds": self.interval,
            "overall_peak": self.overall_stats,
            "stage_statistics": self.stage_stats,
            "csv_file": str(self.csv_path),
        }

        with self.summary_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                summary,
                file,
                ensure_ascii=False,
                indent=2,
            )

        if self.swanlab_run is not None:
            try:
                self.swanlab_run.log({
                    "memory/overall_peak_allocated_gb":
                        self.overall_stats["torch_peak_allocated_gb"],
                    "memory/overall_peak_reserved_gb":
                        self.overall_stats["torch_peak_reserved_gb"],
                    "memory/overall_gpu_peak_used_gb":
                        self.overall_stats["gpu_peak_used_gb"],
                })
            except Exception as exc:
                print(f"[MemoryMonitor] SwanLab 显存记录失败：{exc}")

        if self._csv_file is not None:
            self._csv_file.close()

        self._running = False

        print("[MemoryMonitor] 显存监控已停止")
        print(f"[MemoryMonitor] CSV：{self.csv_path}")
        print(f"[MemoryMonitor] 汇总：{self.summary_path}")
        print(
            "[MemoryMonitor] PyTorch 峰值分配显存："
            f"{self.overall_stats['torch_peak_allocated_gb']:.2f} GB"
        )
        print(
            "[MemoryMonitor] GPU 整体峰值占用："
            f"{self.overall_stats['gpu_peak_used_gb']:.2f} GB"
        )
