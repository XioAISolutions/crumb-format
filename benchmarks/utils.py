"""Utility functions for benchmarking suite.

Provides common functionality for all benchmark modules including:
- Result data structures
- Configuration management
- Reproducibility (seeding)
- Formatting and reporting
- Statistical analysis
- File I/O
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    
    name: str
    model_name: str
    metric: str
    value: float
    unit: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> BenchmarkResult:
        """Create from dictionary."""
        return cls(**data)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking runs."""
    
    # Model settings
    model_path: Optional[str] = None
    model_type: str = "wavefield"  # wavefield, transformer, llama, mistral
    
    # Benchmark settings
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: str = "float32"  # float32, float16, bfloat16
    
    # Dataset settings
    dataset_name: Optional[str] = None
    dataset_path: Optional[str] = None
    max_samples: Optional[int] = None
    
    # Performance settings
    batch_size: int = 1
    sequence_lengths: List[int] = field(default_factory=lambda: [512, 1024, 2048])
    num_warmup: int = 10
    num_iterations: int = 100
    
    # Output settings
    output_dir: str = "benchmark_results"
    save_plots: bool = True
    save_json: bool = True
    save_csv: bool = True
    verbose: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> BenchmarkConfig:
        """Create from dictionary."""
        return cls(**data)


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Make CUDA operations deterministic
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def format_number(value: float, precision: int = 2) -> str:
    """Format number with appropriate units.
    
    Args:
        value: Number to format
        precision: Decimal places
        
    Returns:
        Formatted string with units (K, M, B, etc.)
    """
    if abs(value) >= 1e9:
        return f"{value / 1e9:.{precision}f}B"
    elif abs(value) >= 1e6:
        return f"{value / 1e6:.{precision}f}M"
    elif abs(value) >= 1e3:
        return f"{value / 1e3:.{precision}f}K"
    else:
        return f"{value:.{precision}f}"


def format_time(seconds: float) -> str:
    """Format time duration.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted string (e.g., "1.5s", "250ms", "10.2μs")
    """
    if seconds >= 1.0:
        return f"{seconds:.2f}s"
    elif seconds >= 1e-3:
        return f"{seconds * 1e3:.2f}ms"
    elif seconds >= 1e-6:
        return f"{seconds * 1e6:.2f}μs"
    else:
        return f"{seconds * 1e9:.2f}ns"


def format_memory(bytes_value: float) -> str:
    """Format memory size.
    
    Args:
        bytes_value: Memory in bytes
        
    Returns:
        Formatted string (e.g., "1.5GB", "250MB")
    """
    if bytes_value >= 1024**3:
        return f"{bytes_value / 1024**3:.2f}GB"
    elif bytes_value >= 1024**2:
        return f"{bytes_value / 1024**2:.2f}MB"
    elif bytes_value >= 1024:
        return f"{bytes_value / 1024:.2f}KB"
    else:
        return f"{bytes_value:.0f}B"


def save_results(
    results: List[BenchmarkResult],
    output_dir: Union[str, Path],
    name: str = "results",
    formats: List[str] = ["json", "csv"]
) -> None:
    """Save benchmark results to file.
    
    Args:
        results: List of benchmark results
        output_dir: Output directory
        name: Base filename
        formats: Output formats (json, csv)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if "json" in formats:
        json_path = output_dir / f"{name}.json"
        with open(json_path, "w") as f:
            json.dump([r.to_dict() for r in results], f, indent=2)
    
    if "csv" in formats:
        csv_path = output_dir / f"{name}.csv"
        with open(csv_path, "w") as f:
            if results:
                # Write header
                keys = results[0].to_dict().keys()
                f.write(",".join(keys) + "\n")
                # Write rows
                for result in results:
                    values = [str(v) for v in result.to_dict().values()]
                    f.write(",".join(values) + "\n")


def load_results(path: Union[str, Path]) -> List[BenchmarkResult]:
    """Load benchmark results from JSON file.
    
    Args:
        path: Path to JSON file
        
    Returns:
        List of benchmark results
    """
    with open(path, "r") as f:
        data = json.load(f)
    return [BenchmarkResult.from_dict(d) for d in data]


def compute_statistics(values: List[float]) -> Dict[str, float]:
    """Compute statistical measures.
    
    Args:
        values: List of measurements
        
    Returns:
        Dictionary with mean, std, min, max, median, p95, p99
    """
    arr = np.array(values)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }


def compute_speedup(baseline: float, optimized: float) -> float:
    """Compute speedup ratio.
    
    Args:
        baseline: Baseline measurement
        optimized: Optimized measurement
        
    Returns:
        Speedup factor (baseline / optimized)
    """
    if optimized == 0:
        return float('inf')
    return baseline / optimized


def compute_memory_reduction(baseline: float, optimized: float) -> float:
    """Compute memory reduction percentage.
    
    Args:
        baseline: Baseline memory usage
        optimized: Optimized memory usage
        
    Returns:
        Reduction percentage
    """
    if baseline == 0:
        return 0.0
    return ((baseline - optimized) / baseline) * 100


def get_device_info() -> Dict[str, Any]:
    """Get device information for reproducibility.
    
    Returns:
        Dictionary with device details
    """
    info = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    
    if torch.cuda.is_available():
        info.update({
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "device_name": torch.cuda.get_device_name(0),
            "device_count": torch.cuda.device_count(),
            "device_capability": torch.cuda.get_device_capability(0),
        })
    
    return info


def create_markdown_table(
    headers: List[str],
    rows: List[List[str]],
    alignments: Optional[List[str]] = None
) -> str:
    """Create a markdown table.
    
    Args:
        headers: Column headers
        rows: Table rows
        alignments: Column alignments ('left', 'center', 'right')
        
    Returns:
        Markdown table string
    """
    if alignments is None:
        alignments = ['left'] * len(headers)
    
    # Create header
    table = "| " + " | ".join(headers) + " |\n"
    
    # Create separator
    sep_parts = []
    for align in alignments:
        if align == 'center':
            sep_parts.append(":---:")
        elif align == 'right':
            sep_parts.append("---:")
        else:
            sep_parts.append(":---")
    table += "| " + " | ".join(sep_parts) + " |\n"
    
    # Create rows
    for row in rows:
        table += "| " + " | ".join(str(cell) for cell in row) + " |\n"
    
    return table


class ProgressBar:
    """Simple progress bar for benchmarking."""
    
    def __init__(self, total: int, desc: str = "", width: int = 50):
        self.total = total
        self.desc = desc
        self.width = width
        self.current = 0
        self.start_time = time.time()
    
    def update(self, n: int = 1) -> None:
        """Update progress."""
        self.current += n
        self._display()
    
    def _display(self) -> None:
        """Display progress bar."""
        if self.total == 0:
            return
        
        progress = self.current / self.total
        filled = int(self.width * progress)
        bar = "█" * filled + "░" * (self.width - filled)
        
        elapsed = time.time() - self.start_time
        if self.current > 0:
            eta = elapsed * (self.total - self.current) / self.current
            eta_str = format_time(eta)
        else:
            eta_str = "?"
        
        print(f"\r{self.desc} |{bar}| {self.current}/{self.total} "
              f"[{elapsed:.1f}s<{eta_str}]", end="", flush=True)
        
        if self.current >= self.total:
            print()  # New line when complete
    
    def close(self) -> None:
        """Close progress bar."""
        if self.current < self.total:
            self.current = self.total
            self._display()

# Made with Bob
