#!/usr/bin/env python3
"""
Data Analysis Script for Seismic Sensor CNN Quantization Project
Analyzes train/val index files and sample data structure
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple, Dict
from collections import Counter, defaultdict
import re
import numpy as np

try:
    import torch
except ImportError as e:
    print("Error: PyTorch is required to analyze .pt files")
    print("Install with: pip install torch")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
    PLOTTING_AVAILABLE = True
except ImportError:
    print("Warning: matplotlib not available, plotting disabled")
    PLOTTING_AVAILABLE = False


def read_list_file(file_path: str) -> List[str]:
    """Read a text file where each line is an item."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Index file not found: {path}")
    lines = [line.strip() for line in path.read_text().splitlines()]
    return [line for line in lines if line]


def is_single_vehicle_item(item: str, vehicle_names: List[str]) -> Tuple[bool, str]:
    """Check if item contains only one vehicle name and return the vehicle."""
    matched = [v for v in vehicle_names if re.search(rf"(?:^|[_\-]){re.escape(v)}(?:[_\-]|$)", item)]
    if len(matched) == 1:
        return True, matched[0]
    return False, ""


def is_background_item(item: str) -> bool:
    """Detect background samples identified by run14* or run15* in the filename."""
    name = Path(item).name  # operate on basename
    return bool(re.search(r"run1[45]", name, flags=re.IGNORECASE))


def classify_item(item: str, vehicle_names: List[str]) -> str:
    """Return the class label for an item: one of vehicle_names, 'background', or '' if mixed/unknown."""
    if is_background_item(item):
        return "background"
    is_single, vehicle = is_single_vehicle_item(item, vehicle_names)
    if is_single:
        return vehicle
    return ""


def filter_single_vehicle(items: List[str], vehicle_names: List[str]) -> List[str]:
    """Filter items to only include single-vehicle runs (excludes background)."""
    filtered: List[str] = []
    for item in items:
        label = classify_item(item, vehicle_names)
        if label in vehicle_names:
            filtered.append(item)
    return filtered


def count_by_label(items: List[str], vehicle_names: List[str], include_background: bool = True) -> Counter:
    """Count items by label (vehicles and optionally background)."""
    counts: Counter = Counter()
    for item in items:
        label = classify_item(item, vehicle_names)
        if label:
            counts[label] += 1
    # Ensure all labels appear even if zero
    for v in vehicle_names:
        counts.setdefault(v, 0)
    if include_background:
        counts.setdefault("background", 0)
    return counts


def print_vehicle_counts(counts: Counter, title: str) -> None:
    """Print vehicle counts in a formatted way."""
    print(f"\n{title}")
    print("-" * len(title))
    total = sum(counts.values())
    for vehicle, count in sorted(counts.items()):
        percentage = (count / total * 100) if total > 0 else 0
        print(f"{vehicle:>10}: {count:>4} ({percentage:>5.1f}%)")
    print(f"{'Total':>10}: {total:>4}")


def plot_vehicle_counts(train_counts: Counter, val_counts: Counter, vehicle_names: List[str]) -> None:
    """Create bar plots for counts, including background."""
    if not PLOTTING_AVAILABLE:
        return
    
    labels = vehicle_names + ["background"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Train counts
    train_values = [train_counts[lbl] for lbl in labels]
    ax1.bar(labels, train_values, color='skyblue', alpha=0.7)
    ax1.set_title('Train Set - Counts (vehicles + background)')
    ax1.set_ylabel('Count')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for i, v in enumerate(train_values):
        ax1.text(i, v + 0.1, str(v), ha='center', va='bottom')
    
    # Val counts
    val_values = [val_counts[lbl] for lbl in labels]
    ax2.bar(labels, val_values, color='lightcoral', alpha=0.7)
    ax2.set_title('Validation Set - Counts (vehicles + background)')
    ax2.set_ylabel('Count')
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for i, v in enumerate(val_values):
        ax2.text(i, v + 0.1, str(v), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('/home/misra8/CNN_Quantization_Tf_SeismicSensors/vehicle_counts.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved as: vehicle_counts.png")
    plt.show()


def compute_energy_from_sample(sample: dict) -> float:
    """Compute average energy from seismic data: mean of squared signal."""
    try:
        seismic = sample['data']['shake']['seismic']
    except Exception:
        return float('nan')
    # Convert to numpy array
    if hasattr(seismic, 'detach'):
        seismic = seismic.detach().cpu().numpy()
    elif hasattr(seismic, 'numpy'):
        seismic = seismic.numpy()
    seismic = np.asarray(seismic).astype(np.float64)
    # Flatten and compute mean square
    seismic = seismic.reshape(-1)
    if seismic.size == 0:
        return float('nan')
    return float(np.mean(seismic * seismic))


def extract_distance_meters(sample: dict, vehicle_names: List[str]) -> float:
    """Extract distance in meters for vehicle samples; returns np.nan for background/missing."""
    # distance structure: {'nissan': 93xx}
    try:
        dist_dict = sample.get('distance', {})
        if not isinstance(dist_dict, dict) or not dist_dict:
            return float('nan')
        # Prefer the single vehicle present in label
        labels = sample.get('label', set())
        vehicle_label = None
        if isinstance(labels, set) and len(labels) == 1:
            (vehicle_label,) = tuple(labels)
        # Fallback: pick first known vehicle key
        key = None
        if vehicle_label in dist_dict:
            key = vehicle_label
        else:
            for v in vehicle_names:
                if v in dist_dict:
                    key = v
                    break
        if key is None:
            return float('nan')
        distance_val = dist_dict[key]
        # distance may be int-like; ensure float meters
        return float(distance_val)
    except Exception:
        return float('nan')


def filter_samples_by_max_distance(items: List[str], vehicle_names: List[str], max_distance_m: float) -> List[str]:
    """Filter samples to only include those with distance <= max_distance_m (or background samples)."""
    filtered = []
    for path in items:
        try:
            sample = torch.load(path, map_location='cpu', weights_only=False)
        except Exception:
            continue
        
        label = classify_item(path, vehicle_names)
        # Keep background samples (no distance constraint)
        if label == 'background' or not sample.get('label'):
            filtered.append(path)
            continue
        
        # For vehicle samples, check distance
        distance_m = extract_distance_meters(sample, vehicle_names)
        if np.isfinite(distance_m) and distance_m <= max_distance_m:
            filtered.append(path)
    return filtered


def analyze_energy_vs_distance(items: List[str], vehicle_names: List[str], bin_size_m: float = 1.0) -> Dict[str, Dict[int, Tuple[float, int]]]:
    """Compute per-vehicle energy aggregated by distance bins; also compute background stats.

    Returns mapping from label -> bin_index -> (sum_energy, count). 'background' has a single key -1.
    """
    per_label_bins: Dict[str, Dict[int, Tuple[float, int]]] = defaultdict(lambda: defaultdict(lambda: (0.0, 0)))
    all_energies: list = []
    background_energies: list = []

    for path in items:
        try:
            sample = torch.load(path, map_location='cpu', weights_only=False)
        except Exception:
            continue
        label = classify_item(path, vehicle_names)
        energy = compute_energy_from_sample(sample)
        if not np.isfinite(energy):
            continue
        all_energies.append(energy)
        if label == 'background' or not sample.get('label'):
            background_energies.append(energy)
            # store under bin -1 for background (no distance)
            total, cnt = per_label_bins['background'][-1]
            per_label_bins['background'][-1] = (total + energy, cnt + 1)
            continue
        # vehicle: need distance
        distance_m = extract_distance_meters(sample, vehicle_names)
        if not np.isfinite(distance_m):
            continue
        bin_index = int(np.floor(distance_m / bin_size_m))
        total, cnt = per_label_bins[label][bin_index]
        per_label_bins[label][bin_index] = (total + energy, cnt + 1)

    # Attach summary stats
    per_label_bins['__summary__'] = {
        -1: (
            float(np.nanmean(background_energies) if background_energies else np.nan),
            len(background_energies)
        ),
        -2: (
            float(np.percentile(all_energies, 25)) if all_energies else float('nan'),
            0
        ),
        -3: (
            float(np.percentile(all_energies, 50)) if all_energies else float('nan'),
            0
        ),
        -4: (
            float(np.percentile(all_energies, 75)) if all_energies else float('nan'),
            0
        ),
    }
    return per_label_bins


def plot_energy_vs_distance(per_label_bins: Dict[str, Dict[int, Tuple[float, int]]], vehicle_names: List[str], bin_size_m: float) -> None:
    if not PLOTTING_AVAILABLE:
        return
    plt.figure(figsize=(10, 6))
    colors = {
        'nissan': 'tab:blue',
        'lexus': 'tab:orange',
        'mazda': 'tab:green',
        'benz': 'tab:red'
    }

    # Plot per-vehicle curves
    for v in vehicle_names:
        bins = per_label_bins.get(v, {})
        if not bins:
            continue
        xs = []
        ys = []
        for b in sorted(k for k in bins.keys() if k >= 0):
            total, cnt = bins[b]
            if cnt > 0:
                xs.append((b + 0.5) * bin_size_m)  # bin center
                ys.append(total / cnt)
        if xs:
            plt.plot(xs, ys, label=v, color=colors.get(v, None))

    # Background mean (horizontal)
    bg_mean, bg_count = per_label_bins.get('__summary__', {}).get(-1, (np.nan, 0))
    if np.isfinite(bg_mean):
        plt.axhline(bg_mean, color='k', linestyle='--', label=f'background mean (n={bg_count})')

    # Quantile lines (25, 50, 75)
    q25 = per_label_bins.get('__summary__', {}).get(-2, (np.nan, 0))[0]
    q50 = per_label_bins.get('__summary__', {}).get(-3, (np.nan, 0))[0]
    q75 = per_label_bins.get('__summary__', {}).get(-4, (np.nan, 0))[0]
    if np.isfinite(q25):
        plt.axhline(q25, color='gray', linestyle=':', label='25th percentile')
    if np.isfinite(q50):
        plt.axhline(q50, color='gray', linestyle='-.', label='median')
    if np.isfinite(q75):
        plt.axhline(q75, color='gray', linestyle='--', label='75th percentile')

    plt.xlabel('Distance (m)')
    plt.ylabel('Average energy (mean of squared seismic)')
    plt.title('Energy vs Distance (per 1m bin)')
    plt.legend()
    plt.tight_layout()
    out_path = '/home/misra8/CNN_Quantization_Tf_SeismicSensors/distance_energy.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved distance-energy plot to: {Path(out_path).name}")


def analyze_sample_file(sample_file_path: str) -> None:
    """Analyze the structure of a sample .pt file."""
    print(f"\nAnalyzing sample file: {sample_file_path}")
    print("=" * 60)
    
    if not Path(sample_file_path).exists():
        print(f"Error: Sample file not found: {sample_file_path}")
        return
    
    try:
        # Load with weights_only=False for PyTorch 2.6+ compatibility
        sample = torch.load(sample_file_path, map_location="cpu", weights_only=False)
        
        print(f"Loaded object type: {type(sample)}")
        
        if isinstance(sample, dict):
            print(f"Top-level keys: {list(sample.keys())}")
            print("\nDetailed structure:")
            
            for k, v in sample.items():
                if isinstance(v, dict):
                    sub_keys = list(v.keys())[:10]  # Show first 10 keys
                    print(f"  {k}: dict with {len(v)} keys")
                    if sub_keys:
                        print(f"    Sample keys: {sub_keys}")
                elif hasattr(v, 'shape'):
                    print(f"  {k}: tensor/array with shape {v.shape}, dtype {getattr(v, 'dtype', 'unknown')}")
                elif isinstance(v, (list, tuple)):
                    print(f"  {k}: {type(v).__name__} with {len(v)} items")
                    if v and hasattr(v[0], 'shape'):
                        print(f"    First item shape: {v[0].shape}")
                else:
                    print(f"  {k}: {type(v)}")
        else:
            print(f"Non-dict object: {type(sample)}")
            if hasattr(sample, 'shape'):
                print(f"Shape: {sample.shape}")
                
    except Exception as e:
        print(f"Error loading sample file: {e}")


def main():
    """Main analysis function."""
    print("Seismic Sensor CNN Quantization - Data Analysis")
    print("=" * 50)
    
    # File paths
    train_index_file = "/home/tkimura4/data/datasets/2025-09-21-ICT/train_index.txt"
    val_index_file = "/home/tkimura4/data/datasets/2025-09-21-ICT/val_index.txt"
    sample_file = "/home/tkimura4/data/datasets/2025-09-21-ICT/individual_time_samples/run6_rs5_331_nissan_93.pt"
    
    vehicle_names = ['nissan', 'lexus', 'mazda', 'benz']
    
    # Get max distance filter from environment variable
    try:
        max_dist_env = os.getenv('MAX_DISTANCE_M')
        max_distance_m = float(max_dist_env) if max_dist_env else None
    except Exception:
        max_distance_m = None
    
    try:
        # Load index files
        print("Loading index files...")
        train_items = read_list_file(train_index_file)
        val_items = read_list_file(val_index_file)
        
        print(f"Total train items: {len(train_items)}")
        print(f"Total val items: {len(val_items)}")
        
        # Apply max distance filter if specified
        if max_distance_m is not None:
            print(f"\nFiltering by max distance: {max_distance_m} m")
            train_items = filter_samples_by_max_distance(train_items, vehicle_names, max_distance_m)
            val_items = filter_samples_by_max_distance(val_items, vehicle_names, max_distance_m)
            print(f"After distance filter - train items: {len(train_items)}")
            print(f"After distance filter - val items: {len(val_items)}")
        
        # Filter for single-vehicle items
        print("\nFiltering for single-vehicle items...")
        train_single = filter_single_vehicle(train_items, vehicle_names)
        val_single = filter_single_vehicle(val_items, vehicle_names)
        
        print(f"Single-vehicle train items: {len(train_single)}")
        print(f"Single-vehicle val items: {len(val_single)}")
        
        # Count by label (vehicles + background)
        train_counts = count_by_label(train_items, vehicle_names, include_background=True)
        val_counts = count_by_label(val_items, vehicle_names, include_background=True)
        
        # Print results
        print_vehicle_counts(train_counts, "Train Set - Counts (vehicles + background)")
        print_vehicle_counts(val_counts, "Validation Set - Counts (vehicles + background)")
        
        # Create plots
        if PLOTTING_AVAILABLE:
            plot_vehicle_counts(train_counts, val_counts, vehicle_names)
        
        # Analyze sample file
        analyze_sample_file(sample_file)

        # Energy vs distance analysis (train + val)
        try:
            bin_size_env = os.getenv('BIN_SIZE_M')
            bin_size_m = float(bin_size_env) if bin_size_env else 1.0
        except Exception:
            bin_size_m = 1.0
        print(f"\nComputing energy vs distance with bin size: {bin_size_m} m")
        combined_items = train_items + val_items
        per_label_bins = analyze_energy_vs_distance(combined_items, vehicle_names, bin_size_m=bin_size_m)
        plot_energy_vs_distance(per_label_bins, vehicle_names, bin_size_m)
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please check that the data files exist at the specified paths.")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
