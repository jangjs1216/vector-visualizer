#!/usr/bin/env python3
"""
Vector Distance Simulator — Dual-view streaming dataset simulation.

Based on pca_vector_viewer.py, this tool adds:
  - Left view: Full dataset scatter plot (existing viewer)
  - Right view: Compare multiple simulated datasets side-by-side
  - Streaming simulation: sequentially process data points, deciding
    whether to accept/reject based on distance-based strategies
  - Multiple simulation datasets with different parameters

Usage:
    python vector_distance_simulator.py --csv /path/to/vectors.csv
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
import threading
import tkinter as tk
from dataclasses import dataclass, field
from queue import Empty, Queue
from tkinter import filedialog, messagebox, ttk
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cosine as cosine_distance, cdist

try:
    from PIL import Image, ImageTk
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

META_COLUMNS = ["type", "side", "color", "path"]
PXPY_MISSING = "(missing)"

# Strategy parameter definitions
STRATEGIES = {
    "Min Distance": {
        "desc": "기존 데이터셋 내 가장 가까운 점과의 cosine distance가\nthreshold 이상이면 추가",
        "params": {
            "min_dist_threshold": {"label": "Min Distance Threshold", "default": 0.05, "min": 0.001, "max": 2.0},
        },
    },
    "KNN Density": {
        "desc": "K개 최근접 이웃의 평균 거리가 threshold 이상이면 추가\n(밀집 영역 제거)",
        "params": {
            "k": {"label": "K (neighbors)", "default": 5, "min": 1, "max": 50},
            "density_threshold": {"label": "Density Threshold", "default": 0.05, "min": 0.001, "max": 2.0},
        },
    },
    "Adaptive Threshold": {
        "desc": "현재 데이터셋의 평균 거리를 기반으로\n동적 임계값을 자동 조절",
        "params": {
            "base_threshold": {"label": "Base Threshold", "default": 0.05, "min": 0.001, "max": 2.0},
            "adaptation_rate": {"label": "Adaptation Rate", "default": 1.0, "min": 0.1, "max": 2.0},
        },
    },
}


# ─────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Vector Distance Simulator with dual-view comparison."
    )
    parser.add_argument("--csv", type=str, default="", help="Path to input CSV file.")
    parser.add_argument("--point-size", type=float, default=26.0, help="Scatter point size.")
    parser.add_argument("--alpha", type=float, default=0.8, help="Scatter point alpha.")
    parser.add_argument("--no-standardize", action="store_true", help="Disable standardization.")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────
# CSV scanning & loading
# ─────────────────────────────────────────────────────────────

def _normalize_pxpy_value(value: str | None) -> str:
    text = "" if value is None else str(value).strip()
    return text or PXPY_MISSING


def _extract_filename_from_path(path_value: str) -> str:
    normalized = str(path_value).strip().replace("\\", "/")
    return normalized.rsplit("/", 1)[-1]


def extract_px_py_from_path(path_value: str) -> tuple[str, str]:
    filename = _extract_filename_from_path(path_value)
    tokens = re.findall(r"\[([^\[\]]*)\]", filename)
    if len(tokens) < 4:
        return PXPY_MISSING, PXPY_MISSING
    return _normalize_pxpy_value(tokens[2]), _normalize_pxpy_value(tokens[3])


def add_px_py_columns(df: pd.DataFrame) -> pd.DataFrame:
    px_py = df["path"].astype(str).map(extract_px_py_from_path)
    df = df.copy()
    df["px"] = [px for px, _ in px_py]
    df["py"] = [py for _, py in px_py]
    return df


def quick_scan_csv(csv_path: str) -> tuple[int, list[str]]:
    """Quickly scan to get row count and unique side values."""
    t0 = time.time()
    print("Scanning CSV metadata...", flush=True)

    with open(csv_path, "r", encoding="utf-8") as f:
        header_line = f.readline().strip()
    header_fields = header_line.split(",")
    side_idx = header_fields.index("side") if "side" in header_fields else 1

    sides: set[str] = set()
    row_count = 0
    with open(csv_path, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.split(",", side_idx + 2)
            if len(parts) > side_idx:
                sv = parts[side_idx].strip()
                if sv:
                    sides.add(sv)
            row_count += 1

    print(f"  Scanned {row_count:,} rows in {time.time() - t0:.1f}s", flush=True)
    return row_count, sorted(sides)


def ask_user_options(csv_path: str) -> tuple[list[str] | None, float]:
    """Interactive prompt at startup."""
    row_count, sides = quick_scan_csv(csv_path)

    print(f"\n{'='*55}")
    print(f"  CSV  : {csv_path}")
    print(f"  Rows : {row_count:,}")
    print(f"  Sides: {sides}")
    print(f"{'='*55}\n")

    print("[1] Side filter")
    print(f"    Available sides: {', '.join(sides)}")
    print("    Enter side(s) separated by comma, or press Enter for ALL.")
    side_input = input("    > ").strip()

    side_filter: list[str] | None = None
    if side_input:
        selected = [s.strip() for s in side_input.split(",") if s.strip()]
        valid = [s for s in selected if s in sides]
        if valid:
            side_filter = valid
            print(f"    → Selected: {side_filter}")
        else:
            print("    → No valid sides matched, using ALL.")

    print(f"\n[2] Downsampling")
    print("    Enter a ratio (e.g. 0.1 = 10%, 0.5 = 50%), or press Enter for 100%.")
    ratio_input = input("    > ").strip()

    sample_ratio = 1.0
    if ratio_input:
        try:
            sample_ratio = float(ratio_input)
            sample_ratio = max(0.01, min(1.0, sample_ratio))
            print(f"    → Ratio: {sample_ratio:.0%} (~{int(row_count * sample_ratio):,} rows)")
        except ValueError:
            print("    → Invalid input, using 100%.")
            sample_ratio = 1.0

    print()
    return side_filter, sample_ratio


def load_csv(
    csv_path: str,
    side_filter: list[str] | None = None,
    sample_ratio: float = 1.0,
) -> tuple[pd.DataFrame, list[str], int]:
    """Load CSV, validate, filter, downsample. Returns (df, vector_columns, skipped)."""
    t0 = time.time()
    print("Loading CSV...", flush=True)

    with open(csv_path, "r", encoding="utf-8") as f:
        header_line = f.readline().strip()
    header_fields = header_line.split(",")

    vec_col_names = [f"v_{i}" for i in range(256)]
    dtype_dict = {col: str for col in META_COLUMNS}
    dtype_dict.update({col: np.float64 for col in vec_col_names})

    if len(header_fields) <= 5 and "vector" in header_fields:
        col_names = META_COLUMNS + vec_col_names
        df = pd.read_csv(
            csv_path, header=None, skiprows=1, names=col_names,
            on_bad_lines="skip", encoding="utf-8", dtype=dtype_dict,
        )
    else:
        df = pd.read_csv(csv_path, encoding="utf-8")

    t1 = time.time()
    print(f"  CSV read: {t1 - t0:.1f}s ({len(df):,} rows)", flush=True)

    vector_columns = [col for col in df.columns if col not in META_COLUMNS]
    if not vector_columns:
        raise ValueError("No vector columns found.")

    # Validate
    print("Validating rows...", flush=True)
    meta_mask = df[META_COLUMNS].notna().all(axis=1)
    vector_valid_mask = df[vector_columns].notna().all(axis=1)
    valid_mask = meta_mask & vector_valid_mask
    skipped = int((~valid_mask).sum())
    df = df[valid_mask].reset_index(drop=True)

    t2 = time.time()
    print(f"  Validation: {t2 - t1:.1f}s (valid: {len(df):,}, skipped: {skipped:,})", flush=True)

    df = add_px_py_columns(df)

    # Side filter
    if side_filter:
        before = len(df)
        df = df[df["side"].isin(side_filter)].reset_index(drop=True)
        print(f"  Side filter {side_filter}: {before:,} → {len(df):,} rows", flush=True)

    # Downsample
    if 0.0 < sample_ratio < 1.0:
        before = len(df)
        df = df.sample(frac=sample_ratio, random_state=42).reset_index(drop=True)
        print(f"  Downsampled: {before:,} → {len(df):,} rows ({sample_ratio:.0%})", flush=True)

    print(f"  Total load time: {time.time() - t0:.1f}s", flush=True)
    return df, vector_columns, skipped


# ─────────────────────────────────────────────────────────────
# Embedding computation
# ─────────────────────────────────────────────────────────────

def compute_embedding(
    df: pd.DataFrame,
    vector_columns: list[str],
    method: str = "PCA",
    n_dim: int = 3,
    standardize: bool = True,
) -> np.ndarray:
    """Compute 2D/3D embedding from raw vectors. Returns (N, n_dim) array."""
    t0 = time.time()
    x = df[vector_columns].to_numpy(dtype=np.float64)
    if standardize:
        x = StandardScaler().fit_transform(x)

    if method == "UMAP":
        if not HAS_UMAP:
            raise RuntimeError("umap-learn is not installed. Run: pip install umap-learn")
        reducer = umap.UMAP(n_components=n_dim, random_state=42, n_jobs=1)
        embedding = reducer.fit_transform(x)
    else:  # PCA
        solver = "randomized" if len(df) > 5000 else "full"
        embedding = PCA(
            n_components=n_dim, random_state=42, svd_solver=solver
        ).fit_transform(x)

    elapsed = time.time() - t0
    print(f"  {method} ({n_dim}D): {elapsed:.1f}s on {len(df):,} points", flush=True)
    return embedding


# ─────────────────────────────────────────────────────────────
# Palette
# ─────────────────────────────────────────────────────────────

def build_palette(labels: list[str]) -> dict[str, tuple[float, float, float, float]]:
    cmap = plt.get_cmap("tab20")
    unique_labels = list(dict.fromkeys(labels))
    return {label: cmap(idx % cmap.N) for idx, label in enumerate(unique_labels)}


# ─────────────────────────────────────────────────────────────
# Streaming Simulation Engine
# ─────────────────────────────────────────────────────────────

@dataclass
class SimulationResult:
    """Holds a simulation dataset result."""
    name: str
    strategy: str
    params: dict
    accepted_indices: list[int]  # indices into the source DataFrame
    total_processed: int
    total_accepted: int

    @property
    def acceptance_rate(self) -> float:
        return self.total_accepted / max(self.total_processed, 1)


class StreamingSimulator:
    """Simulates streaming data insertion with distance-based filtering."""

    @staticmethod
    def simulate(
        df: pd.DataFrame,
        vector_columns: list[str],
        strategy: str,
        params: dict,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> SimulationResult:
        """Run streaming simulation on the given dataframe.

        Data is processed row-by-row in order. For each row, the strategy
        decides whether to accept it into the dataset based on distance
        to already-accepted points.
        """
        vectors = df[vector_columns].to_numpy(dtype=np.float64)
        n = len(vectors)

        # Normalize vectors for cosine distance computation
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        normalized = vectors / norms

        accepted_indices: list[int] = []
        accepted_vecs: list[np.ndarray] = []

        for i in range(n):
            vec = normalized[i]

            if len(accepted_vecs) == 0:
                # Always accept the first point
                accepted_indices.append(i)
                accepted_vecs.append(vec)
            else:
                accept = StreamingSimulator._should_accept(
                    vec, accepted_vecs, strategy, params
                )
                if accept:
                    accepted_indices.append(i)
                    accepted_vecs.append(vec)

            # Progress callback
            if progress_callback and (i % 100 == 0 or i == n - 1):
                progress_callback(i + 1, n)

        # Build name
        param_str = ", ".join(f"{k}={v}" for k, v in params.items())
        short_strategy = strategy.replace(" ", "")
        name = f"{short_strategy}({param_str}) [{len(accepted_indices)}/{n}]"

        return SimulationResult(
            name=name,
            strategy=strategy,
            params=params,
            accepted_indices=accepted_indices,
            total_processed=n,
            total_accepted=len(accepted_indices),
        )

    @staticmethod
    def _should_accept(
        vec: np.ndarray,
        accepted_vecs: list[np.ndarray],
        strategy: str,
        params: dict,
    ) -> bool:
        """Determine whether to accept a new vector based on strategy."""
        if strategy == "Min Distance":
            return StreamingSimulator._strategy_min_distance(vec, accepted_vecs, params)
        elif strategy == "KNN Density":
            return StreamingSimulator._strategy_knn_density(vec, accepted_vecs, params)
        elif strategy == "Adaptive Threshold":
            return StreamingSimulator._strategy_adaptive(vec, accepted_vecs, params)
        return True

    @staticmethod
    def _cosine_distances(vec: np.ndarray, others: list[np.ndarray]) -> np.ndarray:
        """Compute cosine distances from vec to all vectors in others."""
        others_arr = np.array(others)
        # cosine distance = 1 - cosine_similarity
        similarities = others_arr @ vec
        return 1.0 - similarities

    @staticmethod
    def _strategy_min_distance(
        vec: np.ndarray, accepted_vecs: list[np.ndarray], params: dict
    ) -> bool:
        """Accept if the minimum cosine distance to any accepted point >= threshold."""
        threshold = params.get("min_dist_threshold", 0.05)
        distances = StreamingSimulator._cosine_distances(vec, accepted_vecs)
        min_dist = np.min(distances)
        return bool(min_dist >= threshold)

    @staticmethod
    def _strategy_knn_density(
        vec: np.ndarray, accepted_vecs: list[np.ndarray], params: dict
    ) -> bool:
        """Accept if the average distance to K nearest neighbors >= threshold."""
        k = int(params.get("k", 5))
        threshold = params.get("density_threshold", 0.05)

        distances = StreamingSimulator._cosine_distances(vec, accepted_vecs)

        # Use min(k, len(accepted)) neighbors
        actual_k = min(k, len(distances))
        nearest_k = np.partition(distances, actual_k - 1)[:actual_k]
        avg_dist = np.mean(nearest_k)

        return bool(avg_dist >= threshold)

    @staticmethod
    def _strategy_adaptive(
        vec: np.ndarray, accepted_vecs: list[np.ndarray], params: dict
    ) -> bool:
        """Accept with adaptive threshold based on current dataset density."""
        base_threshold = params.get("base_threshold", 0.05)
        adaptation_rate = params.get("adaptation_rate", 1.0)

        distances = StreamingSimulator._cosine_distances(vec, accepted_vecs)
        min_dist = np.min(distances)

        # Compute average pairwise distance of accepted set (sampled for efficiency)
        n_accepted = len(accepted_vecs)
        if n_accepted <= 1:
            return bool(min_dist >= base_threshold)

        # Sample up to 50 points for density estimation
        sample_size = min(50, n_accepted)
        if sample_size < n_accepted:
            sample_indices = np.random.choice(n_accepted, sample_size, replace=False)
            sample_vecs = [accepted_vecs[i] for i in sample_indices]
        else:
            sample_vecs = accepted_vecs

        sample_arr = np.array(sample_vecs)
        pairwise_sims = sample_arr @ sample_arr.T
        np.fill_diagonal(pairwise_sims, 1.0)
        pairwise_dists = 1.0 - pairwise_sims
        mean_pair_dist = np.mean(pairwise_dists[np.triu_indices(len(sample_arr), k=1)])

        # Adaptive threshold: higher density → stricter threshold
        adaptive_threshold = base_threshold * (1.0 + adaptation_rate * (1.0 - mean_pair_dist))

        return bool(min_dist >= adaptive_threshold)


BUFFER_STRATEGIES = {
    "FIFO Baseline": {
        "desc": "가장 단순한 기준선. 날짜 순으로 넣고, N을 넘으면 가장 오래된 샘플부터 제거합니다.",
    },
    "Distance Gate FIFO": {
        "desc": "현재 버퍼와 cosine distance가 충분히 멀 때만 넣는 FIFO입니다. 중복 억제는 잘하지만 Mother 비율 보장은 약합니다.",
    },
    "Quota-First FIFO": {
        "desc": "Mother 분포의 영역별 비율을 먼저 맞추고, 같은 영역 안의 과도한 중복만 distance threshold로 줄이는 추천 전략입니다.",
    },
    "Stable Coverage Memory": {
        "desc": "가장 가까운 기존 샘플과의 거리만으로 빈 공간을 채우는 안정형 메모리입니다. 이미 본 이미지는 유지하고, 비어 있는 영역만 추가로 채우는 현재 목적에 가장 가깝습니다.",
    },
    "Approx Density Memory": {
        "desc": "Mother를 coarse bucket으로 나눈 뒤 bucket별 개수를 밀도 근사로 사용합니다. bucket이 비었으면 채우고, 밀집 bucket은 비례해서 더 많이 유지하는 경량 밀도 추종 전략입니다.",
    },
}

TS_COMMON_PARAMS = {
    "buffer_size": {"label": "Buffer Size (N)", "default": 750, "min": 16, "max": 5000, "type": "int"},
    "max_age_days": {"label": "Max Age Days (K)", "default": 360, "min": 1, "max": 365, "type": "int"},
    "region_count": {"label": "Region Count", "default": 15, "min": 1, "max": 64, "type": "int"},
    "distance_threshold": {"label": "Cosine Dist Threshold", "default": 0.001, "min": 0.0, "max": 2.0, "type": "float"},
}

TS_METRIC_HELP = {
    "Tracking Score": "전략 순위를 매기는 종합 점수입니다. Match 55%, Coverage 20%, Fill 15%, Freshness 10%를 반영합니다. 높을수록 Mother 추종이 안정적입니다.",
    "Match": "현재 버퍼의 영역별 비율이 Mother 영역 비율과 얼마나 비슷한지 보여줍니다. 100이면 영역 비율이 거의 동일합니다.",
    "Coverage": "Mother가 차지하는 영역 중 현재 버퍼가 실제로 덮고 있는 비율입니다. 비어 있는 영역이 많아질수록 낮아집니다.",
    "Fill": "버퍼가 목표 크기 N에 대해 얼마나 차 있는지 보여줍니다. 너무 낮으면 실제 학습용 데이터셋이 충분히 구성되지 않은 상태입니다.",
    "Avg Age": "버퍼 안 샘플들의 평균 age입니다. 낮을수록 최근 데이터를 더 잘 반영합니다.",
    "Mean NN": "버퍼 내부 샘플들의 평균 최근접 cosine distance입니다. 높을수록 중복이 적고, 낮을수록 비슷한 샘플이 많이 뭉쳐 있다는 뜻입니다.",
}


@dataclass
class BufferEntry:
    source_index: int
    pos: int
    date_key: str
    time_value: pd.Timestamp | int
    region: int


def allocate_quotas(weights: np.ndarray, total: int) -> np.ndarray:
    if total <= 0 or len(weights) == 0:
        return np.zeros(len(weights), dtype=int)
    weights = np.asarray(weights, dtype=np.float64)
    if float(weights.sum()) <= 0.0:
        return np.zeros(len(weights), dtype=int)
    weights = weights / weights.sum()
    raw = weights * total
    quotas = np.floor(raw).astype(int)
    remain = int(total - quotas.sum())
    if remain > 0:
        order = np.argsort(-(raw - quotas))
        quotas[order[:remain]] += 1
    return quotas


def mean_nearest_neighbor_distance(normalized_vectors: np.ndarray, max_samples: int | None = None) -> float:
    n = len(normalized_vectors)
    if n < 2:
        return 0.0
    if max_samples is not None and n > max_samples:
        idx = np.linspace(0, n - 1, max_samples, dtype=int)
        normalized_vectors = normalized_vectors[idx]
    sims = normalized_vectors @ normalized_vectors.T
    np.fill_diagonal(sims, -np.inf)
    nn_sims = np.max(sims, axis=1)
    nn_dists = 1.0 - nn_sims
    return float(np.mean(nn_dists))


# ─────────────────────────────────────────────────────────────
# GUI
# ─────────────────────────────────────────────────────────────

@dataclass
class PlotStyle:
    point_size: float
    alpha: float


# Dataset colors for simulation results (distinct, vivid)
DATASET_COLORS = [
    (0.894, 0.102, 0.110, 0.85),   # red
    (0.216, 0.494, 0.722, 0.85),   # blue
    (0.302, 0.686, 0.290, 0.85),   # green
    (0.596, 0.306, 0.639, 0.85),   # purple
    (1.000, 0.498, 0.000, 0.85),   # orange
    (0.651, 0.337, 0.157, 0.85),   # brown
    (0.969, 0.506, 0.749, 0.85),   # pink
    (0.400, 0.761, 0.647, 0.85),   # teal
    (0.737, 0.741, 0.133, 0.85),   # olive
    (0.090, 0.745, 0.812, 0.85),   # cyan
]


class VectorDistanceSimulator:
    def __init__(
        self,
        root: tk.Tk,
        df: pd.DataFrame,
        vector_columns: list[str],
        csv_path: str,
        style: PlotStyle,
        skipped: int,
        standardize: bool = True,
    ):
        self.root = root
        self.df = df
        self.vector_columns = vector_columns
        self.csv_path = csv_path
        self.style = style
        self.total_loaded = len(df)
        self.skipped = skipped
        self.standardize = standardize

        self.side_values = sorted(self.df["side"].dropna().astype(str).unique().tolist())
        self.color_values = sorted(self.df["color"].dropna().astype(str).unique().tolist())
        self.px_values = sorted(self.df["px"].dropna().astype(str).unique().tolist())
        self.py_values = sorted(self.df["py"].dropna().astype(str).unique().tolist())

        self.side_vars: dict[str, tk.BooleanVar] = {
            v: tk.BooleanVar(value=True) for v in self.side_values
        }
        self.color_vars: dict[str, tk.BooleanVar] = {
            v: tk.BooleanVar(value=True) for v in self.color_values
        }
        self.px_vars: dict[str, tk.BooleanVar] = {
            v: tk.BooleanVar(value=True) for v in self.px_values
        }
        self.py_vars: dict[str, tk.BooleanVar] = {
            v: tk.BooleanVar(value=True) for v in self.py_values
        }

        self.dimension_var = tk.StringVar(value="3D")
        self.color_mode_var = tk.StringVar(value="color")
        self.method_var = tk.StringVar(value="PCA")
        self.path_filter_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="")
        self.sim_status_var = tk.StringVar(value="")

        self.side_palette = build_palette(self.side_values)
        self.color_palette = build_palette(self.color_values)

        # Left view (full dataset)
        self.left_figure = plt.Figure(figsize=(7, 6), dpi=100, facecolor="white")
        self.left_ax = None
        self.left_canvas = None
        self.left_scatter_lookup: list[tuple[object, pd.DataFrame, bool]] = []

        # Right view (simulation comparison)
        self.right_figure = plt.Figure(figsize=(7, 6), dpi=100, facecolor="white")
        self.right_ax = None
        self.right_canvas = None
        self.right_scatter_lookup: list[tuple[object, pd.DataFrame, bool]] = []
        self.right_hover_cid = None
        self.right_leave_cid = None
        self.right_click_cid = None

        # UMAP cache
        self._umap_cache: dict[int, np.ndarray] = {}

        # Simulation datasets
        self._sim_results: list[SimulationResult] = []
        self._sim_vars: list[tk.BooleanVar] = []  # checkbox vars
        self._sim_widgets: list[tk.Widget] = []  # for cleanup
        self._sim_dataset_frame: ttk.Frame | None = None

        # Simulation strategy controls
        self._strategy_var = tk.StringVar(value="Min Distance")
        self._param_entries: dict[str, tk.StringVar] = {}
        self._param_widgets: list[tk.Widget] = []

        # Hover tooltip
        self._tooltip = None
        self._tooltip_label = None
        self.annotation = None
        self.hover_cid = None
        self.leave_cid = None
        self.click_cid = None

        # Image preview window (reused)
        self._preview_win = None
        self._preview_img_label = None
        self._preview_meta_label = None
        self._preview_tk_img = None

        # ── Time-series mode state ──
        self._mode_var = tk.StringVar(value="normal")  # "normal" or "timeseries"
        self._ts_color_vars: dict[str, tk.BooleanVar] = {
            v: tk.BooleanVar(value=(i == 0)) for i, v in enumerate(self.color_values)
        }
        self._ts_dates: list[str] = []
        self._ts_date_idx: int = 0
        self._ts_view_mode_var = tk.StringVar(value="step")
        self._ts_show_prev_var = tk.BooleanVar(value=True)
        self._ts_strategy_var = tk.StringVar(value="Approx Density Memory")
        self._ts_strategy_desc_var = tk.StringVar(value="")
        self._ts_param_entries: dict[str, tk.StringVar] = {}
        self._ts_param_widgets: list[tk.Widget] = []
        self._ts_sim_result: dict | None = None
        self._ts_results: list[dict] = []
        self._ts_selected_result_idx: int | None = None
        self._ts_compare_tree: ttk.Treeview | None = None
        self._ts_date_tree: ttk.Treeview | None = None
        self._ts_suppress_date_select: bool = False
        self._ts_worker_thread: threading.Thread | None = None
        self._ts_worker_queue: Queue | None = None
        self._ts_worker_replace_all: bool = False
        self._ts_date_label_var = tk.StringVar(value="")
        self._ts_metrics_var = tk.StringVar(value="")
        self._ts_right_mode_var = tk.StringVar(value="step")
        self._ts_right_date_idx: int = 0
        self._ts_right_date_label_var = tk.StringVar(value="")
        self._ts_display_cache: dict[tuple[str, str, tuple[str, ...]], tuple[np.ndarray, np.ndarray]] = {}
        self._help_tooltip = None
        self._help_tooltip_label = None

        self._show_right_bg_var = tk.BooleanVar(value=True)

        self._build_ui()
        self._bind_filter_events()
        self._update_strategy_params()
        self.root.after(200, self.redraw)

    # ── UI ───────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self.root.title("Vector Distance Simulator")
        self.root.geometry("1920x900")

        outer = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        outer.pack(fill=tk.BOTH, expand=True)

        control_outer = ttk.Frame(outer, padding=0)
        self.left_plot_frame = ttk.Frame(outer, padding=4)
        self.right_plot_frame = ttk.Frame(outer, padding=4)
        outer.add(control_outer, weight=1)
        outer.add(self.left_plot_frame, weight=3)
        outer.add(self.right_plot_frame, weight=3)

        # ── Scrollable control panel ──
        control_canvas = tk.Canvas(control_outer, highlightthickness=0, width=320)
        scrollbar = ttk.Scrollbar(control_outer, orient="vertical", command=control_canvas.yview)
        control_frame = ttk.Frame(control_canvas, padding=12)
        control_frame.bind(
            "<Configure>",
            lambda e: control_canvas.configure(scrollregion=control_canvas.bbox("all")),
        )
        control_canvas.create_window((0, 0), window=control_frame, anchor="nw")
        control_canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        control_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        def _on_mousewheel_linux(event):
            control_canvas.yview_scroll(-1 if event.num == 4 else 1, "units")
        control_canvas.bind_all("<Button-4>", _on_mousewheel_linux)
        control_canvas.bind_all("<Button-5>", _on_mousewheel_linux)
        control_canvas.bind_all("<MouseWheel>",
            lambda e: control_canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"))

        # Title & info
        ttk.Label(control_frame, text="Vector Distance Simulator",
                  font=("Arial", 13, "bold")).pack(anchor="w", pady=(0, 8))
        ttk.Label(control_frame, text=f"CSV: {os.path.basename(self.csv_path)}",
                  wraplength=300, justify="left").pack(anchor="w", pady=(0, 6))
        summary = (
            f"Total rows: {self.total_loaded:,}  |  Skipped: {self.skipped:,}\n"
            f"Sides: {len(self.side_values)}, Colors: {len(self.color_values)}\n"
            f"Px: {len(self.px_values)}, Py: {len(self.py_values)}"
        )
        ttk.Label(control_frame, text=summary, justify="left").pack(anchor="w", pady=(0, 12))

        # ── Mode Toggle ──
        mode_box = ttk.LabelFrame(control_frame, text="📌 Mode", padding=8)
        mode_box.pack(fill=tk.X, pady=(0, 8))
        ttk.Radiobutton(mode_box, text="일반 (Simulation)", value="normal",
                        variable=self._mode_var, command=self._on_mode_switch).pack(anchor="w")
        ttk.Radiobutton(mode_box, text="📅 시계열 분석", value="timeseries",
                        variable=self._mode_var, command=self._on_mode_switch).pack(anchor="w")

        # Container for mode-specific panels (swapped on mode change)
        self._normal_panel = ttk.Frame(control_frame)
        self._normal_panel.pack(fill=tk.X)
        self._ts_panel = ttk.Frame(control_frame)
        # ts_panel starts hidden

        # ── NORMAL MODE panel contents ──
        np_ = self._normal_panel  # shorthand

        # ── Method (PCA / UMAP) ──
        method_box = ttk.LabelFrame(np_, text="Method", padding=8)
        method_box.pack(fill=tk.X, pady=(0, 8))
        ttk.Radiobutton(method_box, text="PCA", value="PCA", variable=self.method_var).pack(anchor="w")
        umap_label = "UMAP" if HAS_UMAP else "UMAP (not installed)"
        umap_state = "normal" if HAS_UMAP else "disabled"
        ttk.Radiobutton(method_box, text=umap_label, value="UMAP", variable=self.method_var,
                        state=umap_state).pack(anchor="w")

        # ── Dimension ──
        dim_box = ttk.LabelFrame(np_, text="Dimension", padding=8)
        dim_box.pack(fill=tk.X, pady=(0, 8))
        ttk.Radiobutton(dim_box, text="2D", value="2D", variable=self.dimension_var).pack(anchor="w")
        ttk.Radiobutton(dim_box, text="3D", value="3D", variable=self.dimension_var).pack(anchor="w")

        # ── Color Mode ──
        cmode_box = ttk.LabelFrame(np_, text="Color Mapping (Left View)", padding=8)
        cmode_box.pack(fill=tk.X, pady=(0, 8))
        ttk.Radiobutton(cmode_box, text="Color by color", value="color", variable=self.color_mode_var).pack(anchor="w")
        ttk.Radiobutton(cmode_box, text="Color by side", value="side", variable=self.color_mode_var).pack(anchor="w")

        # ── Path Filter ──
        search_box = ttk.LabelFrame(np_, text="Path Filter", padding=8)
        search_box.pack(fill=tk.X, pady=(0, 8))
        ttk.Entry(search_box, textvariable=self.path_filter_var).pack(fill=tk.X)

        # ── Buttons ──
        btn_box = ttk.Frame(np_)
        btn_box.pack(fill=tk.X, pady=(0, 8))

        redraw_btn = ttk.Button(btn_box, text="★ Redraw (recalculate)", command=self.redraw)
        redraw_btn.pack(fill=tk.X, pady=(0, 6))

        ttk.Label(btn_box, text="PCA: Redraw로 재계산\nUMAP: 필터 변경 시 자동 업데이트",
                  wraplength=300, justify="left", foreground="gray").pack(anchor="w", pady=(0, 6))

        btn_row = ttk.Frame(btn_box)
        btn_row.pack(fill=tk.X)
        ttk.Button(btn_row, text="All sides", command=self._select_all_sides).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
        ttk.Button(btn_row, text="Clear sides", command=self._clear_all_sides).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)

        btn_row2 = ttk.Frame(btn_box)
        btn_row2.pack(fill=tk.X, pady=(2, 0))
        ttk.Button(btn_row2, text="All colors", command=self._select_all_colors).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
        ttk.Button(btn_row2, text="Clear colors", command=self._clear_all_colors).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)

        btn_row3 = ttk.Frame(btn_box)
        btn_row3.pack(fill=tk.X, pady=(2, 0))
        ttk.Button(btn_row3, text="All px", command=self._select_all_px).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
        ttk.Button(btn_row3, text="Clear px", command=self._clear_all_px).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)

        btn_row4 = ttk.Frame(btn_box)
        btn_row4.pack(fill=tk.X, pady=(2, 0))
        ttk.Button(btn_row4, text="All py", command=self._select_all_py).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
        ttk.Button(btn_row4, text="Clear py", command=self._clear_all_py).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)

        # ── Side Filter ──
        side_box = ttk.LabelFrame(np_, text="Side Filter", padding=8)
        side_box.pack(fill=tk.X, pady=(0, 8))
        for v in self.side_values:
            ttk.Checkbutton(side_box, text=v, variable=self.side_vars[v]).pack(anchor="w")

        # ── Color Filter ──
        color_box = ttk.LabelFrame(np_, text="Color Filter", padding=8)
        color_box.pack(fill=tk.X, pady=(0, 8))
        for v in self.color_values:
            ttk.Checkbutton(color_box, text=v, variable=self.color_vars[v]).pack(anchor="w")

        px_box = ttk.LabelFrame(np_, text="Px Filter", padding=8)
        px_box.pack(fill=tk.X, pady=(0, 8))
        for v in self.px_values:
            ttk.Checkbutton(px_box, text=v, variable=self.px_vars[v]).pack(anchor="w")

        py_box = ttk.LabelFrame(np_, text="Py Filter", padding=8)
        py_box.pack(fill=tk.X, pady=(0, 8))
        for v in self.py_values:
            ttk.Checkbutton(py_box, text=v, variable=self.py_vars[v]).pack(anchor="w")

        # ── Simulation Controls ──
        sim_box = ttk.LabelFrame(np_, text="🎯 Simulation Controls", padding=8)
        sim_box.pack(fill=tk.X, pady=(0, 8))

        ttk.Label(sim_box, text="Strategy:", font=("Arial", 10, "bold")).pack(anchor="w")
        strategy_combo = ttk.Combobox(
            sim_box, textvariable=self._strategy_var,
            values=list(STRATEGIES.keys()), state="readonly", width=28
        )
        strategy_combo.pack(fill=tk.X, pady=(2, 4))
        strategy_combo.bind("<<ComboboxSelected>>", lambda e: self._update_strategy_params())

        self._strategy_desc_label = ttk.Label(
            sim_box, text="", wraplength=290, justify="left", foreground="gray"
        )
        self._strategy_desc_label.pack(anchor="w", pady=(0, 6))

        # Dynamic parameter frame
        self._param_frame = ttk.Frame(sim_box)
        self._param_frame.pack(fill=tk.X, pady=(0, 6))

        # Progress bar
        self._progress_var = tk.DoubleVar(value=0)
        self._progress_bar = ttk.Progressbar(
            sim_box, variable=self._progress_var, maximum=100
        )
        self._progress_bar.pack(fill=tk.X, pady=(0, 6))

        # Simulate button
        self._sim_button = ttk.Button(
            sim_box, text="▶ Simulate", command=self._run_simulation
        )
        self._sim_button.pack(fill=tk.X, pady=(0, 4))
        
        ttk.Checkbutton(sim_box, text="Show Original Background",
                        variable=self._show_right_bg_var,
                        command=self._redraw_right_view).pack(anchor="w", pady=(2, 6))

        ttk.Label(sim_box, textvariable=self.sim_status_var, wraplength=290,
                  justify="left", foreground="gray").pack(anchor="w")

        # ── Simulated Datasets ──
        ds_outer = ttk.LabelFrame(np_, text="📊 Simulated Datasets", padding=8)
        ds_outer.pack(fill=tk.X, pady=(0, 8))

        self._sim_dataset_frame = ttk.Frame(ds_outer)
        self._sim_dataset_frame.pack(fill=tk.X)

        self._no_datasets_label = ttk.Label(
            self._sim_dataset_frame, text="시뮬레이션을 실행하세요.",
            foreground="gray"
        )
        self._no_datasets_label.pack(anchor="w")

        ttk.Button(ds_outer, text="Clear All Datasets",
                   command=self._clear_all_datasets).pack(fill=tk.X, pady=(6, 0))

        # ── Status (shared) ──
        status_frame = ttk.LabelFrame(np_, text="Current View", padding=8)
        status_frame.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(status_frame, textvariable=self.status_var,
                  wraplength=300, justify="left").pack(fill=tk.X)

        # ── TIME-SERIES MODE panel contents ──
        self._build_ts_panel()

        # ── LEFT matplotlib canvas ──
        left_label = ttk.Label(self.left_plot_frame, text="📊 Full Dataset View",
                               font=("Arial", 11, "bold"))
        left_label.pack(anchor="w", pady=(0, 4))

        self.left_canvas = FigureCanvasTkAgg(self.left_figure, master=self.left_plot_frame)
        self.left_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar_left = NavigationToolbar2Tk(self.left_canvas, self.left_plot_frame)
        toolbar_left.update()
        toolbar_left.pack(side=tk.BOTTOM, fill=tk.X)

        # ── RIGHT matplotlib canvas ──
        right_label = ttk.Label(self.right_plot_frame, text="🔬 Simulation Comparison View",
                                font=("Arial", 11, "bold"))
        right_label.pack(anchor="w", pady=(0, 4))

        self.right_canvas = FigureCanvasTkAgg(self.right_figure, master=self.right_plot_frame)
        self.right_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar_right = NavigationToolbar2Tk(self.right_canvas, self.right_plot_frame)
        toolbar_right.update()
        toolbar_right.pack(side=tk.BOTTOM, fill=tk.X)

    # ── helpers ──────────────────────────────────────────────

    def _bind_filter_events(self):
        """When filters change and UMAP is active, auto-update both views."""
        def _on_filter_change(*_):
            if self.method_var.get() == "UMAP" and self._umap_cache:
                self._draw_from_umap_cache()
                self._redraw_right_view()

        for v in self.side_vars.values():
            v.trace_add("write", _on_filter_change)
        for v in self.color_vars.values():
            v.trace_add("write", _on_filter_change)
        for v in self.px_vars.values():
            v.trace_add("write", _on_filter_change)
        for v in self.py_vars.values():
            v.trace_add("write", _on_filter_change)
        self.path_filter_var.trace_add("write", _on_filter_change)
        self.color_mode_var.trace_add("write", _on_filter_change)
        self.dimension_var.trace_add("write", _on_filter_change)

    def _select_all_sides(self):
        for v in self.side_vars.values(): v.set(True)

    def _clear_all_sides(self):
        for v in self.side_vars.values(): v.set(False)

    def _select_all_colors(self):
        for v in self.color_vars.values(): v.set(True)

    def _clear_all_colors(self):
        for v in self.color_vars.values(): v.set(False)

    def _select_all_px(self):
        for v in self.px_vars.values(): v.set(True)

    def _clear_all_px(self):
        for v in self.px_vars.values(): v.set(False)

    def _select_all_py(self):
        for v in self.py_vars.values(): v.set(True)

    def _clear_all_py(self):
        for v in self.py_vars.values(): v.set(False)

    def _get_filtered_indices(self) -> np.ndarray:
        """Return boolean mask for self.df based on current filters."""
        sel_sides = {k for k, v in self.side_vars.items() if v.get()}
        sel_colors = {k for k, v in self.color_vars.items() if v.get()}
        sel_px = {k for k, v in self.px_vars.items() if v.get()}
        sel_py = {k for k, v in self.py_vars.items() if v.get()}
        mask = (
            self.df["side"].astype(str).isin(sel_sides)
            & self.df["color"].astype(str).isin(sel_colors)
            & self.df["px"].astype(str).isin(sel_px)
            & self.df["py"].astype(str).isin(sel_py)
        )
        pt = self.path_filter_var.get().strip().lower()
        if pt:
            mask = mask & self.df["path"].astype(str).str.lower().str.contains(pt, na=False)
        return mask.values

    def _get_filtered_df(self) -> pd.DataFrame:
        return self.df[self._get_filtered_indices()]

    # ── Simulation strategy parameter UI ──────────────────────

    def _update_strategy_params(self) -> None:
        """Re-build the parameter input fields when strategy changes."""
        # Clear old widgets
        for w in self._param_widgets:
            w.destroy()
        self._param_widgets.clear()
        self._param_entries.clear()

        strategy = self._strategy_var.get()
        info = STRATEGIES.get(strategy, {})

        # Update description
        self._strategy_desc_label.config(text=info.get("desc", ""))

        # Build parameter entry fields
        params = info.get("params", {})
        for key, pinfo in params.items():
            row = ttk.Frame(self._param_frame)
            row.pack(fill=tk.X, pady=2)
            self._param_widgets.append(row)

            lbl = ttk.Label(row, text=f"{pinfo['label']}:", width=20, anchor="w")
            lbl.pack(side=tk.LEFT)

            var = tk.StringVar(value=str(pinfo["default"]))
            self._param_entries[key] = var

            entry = ttk.Entry(row, textvariable=var, width=10)
            entry.pack(side=tk.LEFT, padx=(4, 0))

            hint = ttk.Label(row, text=f"({pinfo['min']}~{pinfo['max']})",
                            foreground="gray", font=("Arial", 8))
            hint.pack(side=tk.LEFT, padx=(4, 0))

    def _get_strategy_params(self) -> dict:
        """Parse current strategy parameter values."""
        strategy = self._strategy_var.get()
        info = STRATEGIES.get(strategy, {})
        result = {}
        for key, pinfo in info.get("params", {}).items():
            try:
                val = float(self._param_entries[key].get())
                val = max(pinfo["min"], min(pinfo["max"], val))
            except (ValueError, KeyError):
                val = pinfo["default"]
            # Keep k as int
            if key == "k":
                val = int(val)
            result[key] = val
        return result

    # ── Simulation execution ─────────────────────────────────

    def _run_simulation(self) -> None:
        """Run streaming simulation with current filters and strategy."""
        filtered = self._get_filtered_df()
        if len(filtered) < 2:
            self.sim_status_var.set("Not enough data points (need ≥ 2)")
            return

        strategy = self._strategy_var.get()
        params = self._get_strategy_params()

        self._sim_button.config(state="disabled")
        self._progress_var.set(0)
        self.sim_status_var.set(f"Simulating {strategy}...")
        self.root.update_idletasks()

        def progress_cb(current, total):
            self._progress_var.set(100.0 * current / total)
            self.root.update_idletasks()

        t0 = time.time()

        result = StreamingSimulator.simulate(
            filtered.reset_index(drop=True),
            self.vector_columns,
            strategy,
            params,
            progress_callback=progress_cb,
        )

        # Map accepted_indices back to original df indices
        filtered_indices = filtered.index.tolist()
        result.accepted_indices = [filtered_indices[i] for i in result.accepted_indices]

        elapsed = time.time() - t0
        self._sim_results.append(result)
        self._progress_var.set(100)

        self.sim_status_var.set(
            f"✓ {result.total_accepted}/{result.total_processed} accepted "
            f"({result.acceptance_rate:.1%}) in {elapsed:.1f}s"
        )
        self._sim_button.config(state="normal")

        # Add checkbox for this dataset
        self._add_dataset_checkbox(result, len(self._sim_results) - 1)

        # Redraw right view
        self._redraw_right_view()

    def _add_dataset_checkbox(self, result: SimulationResult, idx: int) -> None:
        """Add a checkbox for a new simulation dataset."""
        if self._no_datasets_label.winfo_ismapped():
            self._no_datasets_label.pack_forget()

        var = tk.BooleanVar(value=True)
        self._sim_vars.append(var)

        color = DATASET_COLORS[idx % len(DATASET_COLORS)]
        color_hex = "#{:02x}{:02x}{:02x}".format(
            int(color[0]*255), int(color[1]*255), int(color[2]*255)
        )

        frame = ttk.Frame(self._sim_dataset_frame)
        frame.pack(fill=tk.X, pady=1)
        self._sim_widgets.append(frame)

        # Color indicator
        color_label = tk.Label(frame, text="■", fg=color_hex, font=("Arial", 12))
        color_label.pack(side=tk.LEFT, padx=(0, 4))

        cb = ttk.Checkbutton(
            frame, text=result.name, variable=var,
            command=self._redraw_right_view
        )
        cb.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Delete button for individual dataset
        del_btn = ttk.Button(
            frame, text="✕", width=3,
            command=lambda i=idx: self._delete_dataset(i)
        )
        del_btn.pack(side=tk.RIGHT, padx=(4, 0))

    def _delete_dataset(self, idx: int) -> None:
        """Remove a single simulation dataset."""
        if 0 <= idx < len(self._sim_results):
            self._sim_results[idx] = None  # Mark as deleted
            self._sim_vars[idx].set(False)
            if idx < len(self._sim_widgets):
                self._sim_widgets[idx].pack_forget()
                self._sim_widgets[idx].destroy()
            self._redraw_right_view()

    def _clear_all_datasets(self) -> None:
        """Remove all simulation datasets."""
        self._sim_results.clear()
        self._sim_vars.clear()
        for w in self._sim_widgets:
            w.destroy()
        self._sim_widgets.clear()
        self._no_datasets_label.pack(anchor="w")
        self.sim_status_var.set("")
        self._redraw_right_view()

    # ── hover (tkinter tooltip window) ────────────────────────

    def _build_annotation(self):
        """Create a tkinter Toplevel tooltip (hidden by default)."""
        if self._tooltip is not None:
            self._tooltip.destroy()
        self._tooltip = tk.Toplevel(self.root)
        self._tooltip.wm_overrideredirect(True)
        self._tooltip.withdraw()

        self._tooltip_label = tk.Label(
            self._tooltip, text="", justify="left",
            background="#ffffee", foreground="#333333",
            relief="solid", borderwidth=1,
            font=("Consolas", 9), padx=8, pady=6,
        )
        self._tooltip_label.pack()
        self.annotation = True

    def _show_annotation(self, row, event):
        if self._tooltip is None:
            return
        path_str = str(row["path"])
        text = (
            f"type : {row['type']}\n"
            f"side : {row['side']}\n"
            f"color: {row['color']}\n"
            f"px   : {row['px']}\n"
            f"py   : {row['py']}\n"
            f"path : {path_str}"
        )
        self._tooltip_label.config(text=text)
        x_root = self.root.winfo_pointerx() + 18
        y_root = self.root.winfo_pointery() + 18
        self._tooltip.wm_geometry(f"+{x_root}+{y_root}")
        self._tooltip.deiconify()
        self._tooltip.lift()

    def _hide_annotation(self):
        if self._tooltip is not None:
            self._tooltip.withdraw()

    def _find_closest_point(self, event, scatter_lookup):
        """Find the closest scatter point to the mouse event across given scatter lookups.
        Returns (row, dist) or (None, inf)."""
        if event.inaxes is None:
            return None, float("inf")

        best_row = None
        best_dist = float("inf")

        for scatter, gdf, is_3d in scatter_lookup:
            ok, info = scatter.contains(event)
            ind = info.get("ind", [])
            if not ok or len(ind) == 0:
                continue
            # Pick the closest among matched indices
            for idx in ind:
                row = gdf.iloc[int(idx)]
                # Compute pixel distance to find truly closest
                try:
                    px, py = scatter.axes.transData.transform((float(row["C1"]), float(row["C2"])))
                    dx = px - event.x
                    dy = py - event.y
                    dist = dx * dx + dy * dy
                except Exception:
                    dist = 0.0
                if dist < best_dist:
                    best_dist = dist
                    best_row = row

        return best_row, best_dist

    def _on_hover_left(self, event):
        if self.left_ax is None or event.inaxes != self.left_ax:
            self._hide_annotation()
            return
        row, _ = self._find_closest_point(event, self.left_scatter_lookup)
        if row is not None:
            self._show_annotation(row, event)
        else:
            self._hide_annotation()

    def _on_click_left(self, event):
        """Click on a point to open image preview."""
        if self.left_ax is None or event.inaxes is None:
            return
        row, _ = self._find_closest_point(event, self.left_scatter_lookup)
        if row is not None:
            self._open_image(row)

    def _on_hover_right(self, event):
        if self.right_ax is None or event.inaxes != self.right_ax:
            self._hide_annotation()
            return
        row, _ = self._find_closest_point(event, self.right_scatter_lookup)
        if row is not None:
            self._show_annotation(row, event)
        else:
            self._hide_annotation()

    def _on_click_right(self, event):
        """Click on a simulation view point to open image preview."""
        print(f"[DEBUG-R] _on_click_right: button={event.button}, inaxes={event.inaxes}, right_ax={self.right_ax}")
        print(f"[DEBUG-R] right_scatter_lookup len={len(self.right_scatter_lookup)}")
        if self.right_ax is None or event.inaxes is None:
            print("[DEBUG-R] early return: right_ax or inaxes is None")
            return
        for i, (scatter, gdf, is_3d) in enumerate(self.right_scatter_lookup):
            ok, info = scatter.contains(event)
            ind = info.get("ind", [])
            print(f"[DEBUG-R] scatter[{i}] contains={ok}, ind_len={len(ind)}, gdf_len={len(gdf)}")
        row, _ = self._find_closest_point(event, self.right_scatter_lookup)
        if row is not None:
            print(f"[DEBUG-R] opening image: {row.get('path', 'NO_PATH')}")
            self._open_image(row)
        else:
            print("[DEBUG-R] no point found")

    def _open_image(self, row) -> None:
        """Open/update a reusable image preview window."""
        path = str(row["path"])
        if not os.path.isfile(path):
            from tkinter import messagebox
            messagebox.showinfo("File not found", f"Image file not found:\n{path}")
            return
        if not HAS_PIL:
            from tkinter import messagebox
            messagebox.showinfo("Pillow missing", f"Install Pillow to view images:\npip install Pillow\n\nPath: {path}")
            return
        try:
            img = Image.open(path)
        except Exception as exc:
            from tkinter import messagebox
            messagebox.showerror("Image error", f"Cannot open image:\n{path}\n\n{exc}")
            return

        max_size = 800
        w, h = img.size
        if w > max_size or h > max_size:
            ratio = min(max_size / w, max_size / h)
            img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)

        tk_img = ImageTk.PhotoImage(img)
        self._preview_tk_img = tk_img  # prevent GC

        meta_text = (
            f"path  : {path}\n"
            f"type  : {row['type']}\n"
            f"side  : {row['side']}\n"
            f"color : {row['color']}\n"
            f"px    : {row['px']}\n"
            f"py    : {row['py']}"
        )

        # Reuse existing window or create new one
        existing = self._preview_win is not None and self._preview_win.winfo_exists()
        print(f"[PREVIEW] _preview_win={self._preview_win}, exists={existing}")
        if existing:
            self._preview_win.title(os.path.basename(path))
            self._preview_win.geometry(f"{img.size[0]+20}x{img.size[1]+100}")
            self._preview_img_label.config(image=tk_img)
            self._preview_meta_label.config(text=meta_text)
            self._preview_win.lift()
        else:
            win = tk.Toplevel(self.root)
            win.title(os.path.basename(path))
            win.geometry(f"{img.size[0]+20}x{img.size[1]+100}")

            self._preview_img_label = tk.Label(win, image=tk_img)
            self._preview_img_label.pack(padx=5, pady=5)

            self._preview_meta_label = tk.Label(
                win, text=meta_text, font=("Consolas", 9), fg="#444444",
                justify="left", anchor="w",
            )
            self._preview_meta_label.pack(fill=tk.X, padx=10, pady=(0, 8))
            self._preview_win = win

    def _on_leave_left(self, _):
        self._hide_annotation()

    def _connect_left_hover(self):
        if self.hover_cid is not None:
            self.left_canvas.mpl_disconnect(self.hover_cid)
        if self.leave_cid is not None:
            self.left_canvas.mpl_disconnect(self.leave_cid)
        if self.click_cid is not None:
            self.left_canvas.mpl_disconnect(self.click_cid)
        self.hover_cid = self.left_canvas.mpl_connect("motion_notify_event", self._on_hover_left)
        self.leave_cid = self.left_canvas.mpl_connect("figure_leave_event", self._on_leave_left)
        self.click_cid = self.left_canvas.mpl_connect("button_press_event", self._on_click_left)

    def _connect_right_hover(self):
        if self.right_canvas is None:
            return
        if self.right_hover_cid is not None:
            self.right_canvas.mpl_disconnect(self.right_hover_cid)
        if self.right_leave_cid is not None:
            self.right_canvas.mpl_disconnect(self.right_leave_cid)
        if self.right_click_cid is not None:
            self.right_canvas.mpl_disconnect(self.right_click_cid)
        self.right_hover_cid = self.right_canvas.mpl_connect("motion_notify_event", self._on_hover_right)
        self.right_leave_cid = self.right_canvas.mpl_connect("figure_leave_event", self._on_leave_left)
        self.right_click_cid = self.right_canvas.mpl_connect("button_press_event", self._on_click_right)

    # ── Redraw (core method) ─────────────────────────────────

    def redraw(self) -> None:
        """Main redraw: recompute embedding then draw both views."""
        method = self.method_var.get()

        if method == "UMAP":
            is_3d = self.dimension_var.get() == "3D"
            n_dim = 3 if is_3d else 2

            if n_dim not in self._umap_cache:
                self.status_var.set(f"Computing UMAP ({n_dim}D) on {self.total_loaded:,} points...")
                self.root.update_idletasks()

                try:
                    emb = compute_embedding(
                        self.df, self.vector_columns,
                        method="UMAP", n_dim=n_dim, standardize=self.standardize,
                    )
                    self._umap_cache[n_dim] = emb
                except Exception as exc:
                    self._draw_error(f"UMAP error: {exc}")
                    return

            self._draw_from_umap_cache()
        else:
            self._draw_pca()

        self._redraw_right_view()

    def _draw_error(self, msg: str) -> None:
        self.left_figure.clf()
        is_3d = self.dimension_var.get() == "3D"
        self.left_ax = self.left_figure.add_subplot(111, projection="3d" if is_3d else None)
        if is_3d:
            self.left_ax.text2D(0.05, 0.95, msg, transform=self.left_ax.transAxes, fontsize=9, color="red")
        else:
            self.left_ax.text(0.05, 0.95, msg, transform=self.left_ax.transAxes, fontsize=9, color="red")
        self.status_var.set(msg)
        self.left_canvas.draw()

    def _draw_from_umap_cache(self) -> None:
        """Draw left view using cached UMAP embedding."""
        is_3d = self.dimension_var.get() == "3D"
        n_dim = 3 if is_3d else 2

        if n_dim not in self._umap_cache:
            self.redraw()
            return

        mask = self._get_filtered_indices()
        filtered = self.df[mask].copy()
        emb = self._umap_cache[n_dim][mask]

        filtered["C1"] = emb[:, 0]
        filtered["C2"] = emb[:, 1]
        filtered["C3"] = emb[:, 2] if n_dim >= 3 else 0.0

        self._draw_left_scatter(filtered, "UMAP", is_3d)

    def _draw_pca(self) -> None:
        """Compute PCA on the filtered subset and draw left view."""
        filtered = self._get_filtered_df()
        is_3d = self.dimension_var.get() == "3D"
        n_dim = 3 if is_3d else 2
        visible_count = len(filtered)

        if visible_count < 2:
            self._draw_error(f"PCA — not enough points ({visible_count}). Need ≥ 2.")
            return

        self.status_var.set(f"Computing PCA on {visible_count:,} points...")
        self.root.update_idletasks()

        try:
            emb = compute_embedding(
                filtered, self.vector_columns,
                method="PCA", n_dim=n_dim, standardize=self.standardize,
            )
        except Exception as exc:
            self._draw_error(f"PCA error: {exc}")
            return

        filtered = filtered.copy()
        filtered["C1"] = emb[:, 0]
        filtered["C2"] = emb[:, 1]
        filtered["C3"] = emb[:, 2] if n_dim >= 3 else 0.0

        self._draw_left_scatter(filtered, "PCA", is_3d)

    def _draw_left_scatter(self, filtered: pd.DataFrame, method: str, is_3d: bool) -> None:
        """Draw scatter plot in the LEFT view (full dataset)."""
        self.left_figure.clf()
        self.left_scatter_lookup = []

        self.left_ax = self.left_figure.add_subplot(111, projection="3d" if is_3d else None)
        ax = self.left_ax
        ax.set_xlabel("C1")
        ax.set_ylabel("C2")
        if is_3d:
            ax.set_zlabel("C3")

        self._build_annotation()
        visible_count = len(filtered)

        if visible_count < 2:
            ax.set_title(f"{method} — not enough points ({visible_count})", fontsize=11, pad=12)
            self.status_var.set(f"★ Displayed: {visible_count}\nTotal: {self.total_loaded:,}")
            self._connect_left_hover()
            self.left_canvas.draw()
            return

        ax.set_title(
            f"{method} ({('3D' if is_3d else '2D')})  —  {visible_count:,} points",
            pad=12, fontsize=11,
        )

        group_col = "color" if self.color_mode_var.get() == "color" else "side"
        palette = self.color_palette if group_col == "color" else self.side_palette

        for gval, gdf in filtered.groupby(group_col, sort=True):
            c = palette.get(str(gval), (0.5, 0.5, 0.5, 1.0))
            lab = f"{group_col}={gval} ({len(gdf)})"
            gdf = gdf.reset_index(drop=True)
            kw = dict(s=self.style.point_size, alpha=self.style.alpha, c=[c], label=lab, picker=True)
            if is_3d:
                sc = ax.scatter(gdf["C1"], gdf["C2"], gdf["C3"], depthshade=True, **kw)
            else:
                sc = ax.scatter(gdf["C1"], gdf["C2"], **kw)
            self.left_scatter_lookup.append((sc, gdf, is_3d))

        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, fontsize=8)
        self.left_figure.tight_layout()

        self.status_var.set(
            f"★ Displayed: {visible_count:,}  ({method})\n"
            f"Total: {self.total_loaded:,}  |  Skipped: {self.skipped:,}"
        )
        self._connect_left_hover()
        self.left_canvas.draw()

    # ── RIGHT VIEW drawing ────────────────────────────────────

    def _redraw_right_view(self) -> None:
        """Redraw the right comparison view with checked simulation datasets."""
        self.right_figure.clf()
        self.right_scatter_lookup = []
        is_3d = self.dimension_var.get() == "3D"
        method = self.method_var.get()
        n_dim = 3 if is_3d else 2

        self.right_ax = self.right_figure.add_subplot(111, projection="3d" if is_3d else None)
        ax = self.right_ax
        ax.set_xlabel("C1")
        ax.set_ylabel("C2")
        if is_3d:
            ax.set_zlabel("C3")

        if self._show_right_bg_var.get():
            for sc_obj, gdf, _ in self.left_scatter_lookup:
                bg_kw = dict(s=self.style.point_size * 0.5, c=[(1.0, 0.0, 0.0, 0.15)], label="Original Background")
                if is_3d:
                    ax.scatter(gdf["C1"], gdf["C2"], gdf["C3"], depthshade=True, **bg_kw)
                else:
                    ax.scatter(gdf["C1"], gdf["C2"], **bg_kw)

        # Collect checked datasets
        active_datasets: list[tuple[SimulationResult, int]] = []
        for i, result in enumerate(self._sim_results):
            if result is not None and i < len(self._sim_vars) and self._sim_vars[i].get():
                active_datasets.append((result, i))

        if not active_datasets:
            ax.set_title("Simulation Comparison (no datasets selected)", fontsize=11, pad=12)
            if is_3d:
                ax.text2D(0.5, 0.5, "시뮬레이션을 실행하고\n데이터셋을 선택하세요",
                          transform=ax.transAxes, ha="center", va="center",
                          fontsize=11, color="gray")
            else:
                ax.text(0.5, 0.5, "시뮬레이션을 실행하고\n데이터셋을 선택하세요",
                        transform=ax.transAxes, ha="center", va="center",
                        fontsize=11, color="gray")
            self.right_canvas.draw()
            return

        # Compute embedding for each dataset's points
        total_points = 0
        for result, idx in active_datasets:
            accepted_df = self.df.loc[result.accepted_indices]
            if len(accepted_df) < 2:
                continue

            # Compute embedding
            try:
                if method == "UMAP" and n_dim in self._umap_cache:
                    # Use cached UMAP coordinates
                    emb = self._umap_cache[n_dim][result.accepted_indices]
                else:
                    emb = compute_embedding(
                        accepted_df, self.vector_columns,
                        method=method, n_dim=n_dim, standardize=self.standardize,
                    )
            except Exception:
                continue

            if self._show_right_bg_var.get():
                color = "blue"
            else:
                color = DATASET_COLORS[idx % len(DATASET_COLORS)]
            n_pts = len(accepted_df)
            total_points += n_pts

            kw = dict(
                s=self.style.point_size * 1.2,
                alpha=0.75,
                c=[color],
                label=f"{result.name}",
                edgecolors="white",
                linewidths=0.3,
                picker=True,
            )

            plot_df = accepted_df.copy()
            plot_df["C1"] = emb[:, 0]
            plot_df["C2"] = emb[:, 1]
            if is_3d and n_dim >= 3:
                plot_df["C3"] = emb[:, 2]
                sc = ax.scatter(emb[:, 0], emb[:, 1], emb[:, 2], depthshade=True, **kw)
            else:
                sc = ax.scatter(emb[:, 0], emb[:, 1], **kw)
            self.right_scatter_lookup.append((sc, plot_df.reset_index(drop=True), is_3d))

        ax.set_title(
            f"Simulation Comparison ({len(active_datasets)} datasets, {total_points:,} pts)",
            fontsize=11, pad=12,
        )
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, fontsize=7)
        self.right_figure.tight_layout()
        self._connect_right_hover()
        self.right_canvas.draw()


    # ── TIME-SERIES mode panel builder ──────────────────────────

    def _build_ts_panel(self) -> None:
        """Build the time-series analysis panel."""
        tp = self._ts_panel

        row_md = ttk.Frame(tp)
        row_md.pack(fill=tk.X, pady=(0, 8))
        mf = ttk.LabelFrame(row_md, text="Method", padding=6)
        mf.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        ttk.Radiobutton(mf, text="PCA", value="PCA", variable=self.method_var, command=self._ts_redraw).pack(anchor="w")
        umap_state = "normal" if HAS_UMAP else "disabled"
        ttk.Radiobutton(mf, text="UMAP", value="UMAP", variable=self.method_var,
                        state=umap_state, command=self._ts_redraw).pack(anchor="w")
        df_ = ttk.LabelFrame(row_md, text="Dim", padding=6)
        df_.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Radiobutton(df_, text="2D", value="2D", variable=self.dimension_var, command=self._ts_redraw).pack(anchor="w")
        ttk.Radiobutton(df_, text="3D", value="3D", variable=self.dimension_var, command=self._ts_redraw).pack(anchor="w")

        ttk.Button(tp, text="★ Redraw (same coords for left/right)", command=self._ts_redraw).pack(fill=tk.X, pady=(0, 8))

        color_box = ttk.LabelFrame(tp, text="🎨 Color Selection (multi)", padding=8)
        color_box.pack(fill=tk.X, pady=(0, 8))
        for cv in self.color_values:
            ttk.Checkbutton(color_box, text=cv, variable=self._ts_color_vars[cv],
                            command=self._update_ts_dates).pack(anchor="w")
        csel_row = ttk.Frame(color_box)
        csel_row.pack(fill=tk.X, pady=(4, 0))
        ttk.Button(csel_row, text="All", command=lambda: [v.set(True) for v in self._ts_color_vars.values()] or self._update_ts_dates()).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
        ttk.Button(csel_row, text="Clear", command=lambda: [v.set(False) for v in self._ts_color_vars.values()] or self._update_ts_dates()).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)

        self._ts_date_info_frame = ttk.LabelFrame(tp, text="📅 Dates Found", padding=8)
        self._ts_date_info_frame.pack(fill=tk.X, pady=(0, 8))
        self._ts_date_list_label = ttk.Label(
            self._ts_date_info_frame, text="Color를 선택하세요",
            wraplength=300, justify="left", foreground="gray"
        )
        self._ts_date_list_label.pack(anchor="w")

        vm_box = ttk.LabelFrame(tp, text="📺 Left View", padding=8)
        vm_box.pack(fill=tk.X, pady=(0, 8))
        ttk.Radiobutton(vm_box, text="Step-by-Step", value="step",
                        variable=self._ts_view_mode_var,
                        command=self._ts_refresh_views).pack(anchor="w")
        ttk.Radiobutton(vm_box, text="All Dates Overview", value="overview",
                        variable=self._ts_view_mode_var,
                        command=self._ts_refresh_views).pack(anchor="w")

        nav_box = ttk.LabelFrame(tp, text="🔄 Left Step Navigation", padding=8)
        nav_box.pack(fill=tk.X, pady=(0, 8))
        nav_row = ttk.Frame(nav_box)
        nav_row.pack(fill=tk.X)
        ttk.Button(nav_row, text="◀", width=4, command=self._ts_prev_date).pack(side=tk.LEFT, padx=2)
        ttk.Label(nav_row, textvariable=self._ts_date_label_var,
                  font=("Consolas", 10, "bold"), anchor="center").pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(nav_row, text="▶", width=4, command=self._ts_next_date).pack(side=tk.RIGHT, padx=2)
        ttk.Checkbutton(nav_box, text="Show previous dates (gray)",
                        variable=self._ts_show_prev_var,
                        command=self._ts_refresh_views).pack(anchor="w", pady=(6, 0))

        sim_box = ttk.LabelFrame(tp, text="🎯 Buffer Tracking Simulation", padding=8)
        sim_box.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(sim_box, text="Strategy", font=("Arial", 10, "bold")).pack(anchor="w")
        ts_strategy_combo = ttk.Combobox(
            sim_box, textvariable=self._ts_strategy_var,
            values=list(BUFFER_STRATEGIES.keys()), state="readonly", width=28
        )
        ts_strategy_combo.pack(fill=tk.X, pady=(2, 4))
        ts_strategy_combo.bind("<<ComboboxSelected>>", lambda e: self._update_ts_strategy_params())

        self._ts_strategy_desc_label = ttk.Label(
            sim_box, textvariable=self._ts_strategy_desc_var,
            wraplength=300, justify="left", foreground="gray"
        )
        self._ts_strategy_desc_label.pack(anchor="w", pady=(0, 6))

        self._ts_param_frame = ttk.Frame(sim_box)
        self._ts_param_frame.pack(fill=tk.X, pady=(0, 6))

        thr_row = ttk.Frame(sim_box)
        thr_row.pack(fill=tk.X, pady=(0, 6))
        ttk.Button(thr_row, text="Estimate Threshold For N", command=self._ts_estimate_threshold_for_target).pack(side=tk.LEFT, fill=tk.X, expand=True)

        btn_row = ttk.Frame(sim_box)
        btn_row.pack(fill=tk.X, pady=(0, 4))
        self._ts_sim_button = ttk.Button(btn_row, text="Run Selected", command=self._ts_run_selected_strategy)
        self._ts_sim_button.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 2))
        self._ts_run_all_button = ttk.Button(btn_row, text="Run All", command=self._ts_run_all_strategies)
        self._ts_run_all_button.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        ttk.Button(btn_row, text="Clear", command=self._clear_ts_results).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(2, 0))

        self._ts_progress_var = tk.DoubleVar(value=0)
        ttk.Progressbar(sim_box, variable=self._ts_progress_var, maximum=100).pack(fill=tk.X, pady=(0, 6))

        compare_box = ttk.LabelFrame(tp, text="🏁 Strategy Comparison Table", padding=8)
        compare_box.pack(fill=tk.X, pady=(0, 8))
        columns = ("strategy", "track", "match", "coverage", "fill", "age", "nn")
        self._ts_compare_tree = ttk.Treeview(compare_box, columns=columns, show="headings", height=5, selectmode="browse")
        headings = {
            "strategy": "Strategy",
            "track": "Track",
            "match": "Match",
            "coverage": "Cover",
            "fill": "Fill",
            "age": "AvgAge",
            "nn": "MeanNN",
        }
        widths = {
            "strategy": 168,
            "track": 52,
            "match": 52,
            "coverage": 52,
            "fill": 48,
            "age": 58,
            "nn": 58,
        }
        for col in columns:
            self._ts_compare_tree.heading(col, text=headings[col])
            self._ts_compare_tree.column(col, width=widths[col], anchor="center", stretch=(col == "strategy"))
        self._ts_compare_tree.pack(fill=tk.X)
        self._ts_compare_tree.bind("<<TreeviewSelect>>", self._on_ts_result_select)

        guide_box = ttk.LabelFrame(tp, text="ℹ Metric Guide (hover for help)", padding=8)
        guide_box.pack(fill=tk.X, pady=(0, 8))
        for name, help_text in TS_METRIC_HELP.items():
            lbl = ttk.Label(guide_box, text=name, foreground="#1f4e79")
            lbl.pack(anchor="w")
            self._bind_help_tooltip(lbl, help_text)

        rv_box = ttk.LabelFrame(tp, text="🔬 Selected Strategy View", padding=8)
        rv_box.pack(fill=tk.X, pady=(0, 8))
        ttk.Radiobutton(rv_box, text="Current Buffer Snapshot", value="step",
                        variable=self._ts_right_mode_var,
                        command=self._on_ts_right_mode_change).pack(anchor="w")
        ttk.Radiobutton(rv_box, text="Buffer Evolution", value="all",
                        variable=self._ts_right_mode_var,
                        command=self._on_ts_right_mode_change).pack(anchor="w")
        ttk.Label(rv_box, text="Snapshot은 해당 날짜 단독 시뮬레이션, Evolution은 첫 날짜부터 누적 버퍼입니다.",
                  wraplength=300, justify="left", foreground="gray").pack(anchor="w", pady=(4, 0))
        rv_nav = ttk.Frame(rv_box)
        rv_nav.pack(fill=tk.X, pady=(4, 0))
        ttk.Button(rv_nav, text="◀", width=4, command=self._ts_right_prev).pack(side=tk.LEFT, padx=2)
        ttk.Label(rv_nav, textvariable=self._ts_right_date_label_var,
                  font=("Consolas", 9), anchor="center").pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(rv_nav, text="▶", width=4, command=self._ts_right_next).pack(side=tk.RIGHT, padx=2)
        ttk.Checkbutton(rv_box, text="Show Mother Background Overlay",
                        variable=self._show_right_bg_var,
                        command=self._ts_draw_right).pack(anchor="w", pady=(6, 0))

        date_box = ttk.LabelFrame(tp, text="🗓 Date Snapshot Table", padding=8)
        date_box.pack(fill=tk.X, pady=(0, 8))
        date_cols = ("date", "buf", "keep", "add", "drop", "track", "match")
        self._ts_date_tree = ttk.Treeview(date_box, columns=date_cols, show="headings", height=6, selectmode="browse")
        date_heads = {
            "date": "Date",
            "buf": "Buffer",
            "keep": "Keep",
            "add": "Add",
            "drop": "Drop",
            "track": "Track",
            "match": "Match",
        }
        date_widths = {
            "date": 80,
            "buf": 48,
            "keep": 48,
            "add": 48,
            "drop": 48,
            "track": 52,
            "match": 52,
        }
        for col in date_cols:
            self._ts_date_tree.heading(col, text=date_heads[col])
            self._ts_date_tree.column(col, width=date_widths[col], anchor="center", stretch=(col == "date"))
        self._ts_date_tree.pack(fill=tk.X)
        self._ts_date_tree.bind("<<TreeviewSelect>>", self._on_ts_date_select)

        metrics_box = ttk.LabelFrame(tp, text="📊 Selected Strategy Summary", padding=8)
        metrics_box.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(metrics_box, textvariable=self._ts_metrics_var,
                  wraplength=300, justify="left", font=("Consolas", 9)).pack(fill=tk.X)

        ts_status = ttk.LabelFrame(tp, text="Status", padding=8)
        ts_status.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(ts_status, textvariable=self.status_var,
                  wraplength=300, justify="left").pack(fill=tk.X)

        self._update_ts_strategy_params()

    def _bind_help_tooltip(self, widget: tk.Widget, text: str) -> None:
        widget.bind("<Enter>", lambda e, t=text: self._show_help_tooltip(t))
        widget.bind("<Leave>", lambda e: self._hide_help_tooltip())

    def _show_help_tooltip(self, text: str) -> None:
        if self._help_tooltip is None or not self._help_tooltip.winfo_exists():
            self._help_tooltip = tk.Toplevel(self.root)
            self._help_tooltip.wm_overrideredirect(True)
            self._help_tooltip.withdraw()
            self._help_tooltip_label = tk.Label(
                self._help_tooltip, text="", justify="left",
                background="#ffffee", foreground="#333333",
                relief="solid", borderwidth=1,
                font=("Consolas", 9), padx=8, pady=6,
            )
            self._help_tooltip_label.pack()
        self._help_tooltip_label.config(text=text)
        x_root = self.root.winfo_pointerx() + 18
        y_root = self.root.winfo_pointery() + 18
        self._help_tooltip.wm_geometry(f"+{x_root}+{y_root}")
        self._help_tooltip.deiconify()
        self._help_tooltip.lift()

    def _hide_help_tooltip(self) -> None:
        if self._help_tooltip is not None and self._help_tooltip.winfo_exists():
            self._help_tooltip.withdraw()

    def _refresh_ts_comparison_table(self) -> None:
        if self._ts_compare_tree is None:
            return
        self._ts_compare_tree.delete(*self._ts_compare_tree.get_children())
        self._ts_results.sort(key=lambda r: r["mean_tracking_score"], reverse=True)
        for idx, result in enumerate(self._ts_results):
            self._ts_compare_tree.insert(
                "", "end", iid=str(idx),
                values=(
                    result["strategy"],
                    f"{result['mean_tracking_score']:.1f}",
                    f"{result['final_match_score']:.1f}",
                    f"{result['final_coverage_score']:.1f}",
                    f"{result['mean_fill_ratio']:.0f}%",
                    f"{result['mean_age_days']:.1f}",
                    f"{result['mean_nn_dist']:.4f}",
                ),
            )
        if self._ts_results:
            self._ts_compare_tree.selection_set("0")
            self._set_selected_ts_result(0)

    def _on_ts_result_select(self, _event=None) -> None:
        if self._ts_compare_tree is None:
            return
        selected = self._ts_compare_tree.selection()
        if not selected:
            return
        self._set_selected_ts_result(int(selected[0]))

    def _set_selected_ts_result(self, idx: int) -> None:
        if idx < 0 or idx >= len(self._ts_results):
            return
        self._ts_selected_result_idx = idx
        self._ts_sim_result = self._ts_results[idx]
        per_date = list(self._ts_sim_result["per_date"].keys())
        if per_date:
            self._ts_right_date_idx = min(self._ts_right_date_idx, len(per_date) - 1)
            current_date = per_date[self._ts_right_date_idx]
            self._ts_right_date_label_var.set(
                f"{current_date}  ({self._ts_right_date_idx + 1}/{len(per_date)})"
            )
        self._refresh_ts_date_table()
        self._ts_metrics_var.set(self._format_ts_result_summary(self._ts_sim_result))
        self._ts_draw_right()

    def _clear_ts_results(self) -> None:
        self._ts_results.clear()
        self._ts_selected_result_idx = None
        self._ts_sim_result = None
        self._ts_metrics_var.set("")
        self._ts_right_date_idx = 0
        self._ts_right_date_label_var.set("")
        if self._ts_compare_tree is not None:
            self._ts_compare_tree.delete(*self._ts_compare_tree.get_children())
        if self._ts_date_tree is not None:
            self._ts_date_tree.delete(*self._ts_date_tree.get_children())
        self._ts_draw_right()

    def _format_ts_result_summary(self, result: dict) -> str:
        params = result["params"]
        lines = [
            f"Strategy : {result['strategy']}",
            f"Track    : {result['mean_tracking_score']:.1f}",
            f"Match    : {result['final_match_score']:.1f}",
            f"Coverage : {result['final_coverage_score']:.1f}",
            f"Fill     : {result['mean_fill_ratio']:.0f}% avg",
            f"Avg Age  : {result['mean_age_days']:.1f} days",
            f"Mean NN  : {result['mean_nn_dist']:.4f}",
            "",
            f"N={result['buffer_capacity']}  K={params['max_age_days']}  regions={result['region_count']}  thr={params['distance_threshold']:.3f}",
            "",
            f"{'Date':<10} {'Buf':>4} {'Track':>6} {'Match':>6}",
            "-" * 32,
        ]
        recent_items = list(result["per_date"].items())[-5:]
        for date, info in recent_items:
            lines.append(
                f"{date:<10} {len(info['buffer_indices']):>4} {info['tracking_score']:>6.1f} {info['match_score']:>6.1f}"
            )
        return "\n".join(lines)

    def _ts_get_view_snapshot(self, info: dict, mode: str | None = None) -> dict:
        mode = mode or self._ts_right_mode_var.get()
        if mode == "step":
            return {
                "buffer_indices": info.get("daily_buffer_indices", []),
                "tracking_score": float(info.get("daily_tracking_score", 0.0)),
                "match_score": float(info.get("daily_match_score", 0.0)),
                "coverage_score": float(info.get("daily_coverage_score", 0.0)),
                "avg_age_days": float(info.get("daily_avg_age_days", 0.0)),
                "label": "Daily Snapshot",
            }
        return {
            "buffer_indices": info.get("buffer_indices", []),
            "tracking_score": float(info.get("tracking_score", 0.0)),
            "match_score": float(info.get("match_score", 0.0)),
            "coverage_score": float(info.get("coverage_score", 0.0)),
            "avg_age_days": float(info.get("avg_age_days", 0.0)),
            "label": "Cumulative Evolution",
        }

    def _refresh_ts_date_table(self) -> None:
        if self._ts_date_tree is None:
            return
        self._ts_suppress_date_select = True
        try:
            self._ts_date_tree.delete(*self._ts_date_tree.get_children())
            if not self._ts_sim_result:
                return
            mode = self._ts_right_mode_var.get()
            for idx, (date, info) in enumerate(self._ts_sim_result["per_date"].items()):
                snap = self._ts_get_view_snapshot(info, mode)
                self._ts_date_tree.insert(
                    "", "end", iid=str(idx),
                    values=(
                        date,
                        len(snap["buffer_indices"]),
                        int(info.get("retained_count", 0)),
                        int(info.get("added_count", 0)),
                        int(info.get("dropped_count", 0)),
                        f"{snap['tracking_score']:.1f}",
                        f"{snap['match_score']:.1f}",
                    ),
                )
            current_idx = min(self._ts_right_date_idx, len(self._ts_sim_result["per_date"]) - 1)
            self._select_ts_date_row(current_idx)
        finally:
            self._ts_suppress_date_select = False

    def _select_ts_date_row(self, idx: int) -> None:
        if self._ts_date_tree is None:
            return
        iid = str(idx)
        if not self._ts_date_tree.exists(iid):
            return
        self._ts_suppress_date_select = True
        try:
            self._ts_date_tree.selection_set(iid)
            self._ts_date_tree.focus(iid)
            self._ts_date_tree.see(iid)
        finally:
            self._ts_suppress_date_select = False

    def _on_ts_date_select(self, _event=None) -> None:
        if self._ts_suppress_date_select or self._ts_date_tree is None or not self._ts_sim_result:
            return
        selected = self._ts_date_tree.selection()
        if not selected:
            return
        self._ts_right_date_idx = int(selected[0])
        dates = list(self._ts_sim_result["per_date"].keys())
        if dates:
            date = dates[self._ts_right_date_idx]
            self._ts_right_date_label_var.set(f"{date}  ({self._ts_right_date_idx + 1}/{len(dates)})")
        self._ts_draw_right()

    def _on_ts_right_mode_change(self) -> None:
        self._refresh_ts_date_table()
        self._ts_draw_right()

    # ── Mode switching ────────────────────────────────────────

    def _on_mode_switch(self) -> None:
        """Switch between normal and time-series mode."""
        mode = self._mode_var.get()
        if mode == "normal":
            self._ts_panel.pack_forget()
            self._normal_panel.pack(fill=tk.X)
            self.redraw()
        else:
            self._normal_panel.pack_forget()
            self._ts_panel.pack(fill=tk.X)
            self._update_ts_dates()

    # ── Time-series helpers ───────────────────────────────────

    def _update_ts_dates(self) -> None:
        sel_colors = {k for k, v in self._ts_color_vars.items() if v.get()}
        self._ts_display_cache.clear()
        if not sel_colors:
            self._ts_dates = []
            self._ts_date_list_label.config(text="Color를 선택하세요", foreground="gray")
            self._ts_date_label_var.set("—")
            self._clear_ts_results()
            return
        mask = self.df["color"].astype(str).isin(sel_colors)
        dates = sorted(self.df[mask]["side"].astype(str).unique().tolist())
        self._ts_dates = dates
        self._ts_date_idx = 0

        color_str = ", ".join(sorted(sel_colors))
        if dates:
            self._ts_date_list_label.config(
                text=f"[{color_str}] {len(dates)} dates: {', '.join(dates)}",
                foreground="black",
            )
            self._ts_date_label_var.set(f"{dates[0]}  (1/{len(dates)})")
        else:
            self._ts_date_list_label.config(text="해당 color에 데이터 없음", foreground="red")
            self._ts_date_label_var.set("—")

        self._clear_ts_results()
        self._ts_refresh_views()

    def _ts_prev_date(self) -> None:
        if not self._ts_dates:
            return
        self._ts_date_idx = max(0, self._ts_date_idx - 1)
        self._ts_date_label_var.set(
            f"{self._ts_dates[self._ts_date_idx]}  ({self._ts_date_idx + 1}/{len(self._ts_dates)})"
        )
        self._ts_refresh_views()

    def _ts_next_date(self) -> None:
        if not self._ts_dates:
            return
        self._ts_date_idx = min(len(self._ts_dates) - 1, self._ts_date_idx + 1)
        self._ts_date_label_var.set(
            f"{self._ts_dates[self._ts_date_idx]}  ({self._ts_date_idx + 1}/{len(self._ts_dates)})"
        )
        self._ts_refresh_views()

    def _ts_redraw(self) -> None:
        self._ts_display_cache.clear()
        self._umap_cache.clear()
        self._ts_refresh_views()

    def _update_ts_strategy_params(self) -> None:
        for w in self._ts_param_widgets:
            w.destroy()
        self._ts_param_widgets.clear()
        self._ts_param_entries.clear()

        strategy = self._ts_strategy_var.get()
        self._ts_strategy_desc_var.set(BUFFER_STRATEGIES.get(strategy, {}).get("desc", ""))

        for key, pinfo in TS_COMMON_PARAMS.items():
            row = ttk.Frame(self._ts_param_frame)
            row.pack(fill=tk.X, pady=2)
            self._ts_param_widgets.append(row)
            ttk.Label(row, text=f"{pinfo['label']}:", width=20, anchor="w").pack(side=tk.LEFT)
            var = tk.StringVar(value=str(pinfo["default"]))
            self._ts_param_entries[key] = var
            ttk.Entry(row, textvariable=var, width=10).pack(side=tk.LEFT, padx=(4, 0))
            ttk.Label(row, text=f"({pinfo['min']}~{pinfo['max']})",
                     foreground="gray", font=("Arial", 8)).pack(side=tk.LEFT, padx=(4, 0))

    def _get_ts_strategy_params(self) -> dict:
        result = {}
        for key, pinfo in TS_COMMON_PARAMS.items():
            raw = self._ts_param_entries.get(key, tk.StringVar(value=str(pinfo["default"]))).get()
            try:
                val = int(float(raw)) if pinfo["type"] == "int" else float(raw)
            except ValueError:
                val = pinfo["default"]
            val = max(pinfo["min"], min(pinfo["max"], val))
            if pinfo["type"] == "int":
                val = int(val)
            result[key] = val
        return result

    def _ts_right_prev(self) -> None:
        if not self._ts_sim_result:
            return
        dates = list(self._ts_sim_result["per_date"].keys())
        if not dates:
            return
        self._ts_right_date_idx = max(0, self._ts_right_date_idx - 1)
        date = dates[self._ts_right_date_idx]
        self._ts_right_date_label_var.set(f"{date}  ({self._ts_right_date_idx + 1}/{len(dates)})")
        self._select_ts_date_row(self._ts_right_date_idx)
        self._ts_draw_right()

    def _ts_right_next(self) -> None:
        if not self._ts_sim_result:
            return
        dates = list(self._ts_sim_result["per_date"].keys())
        if not dates:
            return
        self._ts_right_date_idx = min(len(dates) - 1, self._ts_right_date_idx + 1)
        date = dates[self._ts_right_date_idx]
        self._ts_right_date_label_var.set(f"{date}  ({self._ts_right_date_idx + 1}/{len(dates)})")
        self._select_ts_date_row(self._ts_right_date_idx)
        self._ts_draw_right()

    def _get_selected_colors(self) -> set[str]:
        return {k for k, v in self._ts_color_vars.items() if v.get()}

    def _ts_refresh_views(self) -> None:
        if self._mode_var.get() != "timeseries" or not self._ts_dates:
            return
        self._ts_draw_left()
        self._ts_draw_right()

    def _ts_get_display_df(self) -> pd.DataFrame | None:
        sel_colors = tuple(sorted(self._get_selected_colors()))
        if not sel_colors:
            return None
        key = (self.method_var.get(), self.dimension_var.get(), sel_colors)
        is_3d = self.dimension_var.get() == "3D"
        n_dim = 3 if is_3d else 2
        method = self.method_var.get()
        color_mask = self.df["color"].astype(str).isin(sel_colors)
        color_df = self.df[color_mask].copy()
        if len(color_df) < 2:
            return None
        if key not in self._ts_display_cache:
            emb = compute_embedding(
                color_df, self.vector_columns,
                method=method, n_dim=n_dim, standardize=self.standardize,
            )
            self._ts_display_cache[key] = (color_mask.to_numpy(), emb)
        mask_values, emb = self._ts_display_cache[key]
        color_df = self.df[mask_values].copy()
        color_df["C1"] = emb[:, 0]
        color_df["C2"] = emb[:, 1]
        color_df["C3"] = emb[:, 2] if n_dim >= 3 else 0.0
        return color_df

    # ── Time-series LEFT view ─────────────────────────────────

    def _ts_draw_left(self) -> None:
        sel_colors = self._get_selected_colors()
        if not sel_colors or not self._ts_dates:
            return

        is_3d = self.dimension_var.get() == "3D"
        view_mode = self._ts_view_mode_var.get()
        try:
            color_df = self._ts_get_display_df()
        except Exception as exc:
            self._draw_error(f"Embedding error: {exc}")
            return
        if color_df is None or len(color_df) < 2:
            self._draw_error("Not enough data for this color")
            return

        self.left_figure.clf()
        self.left_scatter_lookup = []
        self.left_ax = self.left_figure.add_subplot(111, projection="3d" if is_3d else None)
        ax = self.left_ax
        ax.set_xlabel("C1")
        ax.set_ylabel("C2")
        if is_3d:
            ax.set_zlabel("C3")

        self._build_annotation()

        if view_mode == "overview":
            self._ts_draw_left_overview(ax, color_df, is_3d)
        else:
            self._ts_draw_left_step(ax, color_df, is_3d)

        ax.grid(True, alpha=0.25)
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, fontsize=7)
        self.left_figure.tight_layout()
        self._connect_left_hover()
        self.left_canvas.draw()

    def _ts_draw_left_overview(self, ax, color_df: pd.DataFrame, is_3d: bool) -> None:
        n_dates = len(self._ts_dates)
        cmap = plt.get_cmap("coolwarm")

        ax.set_title(
            f"Mother Overview — {', '.join(sorted(self._get_selected_colors()))} ({n_dates} dates)",
            fontsize=11, pad=12,
        )

        for i, date in enumerate(self._ts_dates):
            date_df = color_df[color_df["side"].astype(str) == date]
            if len(date_df) == 0:
                continue
            c = cmap(i / max(n_dates - 1, 1))
            lab = f"{date} ({len(date_df)})"
            date_df = date_df.reset_index(drop=True)
            kw = dict(s=self.style.point_size, alpha=0.65, c=[c], label=lab, picker=True)
            if is_3d:
                sc = ax.scatter(date_df["C1"], date_df["C2"], date_df["C3"], depthshade=True, **kw)
            else:
                sc = ax.scatter(date_df["C1"], date_df["C2"], **kw)
            self.left_scatter_lookup.append((sc, date_df, is_3d))

        self.status_var.set(f"Mother overview: {n_dates} dates, {len(color_df):,} pts")

    def _ts_draw_left_step(self, ax, color_df: pd.DataFrame, is_3d: bool) -> None:
        if not self._ts_dates:
            return
        current_date = self._ts_dates[self._ts_date_idx]
        show_prev = self._ts_show_prev_var.get()

        ax.set_title(f"Mother Step: {current_date}", fontsize=11, pad=12)

        if show_prev and self._ts_date_idx > 0:
            prev_dates = self._ts_dates[:self._ts_date_idx]
            prev_df = color_df[color_df["side"].astype(str).isin(prev_dates)]
            if len(prev_df) > 0:
                prev_df = prev_df.reset_index(drop=True)
                kw = dict(s=self.style.point_size * 0.6, alpha=0.15,
                         c=[(0.5, 0.5, 0.5, 0.3)], label=f"prev ({len(prev_df)})")
                if is_3d:
                    sc = ax.scatter(prev_df["C1"], prev_df["C2"], prev_df["C3"], depthshade=True, **kw)
                else:
                    sc = ax.scatter(prev_df["C1"], prev_df["C2"], **kw)
                self.left_scatter_lookup.append((sc, prev_df, is_3d))

        cur_df = color_df[color_df["side"].astype(str) == current_date]
        if len(cur_df) > 0:
            cur_df = cur_df.reset_index(drop=True)
            kw = dict(s=self.style.point_size * 1.5, alpha=0.9,
                     c=[(0.894, 0.102, 0.110, 0.9)], label=f"{current_date} ({len(cur_df)})",
                     edgecolors="white", linewidths=0.5, picker=True)
            if is_3d:
                sc = ax.scatter(cur_df["C1"], cur_df["C2"], cur_df["C3"], depthshade=True, **kw)
            else:
                sc = ax.scatter(cur_df["C1"], cur_df["C2"], **kw)
            self.left_scatter_lookup.append((sc, cur_df, is_3d))

        self.status_var.set(
            f"Mother step {self._ts_date_idx + 1}/{len(self._ts_dates)}: {current_date}"
        )

    # ── Time-series RIGHT view ────────────────────────────────

    def _ts_draw_right(self) -> None:
        self.right_figure.clf()
        is_3d = self.dimension_var.get() == "3D"
        self.right_ax = self.right_figure.add_subplot(111, projection="3d" if is_3d else None)
        self.right_scatter_lookup = []
        ax = self.right_ax
        ax.set_xlabel("C1")
        ax.set_ylabel("C2")
        if is_3d:
            ax.set_zlabel("C3")

        try:
            color_df = self._ts_get_display_df()
        except Exception as exc:
            if is_3d:
                ax.text2D(0.05, 0.95, str(exc), transform=ax.transAxes, color="red")
            else:
                ax.text(0.05, 0.95, str(exc), transform=ax.transAxes, color="red")
            self.right_canvas.draw()
            return

        if color_df is None:
            ax.set_title("Selected Strategy View", fontsize=11, pad=12)
            msg = "전략을 실행한 뒤 비교 테이블에서 하나를 선택하세요"
            if is_3d:
                ax.text2D(0.5, 0.5, msg, transform=ax.transAxes, ha="center", va="center", fontsize=11, color="gray")
            else:
                ax.text(0.5, 0.5, msg, transform=ax.transAxes, ha="center", va="center", fontsize=11, color="gray")
            self.right_canvas.draw()
            return

        if self._show_right_bg_var.get():
            bg_kw = dict(s=self.style.point_size * 0.45, c=[(0.7, 0.2, 0.2, 0.12)], label="Mother Background")
            if is_3d:
                ax.scatter(color_df["C1"], color_df["C2"], color_df["C3"], depthshade=True, **bg_kw)
            else:
                ax.scatter(color_df["C1"], color_df["C2"], **bg_kw)

        result = self._ts_sim_result
        if not result:
            ax.set_title("Selected Strategy View", fontsize=11, pad=12)
            msg = "Run Selected 또는 Run All 실행 후, 비교 테이블에서 전략 1개를 선택하세요"
            if is_3d:
                ax.text2D(0.5, 0.5, msg, transform=ax.transAxes, ha="center", va="center", fontsize=11, color="gray")
            else:
                ax.text(0.5, 0.5, msg, transform=ax.transAxes, ha="center", va="center", fontsize=11, color="gray")
            self.right_canvas.draw()
            return

        items = list(result["per_date"].items())
        if not items:
            self.right_canvas.draw()
            return

        cmap = plt.get_cmap("coolwarm")
        right_mode = self._ts_right_mode_var.get()

        idx = min(self._ts_right_date_idx, len(items) - 1)
        date, snap = items[idx]
        view = self._ts_get_view_snapshot(snap, right_mode)
        buffer_df = color_df[color_df.index.isin(view["buffer_indices"])]
        if right_mode == "all" and idx > 0:
            prev_date, prev_snap = items[idx - 1]
            prev_set = set(prev_snap["buffer_indices"])
            cur_set = set(view["buffer_indices"])
            kept_df = color_df[color_df.index.isin(prev_set & cur_set)]
            added_df = color_df[color_df.index.isin(cur_set - prev_set)]
            dropped_df = color_df[color_df.index.isin(prev_set - cur_set)]
            if len(kept_df) > 0:
                kept_df = kept_df.reset_index(drop=True)
                kw = dict(s=self.style.point_size * 1.15, alpha=0.72, c=[(0.216, 0.494, 0.722, 0.75)], label=f"Retained ({len(kept_df)})", picker=True)
                if is_3d:
                    sc = ax.scatter(kept_df["C1"], kept_df["C2"], kept_df["C3"], depthshade=True, **kw)
                else:
                    sc = ax.scatter(kept_df["C1"], kept_df["C2"], **kw)
                self.right_scatter_lookup.append((sc, kept_df, is_3d))
            if len(added_df) > 0:
                added_df = added_df.reset_index(drop=True)
                kw = dict(s=self.style.point_size * 1.45, alpha=0.92, c=[(1.000, 0.498, 0.000, 0.90)], label=f"Added ({len(added_df)})", edgecolors="white", linewidths=0.4, picker=True)
                if is_3d:
                    sc = ax.scatter(added_df["C1"], added_df["C2"], added_df["C3"], depthshade=True, **kw)
                else:
                    sc = ax.scatter(added_df["C1"], added_df["C2"], **kw)
                self.right_scatter_lookup.append((sc, added_df, is_3d))
            if len(dropped_df) > 0:
                dropped_df = dropped_df.reset_index(drop=True)
                kw = dict(s=self.style.point_size * 1.55, alpha=0.95, c=[(0.839, 0.153, 0.157, 0.92)], label=f"Dropped ({len(dropped_df)})", marker="x", linewidths=1.0, picker=True)
                if is_3d:
                    sc = ax.scatter(dropped_df["C1"], dropped_df["C2"], dropped_df["C3"], depthshade=False, **kw)
                else:
                    sc = ax.scatter(dropped_df["C1"], dropped_df["C2"], **kw)
                self.right_scatter_lookup.append((sc, dropped_df, is_3d))
        else:
            if len(buffer_df) > 0:
                buffer_df = buffer_df.reset_index(drop=True)
                color = "blue" if self._show_right_bg_var.get() else (DATASET_COLORS[1] if right_mode == "step" else DATASET_COLORS[0])
                kw = dict(
                    s=self.style.point_size * (1.4 if right_mode == "step" else 1.3),
                    alpha=0.88 if right_mode == "step" else 0.9,
                    c=[color],
                    label=f"{view['label']} ({len(buffer_df)})",
                    edgecolors="white",
                    linewidths=0.5,
                    picker=True,
                )
                if is_3d:
                    sc = ax.scatter(buffer_df["C1"], buffer_df["C2"], buffer_df["C3"], depthshade=True, **kw)
                else:
                    sc = ax.scatter(buffer_df["C1"], buffer_df["C2"], **kw)
                self.right_scatter_lookup.append((sc, buffer_df, is_3d))
        title_mode = "Snapshot" if right_mode == "step" else "Evolution"
        ax.set_title(
            f"Selected: {result['strategy']} | {title_mode}: {date} | Buffer {len(view['buffer_indices'])} | Keep {int(snap.get('retained_count', 0))} | Add {int(snap.get('added_count', 0))} | Drop {int(snap.get('dropped_count', 0))}",
            fontsize=11, pad=12,
        )

        ax.grid(True, alpha=0.25)
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, fontsize=7)
        self.right_figure.tight_layout()
        self._connect_right_hover()
        self.right_canvas.draw()

    # ── Time-series simulation engine ─────────────────────────

    def _prepare_ts_simulation_context_for(self, sel_colors: list[str], dates: list[str], params: dict) -> dict | None:
        if not sel_colors or not dates:
            return None

        color_mask = self.df["color"].astype(str).isin(sel_colors)
        color_df = self.df[color_mask].copy()
        if len(color_df) < 2:
            return None

        vectors = color_df[self.vector_columns].to_numpy(dtype=np.float64)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        normalized = vectors / norms

        ref_input = StandardScaler().fit_transform(vectors) if self.standardize else vectors
        ref_dim = int(min(8, vectors.shape[1], max(1, len(color_df) - 1)))
        if ref_dim <= 0:
            ref_coords = np.zeros((len(color_df), 1), dtype=np.float64)
        else:
            ref_coords = PCA(n_components=ref_dim, svd_solver="full", random_state=42).fit_transform(ref_input)

        capacity = max(1, min(int(params["buffer_size"]), len(color_df)))
        region_count = max(1, min(int(params["region_count"]), capacity, len(color_df)))
        if region_count == 1:
            cluster_labels = np.zeros(len(color_df), dtype=int)
        else:
            cluster_labels = KMeans(n_clusters=region_count, random_state=42, n_init=10).fit_predict(ref_coords)

        mother_props = np.bincount(cluster_labels, minlength=region_count).astype(np.float64)
        mother_props /= max(mother_props.sum(), 1.0)
        quotas = allocate_quotas(mother_props, capacity)

        parsed = pd.to_datetime(pd.Series(dates, dtype=str), format="%y%m%d", errors="coerce")
        if parsed.notna().all():
            date_values = {date: parsed.iloc[i].normalize() for i, date in enumerate(dates)}
        else:
            fallback = pd.to_datetime(pd.Series(dates, dtype=str), errors="coerce")
            if fallback.notna().all():
                date_values = {date: fallback.iloc[i].normalize() for i, date in enumerate(dates)}
            else:
                date_values = {date: i for i, date in enumerate(dates)}

        date_positions = {}
        side_values = color_df["side"].astype(str).to_numpy()
        for date in dates:
            date_positions[date] = np.where(side_values == date)[0].tolist()

        return {
            "color_df": color_df,
            "normalized": normalized,
            "cluster_labels": cluster_labels,
            "mother_props": mother_props,
            "quotas": quotas,
            "capacity": capacity,
            "region_count": region_count,
            "date_values": date_values,
            "date_positions": date_positions,
            "dates": list(dates),
            "mother_mean_nn": mean_nearest_neighbor_distance(normalized),
        }

    def _prepare_ts_simulation_context(self, params: dict) -> dict | None:
        sel_colors = sorted(self._get_selected_colors())
        return self._prepare_ts_simulation_context_for(sel_colors, list(self._ts_dates), params)

    def _ts_age_days(self, current_value: pd.Timestamp | int, sample_value: pd.Timestamp | int) -> float:
        if isinstance(current_value, pd.Timestamp) and isinstance(sample_value, pd.Timestamp):
            return float(max((current_value - sample_value).days, 0))
        return float(max(int(current_value) - int(sample_value), 0))

    def _ts_min_cosine_distance(self, normalized: np.ndarray, pos: int, candidate_positions: list[int]) -> float:
        if not candidate_positions:
            return float("inf")
        sims = normalized[np.asarray(candidate_positions, dtype=int)] @ normalized[pos]
        return float(1.0 - np.max(sims))

    def _ts_effective_distance_threshold(self, strategy_name: str, context: dict, params: dict) -> float:
        base = float(params["distance_threshold"])
        mother_nn = float(context.get("mother_mean_nn", 0.0))
        if strategy_name == "Distance Gate FIFO":
            adaptive = max(0.005, mother_nn * 0.35)
        elif strategy_name == "Stable Coverage Memory":
            adaptive = max(0.004, mother_nn * 0.45)
        elif strategy_name == "Approx Density Memory":
            adaptive = max(0.003, mother_nn * 0.18)
        else:
            adaptive = max(0.003, mother_nn * 0.20)
        return float(min(base, adaptive) if base > 0 else adaptive)

    def _ts_should_accept_sample(
        self,
        strategy_name: str,
        pos: int,
        buffer: list[BufferEntry],
        cluster_counts: np.ndarray,
        context: dict,
        params: dict,
    ) -> bool:
        if strategy_name == "FIFO Baseline":
            return True

        if not buffer:
            return True

        normalized = context["normalized"]
        buffer_positions = [entry.pos for entry in buffer]
        global_min = self._ts_min_cosine_distance(normalized, pos, buffer_positions)
        effective_threshold = self._ts_effective_distance_threshold(strategy_name, context, params)

        if strategy_name == "Distance Gate FIFO":
            warmup_target = min(context["capacity"], max(8, context["region_count"] * 2))
            threshold = effective_threshold * (0.5 if len(buffer) < warmup_target else 1.0)
            return np.isinf(global_min) or global_min >= threshold

        if strategy_name == "Stable Coverage Memory":
            warmup_target = min(context["capacity"], max(16, context["region_count"] * 3))
            threshold = effective_threshold * (0.45 if len(buffer) < warmup_target else 1.0)
            return np.isinf(global_min) or global_min >= threshold

        if strategy_name == "Approx Density Memory":
            region = int(context["cluster_labels"][pos])
            quota = int(context["quotas"][region])
            region_count = int(cluster_counts[region])
            local_positions = [entry.pos for entry in buffer if entry.region == region]
            local_min = self._ts_min_cosine_distance(normalized, pos, local_positions if local_positions else buffer_positions)
            warmup_target = min(context["capacity"], max(12, context["region_count"] * 2))
            if region_count < quota:
                local_threshold = effective_threshold * (0.15 if region_count < max(1, quota // 3) else 0.5)
                return np.isinf(local_min) or local_min >= local_threshold
            if len(buffer) < context["capacity"] and region_count == 0:
                return True
            if len(buffer) < warmup_target:
                return np.isinf(global_min) or global_min >= effective_threshold * 0.5
            return np.isinf(local_min) or local_min >= effective_threshold

        region = int(context["cluster_labels"][pos])
        quota = int(context["quotas"][region])
        region_count = int(cluster_counts[region])
        deficit = quota - region_count
        local_positions = [entry.pos for entry in buffer if entry.region == region]
        local_min = self._ts_min_cosine_distance(normalized, pos, local_positions if local_positions else buffer_positions)

        # Quota-First: 목표 비율이 부족한 region은 우선 채운다.
        if deficit > 0:
            return True

        # 버퍼가 아직 덜 찼다면, 빈 region을 먼저 확보하고 같은 region 중복만 약하게 억제한다.
        if len(buffer) < context["capacity"]:
            if region_count == 0:
                return True
            return np.isinf(local_min) or local_min >= effective_threshold * 0.35

        # 버퍼가 찬 이후에는 over-quota region에서만 중복 억제를 강하게 적용한다.
        if region_count > quota:
            return np.isinf(local_min) or local_min >= effective_threshold

        return np.isinf(local_min) or local_min >= effective_threshold * 0.5

    def _ts_choose_eviction_candidate(
        self,
        strategy_name: str,
        buffer: list[BufferEntry],
        cluster_counts: np.ndarray,
        context: dict,
        current_value: pd.Timestamp | int,
    ) -> int:
        if strategy_name == "FIFO Baseline" or len(buffer) <= 1:
            return 0

        quotas = context["quotas"]
        normalized = context["normalized"]
        over_quota = np.maximum(cluster_counts - quotas, 0)
        must_reduce_regions = {i for i, v in enumerate(over_quota) if v > 0}
        if strategy_name == "Stable Coverage Memory":
            must_reduce_regions = set()

        best_idx = 0
        best_score = -float("inf")
        for i, entry in enumerate(buffer):
            if must_reduce_regions and entry.region not in must_reduce_regions:
                continue
            peers = [other.pos for j, other in enumerate(buffer) if j != i and other.region == entry.region]
            if not peers:
                peers = [other.pos for j, other in enumerate(buffer) if j != i]
            min_dist = self._ts_min_cosine_distance(normalized, entry.pos, peers)
            redundancy = 0.0 if np.isinf(min_dist) else max(0.0, 1.0 - min_dist)
            age = self._ts_age_days(current_value, entry.time_value)
            if strategy_name == "Stable Coverage Memory":
                score = redundancy * 100.0 + age
            elif strategy_name == "Approx Density Memory":
                score = float(over_quota[entry.region]) * 1000.0 + redundancy * 120.0 + age
            elif strategy_name == "Distance Gate FIFO":
                score = redundancy * 50.0 + age
            else:
                score = float(over_quota[entry.region]) * 1000.0 + redundancy * 100.0 + age
            if score > best_score:
                best_score = score
                best_idx = i
        return best_idx

    def _ts_snapshot_metrics(
        self,
        buffer: list[BufferEntry],
        cluster_counts: np.ndarray,
        context: dict,
        current_value: pd.Timestamp | int,
        params: dict,
    ) -> dict:
        if not buffer:
            return {
                "tracking_score": 0.0,
                "match_score": 0.0,
                "coverage_score": 0.0,
                "fill_ratio": 0.0,
                "avg_age_days": 0.0,
                "mean_nn_dist": 0.0,
            }

        capacity = context["capacity"]
        counts = cluster_counts.astype(np.float64)
        buffer_props = counts / max(float(len(buffer)), 1.0)
        l1_error = float(np.sum(np.abs(buffer_props - context["mother_props"])) / 2.0)
        match_score = max(0.0, 100.0 * (1.0 - l1_error))
        coverage_score = float(100.0 * context["mother_props"][counts > 0].sum())
        fill_ratio = float(100.0 * len(buffer) / max(capacity, 1))
        ages = [self._ts_age_days(current_value, entry.time_value) for entry in buffer]
        avg_age_days = float(np.mean(ages)) if ages else 0.0
        freshness_score = max(0.0, 100.0 * (1.0 - avg_age_days / max(float(params["max_age_days"]), 1.0)))
        positions = [entry.pos for entry in buffer]
        mean_nn_dist = mean_nearest_neighbor_distance(context["normalized"][positions], max_samples=256)
        tracking_score = (
            0.55 * match_score
            + 0.20 * coverage_score
            + 0.15 * fill_ratio
            + 0.10 * freshness_score
        )
        return {
            "tracking_score": float(tracking_score),
            "match_score": float(match_score),
            "coverage_score": float(coverage_score),
            "fill_ratio": float(fill_ratio),
            "avg_age_days": float(avg_age_days),
            "mean_nn_dist": float(mean_nn_dist),
        }

    def _ts_simulate_strategy(
        self,
        strategy_name: str,
        params: dict,
        context: dict,
        progress_callback: Callable[[float], None] | None = None,
    ) -> dict:
        buffer: list[BufferEntry] = []
        cluster_counts = np.zeros(context["region_count"], dtype=int)
        per_date: dict[str, dict] = {}
        total_samples = sum(len(v) for v in context["date_positions"].values())
        processed = 0
        prev_buffer_set: set[int] = set()

        for date in context["dates"]:
            current_value = context["date_values"][date]

            kept_buffer = []
            expired_count = 0
            for entry in buffer:
                if self._ts_age_days(current_value, entry.time_value) > params["max_age_days"]:
                    cluster_counts[entry.region] -= 1
                    expired_count += 1
                else:
                    kept_buffer.append(entry)
            buffer = kept_buffer

            accepted_count = 0
            evicted_count = 0
            for pos in context["date_positions"].get(date, []):
                processed += 1
                accept = self._ts_should_accept_sample(strategy_name, pos, buffer, cluster_counts, context, params)
                if accept:
                    accepted_count += 1
                    entry = BufferEntry(
                        source_index=int(context["color_df"].index[pos]),
                        pos=int(pos),
                        date_key=date,
                        time_value=current_value,
                        region=int(context["cluster_labels"][pos]),
                    )
                    buffer.append(entry)
                    cluster_counts[entry.region] += 1
                    while len(buffer) > context["capacity"]:
                        evict_idx = self._ts_choose_eviction_candidate(strategy_name, buffer, cluster_counts, context, current_value)
                        evicted = buffer.pop(evict_idx)
                        cluster_counts[evicted.region] -= 1
                        evicted_count += 1

                if progress_callback and (processed % 100 == 0 or processed == total_samples):
                    progress_callback(processed / max(total_samples, 1))

            incoming_count = len(context["date_positions"].get(date, []))
            buffer_indices = [entry.source_index for entry in buffer]
            buffer_set = set(buffer_indices)
            retained_count = len(prev_buffer_set & buffer_set)
            added_count = len(buffer_set - prev_buffer_set)
            dropped_count = len(prev_buffer_set - buffer_set)
            metrics = self._ts_snapshot_metrics(buffer, cluster_counts, context, current_value, params)
            daily_entries = [entry for entry in buffer if entry.date_key == date]
            daily_counts = np.zeros(context["region_count"], dtype=int)
            for entry in daily_entries:
                daily_counts[entry.region] += 1
            daily_metrics = self._ts_snapshot_metrics(daily_entries, daily_counts, context, current_value, params)
            per_date[date] = {
                "incoming": incoming_count,
                "accepted": accepted_count,
                "rejected": max(incoming_count - accepted_count, 0),
                "expired": expired_count,
                "evicted": evicted_count,
                "buffer_indices": buffer_indices,
                "retained_count": retained_count,
                "added_count": added_count,
                "dropped_count": dropped_count,
                "daily_buffer_indices": [entry.source_index for entry in daily_entries],
                "daily_tracking_score": daily_metrics["tracking_score"],
                "daily_match_score": daily_metrics["match_score"],
                "daily_coverage_score": daily_metrics["coverage_score"],
                "daily_avg_age_days": daily_metrics["avg_age_days"],
                "daily_mean_nn_dist": daily_metrics["mean_nn_dist"],
                **metrics,
            }
            prev_buffer_set = buffer_set

        snapshots = list(per_date.values())
        final = snapshots[-1] if snapshots else {
            "match_score": 0.0,
            "coverage_score": 0.0,
        }
        result = {
            "strategy": strategy_name,
            "params": params.copy(),
            "buffer_capacity": context["capacity"],
            "region_count": context["region_count"],
            "per_date": per_date,
            "mean_tracking_score": float(np.mean([snap["tracking_score"] for snap in snapshots])) if snapshots else 0.0,
            "final_match_score": float(final["match_score"]),
            "final_coverage_score": float(final["coverage_score"]),
            "mean_fill_ratio": float(np.mean([snap["fill_ratio"] for snap in snapshots])) if snapshots else 0.0,
            "mean_age_days": float(np.mean([snap["avg_age_days"] for snap in snapshots])) if snapshots else 0.0,
            "mean_nn_dist": float(np.mean([snap["mean_nn_dist"] for snap in snapshots])) if snapshots else 0.0,
        }
        return result

    def _ts_set_run_controls(self, running: bool) -> None:
        state = "disabled" if running else "normal"
        self._ts_sim_button.config(state=state)
        self._ts_run_all_button.config(state=state)

    def _ts_worker_main(
        self,
        strategy_names: list[str],
        params: dict,
        sel_colors: list[str],
        dates: list[str],
        replace_all: bool,
    ) -> None:
        assert self._ts_worker_queue is not None
        try:
            context = self._prepare_ts_simulation_context_for(sel_colors, dates, params)
            if context is None:
                self._ts_worker_queue.put({"type": "error", "message": "Not enough data"})
                return

            total = max(len(strategy_names), 1)
            results = []
            for idx, strategy_name in enumerate(strategy_names):
                base = idx / total
                span = 1.0 / total
                self._ts_worker_queue.put({
                    "type": "status",
                    "message": f"Running {strategy_name} ({idx + 1}/{total})...",
                })
                result = self._ts_simulate_strategy(
                    strategy_name,
                    params,
                    context,
                    progress_callback=lambda frac, b=base, s=span, name=strategy_name: self._ts_worker_queue.put({
                        "type": "progress",
                        "progress": (b + frac * s) * 100.0,
                        "message": f"Running {name}... {(b + frac * s) * 100.0:.0f}%",
                    }),
                )
                results.append(result)
                self._ts_worker_queue.put({
                    "type": "partial_result",
                    "result": result,
                    "replace_all": replace_all,
                })
            self._ts_worker_queue.put({"type": "done", "results": results, "replace_all": replace_all})
        except Exception as exc:
            self._ts_worker_queue.put({"type": "error", "message": str(exc)})

    def _ts_apply_partial_result(self, result: dict, replace_all: bool) -> None:
        if replace_all:
            self._ts_results = [r for r in self._ts_results if r["strategy"] != result["strategy"]]
            self._ts_results.append(result)
        else:
            self._ts_results = [r for r in self._ts_results if r["strategy"] != result["strategy"]]
            self._ts_results.append(result)
        self._refresh_ts_comparison_table()

    def _ts_poll_worker_queue(self) -> None:
        if self._ts_worker_queue is None:
            return
        done = False
        while True:
            try:
                msg = self._ts_worker_queue.get_nowait()
            except Empty:
                break

            msg_type = msg.get("type")
            if msg_type == "progress":
                self._ts_progress_var.set(float(msg.get("progress", 0.0)))
                self.status_var.set(msg.get("message", "Running simulation..."))
            elif msg_type == "status":
                self.status_var.set(msg.get("message", "Running simulation..."))
            elif msg_type == "partial_result":
                self._ts_apply_partial_result(msg["result"], bool(msg.get("replace_all", False)))
            elif msg_type == "done":
                done = True
                self._ts_progress_var.set(100)
                if self._ts_results:
                    best = max(self._ts_results, key=lambda r: r["mean_tracking_score"])
                    self.status_var.set(f"Best strategy: {best['strategy']} | Track {best['mean_tracking_score']:.1f}")
            elif msg_type == "error":
                done = True
                self._ts_metrics_var.set(msg.get("message", "Simulation failed"))
                self.status_var.set(msg.get("message", "Simulation failed"))

        if done:
            self._ts_set_run_controls(False)
            self._ts_worker_thread = None
            self._ts_worker_queue = None
            return

        self.root.after(80, self._ts_poll_worker_queue)

    def _ts_start_worker(self, strategy_names: list[str], replace_all: bool) -> None:
        if self._ts_worker_thread is not None and self._ts_worker_thread.is_alive():
            return
        if not self._ts_dates:
            return

        params = self._get_ts_strategy_params()
        sel_colors = sorted(self._get_selected_colors())
        if not sel_colors:
            self._ts_metrics_var.set("Not enough data")
            return

        if replace_all:
            self._clear_ts_results()
        self._ts_worker_replace_all = replace_all
        self._ts_progress_var.set(0)
        self.status_var.set("Preparing simulation...")
        self._ts_set_run_controls(True)
        self._ts_worker_queue = Queue()
        self._ts_worker_thread = threading.Thread(
            target=self._ts_worker_main,
            args=(strategy_names, params, sel_colors, list(self._ts_dates), replace_all),
            daemon=True,
        )
        self._ts_worker_thread.start()
        self.root.after(80, self._ts_poll_worker_queue)

    def _ts_run_selected_strategy(self) -> None:
        self._ts_start_worker([self._ts_strategy_var.get()], replace_all=False)

    def _ts_run_all_strategies(self) -> None:
        self._ts_start_worker(list(BUFFER_STRATEGIES.keys()), replace_all=True)

    def _ts_result_final_buffer_size(self, result: dict) -> int:
        per_date = result.get("per_date", {})
        if not per_date:
            return 0
        last = next(reversed(per_date.values()))
        return len(last.get("buffer_indices", []))

    def _ts_estimate_threshold_for_target(self) -> None:
        strategy_name = self._ts_strategy_var.get()
        if strategy_name == "FIFO Baseline":
            self.status_var.set("FIFO Baseline은 threshold를 사용하지 않습니다.")
            return
        if self._ts_worker_thread is not None and self._ts_worker_thread.is_alive():
            self.status_var.set("시뮬레이션 실행 중에는 threshold를 추정할 수 없습니다.")
            return
        if not self._ts_dates:
            self.status_var.set("날짜 데이터가 없습니다.")
            return

        params = self._get_ts_strategy_params()
        context = self._prepare_ts_simulation_context(params)
        if context is None:
            self.status_var.set("Threshold 추정을 위한 데이터가 부족합니다.")
            return

        target_n = int(params["buffer_size"])
        lo = 0.0
        hi = max(0.05, float(context.get("mother_mean_nn", 0.01)) * 4.0, float(params["distance_threshold"]) * 2.0)
        best_thr = hi
        best_gap = float("inf")
        best_count = 0

        self.status_var.set(f"Estimating threshold for N={target_n}...")
        self.root.update_idletasks()

        for _ in range(12):
            mid = (lo + hi) / 2.0
            trial = params.copy()
            trial["distance_threshold"] = mid
            result = self._ts_simulate_strategy(strategy_name, trial, context)
            count = self._ts_result_final_buffer_size(result)
            gap = abs(count - target_n)
            if gap < best_gap:
                best_gap = gap
                best_thr = mid
                best_count = count
            if count > target_n:
                lo = mid
            else:
                hi = mid

        if "distance_threshold" in self._ts_param_entries:
            self._ts_param_entries["distance_threshold"].set(f"{best_thr:.4f}")
        self.status_var.set(f"Estimated threshold {best_thr:.4f} for target N={target_n} | predicted final buffer {best_count}")

    def _ts_run_simulation(self) -> None:
        self._ts_run_selected_strategy()


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

def choose_csv_file(initial_path: str = "") -> str:
    if initial_path:
        return initial_path
    root = tk.Tk()
    root.withdraw()
    csv_path = filedialog.askopenfilename(
        title="Select vector CSV",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
    )
    root.destroy()
    return csv_path


def main() -> int:
    args = parse_args()
    csv_path = choose_csv_file(args.csv)
    if not csv_path:
        print("No CSV file selected.")
        return 1

    side_filter, sample_ratio = ask_user_options(csv_path)

    try:
        df, vector_columns, skipped = load_csv(
            csv_path,
            side_filter=side_filter,
            sample_ratio=sample_ratio,
        )
    except Exception as exc:
        print(f"CSV load error: {exc}", file=sys.stderr)
        return 1

    if len(df) < 2:
        print(f"Not enough valid data rows ({len(df)}). Need at least 2.")
        return 1

    print(f"Ready: {len(df):,} rows, {len(vector_columns)} vector dims, {skipped:,} skipped.", flush=True)

    root = tk.Tk()
    VectorDistanceSimulator(
        root=root,
        df=df,
        vector_columns=vector_columns,
        csv_path=csv_path,
        style=PlotStyle(point_size=args.point_size, alpha=args.alpha),
        skipped=skipped,
        standardize=not args.no_standardize,
    )
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
