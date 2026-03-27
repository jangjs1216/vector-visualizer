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
import sys
import time
import tkinter as tk
from dataclasses import dataclass, field
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

    # Side filter
    print("[1] Side filter")
    print(f"    Available sides: {', '.join(sides)}")
    print(f"    Enter side(s) separated by comma, or press Enter for ALL.")
    side_input = input("    > ").strip()

    side_filter: list[str] | None = None
    if side_input:
        selected = [s.strip() for s in side_input.split(",") if s.strip()]
        valid = [s for s in selected if s in sides]
        if valid:
            side_filter = valid
            print(f"    → Selected: {side_filter}")
        else:
            print(f"    → No valid sides matched, using ALL.")

    # Downsample
    print(f"\n[2] Downsampling")
    print(f"    Enter a ratio (e.g. 0.1 = 10%, 0.5 = 50%), or press Enter for 100%.")
    ratio_input = input("    > ").strip()

    sample_ratio = 1.0
    if ratio_input:
        try:
            sample_ratio = float(ratio_input)
            sample_ratio = max(0.01, min(1.0, sample_ratio))
            print(f"    → Ratio: {sample_ratio:.0%} (~{int(row_count * sample_ratio):,} rows)")
        except ValueError:
            print(f"    → Invalid input, using 100%.")
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
}

TS_COMMON_PARAMS = {
    "buffer_size": {"label": "Buffer Size (N)", "default": 256, "min": 16, "max": 5000, "type": "int"},
    "max_age_days": {"label": "Max Age Days (K)", "default": 30, "min": 1, "max": 365, "type": "int"},
    "region_count": {"label": "Region Count", "default": 8, "min": 1, "max": 64, "type": "int"},
    "distance_threshold": {"label": "Cosine Dist Threshold", "default": 0.08, "min": 0.0, "max": 2.0, "type": "float"},
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


def mean_nearest_neighbor_distance(normalized_vectors: np.ndarray) -> float:
    if len(normalized_vectors) < 2:
        return 0.0
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

        self.side_vars: dict[str, tk.BooleanVar] = {
            v: tk.BooleanVar(value=True) for v in self.side_values
        }
        self.color_vars: dict[str, tk.BooleanVar] = {
            v: tk.BooleanVar(value=True) for v in self.color_values
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
        self._ts_strategy_var = tk.StringVar(value="Quota-First FIFO")
        self._ts_strategy_desc_var = tk.StringVar(value="")
        self._ts_param_entries: dict[str, tk.StringVar] = {}
        self._ts_param_widgets: list[tk.Widget] = []
        self._ts_sim_result: dict | None = None
        self._ts_results: list[dict] = []
        self._ts_selected_result_idx: int | None = None
        self._ts_compare_tree: ttk.Treeview | None = None
        self._ts_date_tree: ttk.Treeview | None = None
        self._ts_suppress_date_select: bool = False
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
            f"Sides: {len(self.side_values)}, Colors: {len(self.color_values)}"
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

    def _get_filtered_indices(self) -> np.ndarray:
        """Return boolean mask for self.df based on current filters."""
        sel_sides = {k for k, v in self.side_vars.items() if v.get()}
        sel_colors = {k for k, v in self.color_vars.items() if v.get()}
        mask = (
            self.df["side"].astype(str).isin(sel_sides)
            & self.df["color"].astype(str).isin(sel_colors)
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
            f"color : {row['color']}"
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
            "strategy": 118,
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
        date_cols = ("date", "buf", "track", "match", "cover", "age")
        self._ts_date_tree = ttk.Treeview(date_box, columns=date_cols, show="headings", height=6, selectmode="browse")
        date_heads = {
            "date": "Date",
            "buf": "Buffer",
            "track": "Track",
            "match": "Match",
            "cover": "Cover",
            "age": "AvgAge",
        }
        date_widths = {
            "date": 88,
            "buf": 48,
            "track": 52,
            "match": 52,
            "cover": 52,
            "age": 58,
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

    def _refresh_ts_date_table(self) -> None:
        if self._ts_date_tree is None:
            return
        self._ts_suppress_date_select = True
        try:
            self._ts_date_tree.delete(*self._ts_date_tree.get_children())
            if not self._ts_sim_result:
                return
            for idx, (date, info) in enumerate(self._ts_sim_result["per_date"].items()):
                self._ts_date_tree.insert(
                    "", "end", iid=str(idx),
                    values=(
                        date,
                        len(info["buffer_indices"]),
                        f"{info['tracking_score']:.1f}",
                        f"{info['match_score']:.1f}",
                        f"{info['coverage_score']:.1f}",
                        f"{info['avg_age_days']:.1f}",
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

        if right_mode == "step":
            idx = min(self._ts_right_date_idx, len(items) - 1)
            date, snap = items[idx]
            buffer_df = color_df[color_df.index.isin(snap["buffer_indices"])]
            if len(buffer_df) > 0:
                buffer_df = buffer_df.reset_index(drop=True)
                color = "blue" if self._show_right_bg_var.get() else DATASET_COLORS[1]
                kw = dict(s=self.style.point_size * 1.4, alpha=0.88, c=[color],
                         label=f"{result['strategy']} ({len(buffer_df)})",
                         edgecolors="white", linewidths=0.5, picker=True)
                if is_3d:
                    sc = ax.scatter(buffer_df["C1"], buffer_df["C2"], buffer_df["C3"], depthshade=True, **kw)
                else:
                    sc = ax.scatter(buffer_df["C1"], buffer_df["C2"], **kw)
                self.right_scatter_lookup.append((sc, buffer_df, is_3d))
            ax.set_title(
                f"Selected: {result['strategy']} | Snapshot: {date} | Buffer {len(snap['buffer_indices'])} | Track {snap['tracking_score']:.1f} | Match {snap['match_score']:.1f}",
                fontsize=11, pad=12,
            )
        else:
            idx = min(self._ts_right_date_idx, len(items) - 1)
            date, snap = items[idx]
            buffer_df = color_df[color_df.index.isin(snap["buffer_indices"])]
            if len(buffer_df) > 0:
                buffer_df = buffer_df.reset_index(drop=True)
                c = "blue" if self._show_right_bg_var.get() else DATASET_COLORS[0]
                kw = dict(
                    s=self.style.point_size * 1.3,
                    alpha=0.9,
                    c=[c],
                    label=f"{date} ({len(buffer_df)})",
                    edgecolors="white",
                    linewidths=0.5,
                    picker=True,
                )
                if is_3d:
                    sc = ax.scatter(buffer_df["C1"], buffer_df["C2"], buffer_df["C3"], depthshade=True, **kw)
                else:
                    sc = ax.scatter(buffer_df["C1"], buffer_df["C2"], **kw)
                self.right_scatter_lookup.append((sc, buffer_df, is_3d))
            ax.set_title(
                f"Selected: {result['strategy']} | Evolution Date: {date} | Buffer {len(snap['buffer_indices'])} | Track {snap['tracking_score']:.1f} | Match {snap['match_score']:.1f}",
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

    def _prepare_ts_simulation_context(self, params: dict) -> dict | None:
        sel_colors = self._get_selected_colors()
        if not sel_colors or not self._ts_dates:
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

        dates = self._ts_dates
        parsed = pd.to_datetime(pd.Series(dates), errors="coerce")
        if parsed.notna().all():
            date_values = {date: parsed.iloc[i].normalize() for i, date in enumerate(dates)}
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
            "mother_mean_nn": mean_nearest_neighbor_distance(normalized),
        }

    def _ts_age_days(self, current_value: pd.Timestamp | int, sample_value: pd.Timestamp | int) -> float:
        if isinstance(current_value, pd.Timestamp) and isinstance(sample_value, pd.Timestamp):
            return float(max((current_value - sample_value).days, 0))
        return float(max(int(current_value) - int(sample_value), 0))

    def _ts_min_cosine_distance(self, normalized: np.ndarray, pos: int, candidate_positions: list[int]) -> float:
        if not candidate_positions:
            return float("inf")
        sims = normalized[np.asarray(candidate_positions, dtype=int)] @ normalized[pos]
        return float(1.0 - np.max(sims))

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

        normalized = context["normalized"]
        buffer_positions = [entry.pos for entry in buffer]
        global_min = self._ts_min_cosine_distance(normalized, pos, buffer_positions)

        if strategy_name == "Distance Gate FIFO":
            return np.isinf(global_min) or global_min >= params["distance_threshold"]

        region = int(context["cluster_labels"][pos])
        deficit = int(context["quotas"][region] - cluster_counts[region])
        local_positions = [entry.pos for entry in buffer if entry.region == region]
        local_min = self._ts_min_cosine_distance(normalized, pos, local_positions if local_positions else buffer_positions)

        if deficit > 0:
            return np.isinf(local_min) or local_min >= params["distance_threshold"] * 0.25
        if len(buffer) < context["capacity"]:
            return np.isinf(global_min) or global_min >= params["distance_threshold"]
        return np.isinf(local_min) or local_min >= params["distance_threshold"]

    def _ts_choose_eviction_candidate(
        self,
        strategy_name: str,
        buffer: list[BufferEntry],
        cluster_counts: np.ndarray,
        context: dict,
        current_value: pd.Timestamp | int,
    ) -> int:
        if strategy_name in {"FIFO Baseline", "Distance Gate FIFO"} or len(buffer) <= 1:
            return 0

        quotas = context["quotas"]
        normalized = context["normalized"]
        over_quota = np.maximum(cluster_counts - quotas, 0)
        must_reduce_regions = {i for i, v in enumerate(over_quota) if v > 0}

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
        mean_nn_dist = mean_nearest_neighbor_distance(context["normalized"][positions])
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

        for date in self._ts_dates:
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

            evicted_count = 0
            for pos in context["date_positions"].get(date, []):
                processed += 1
                accept = self._ts_should_accept_sample(strategy_name, pos, buffer, cluster_counts, context, params)
                if accept:
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

            buffer_indices = [entry.source_index for entry in buffer]
            kept_today = sum(1 for entry in buffer if entry.date_key == date)
            metrics = self._ts_snapshot_metrics(buffer, cluster_counts, context, current_value, params)
            per_date[date] = {
                "incoming": len(context["date_positions"].get(date, [])),
                "accepted": kept_today,
                "rejected": len(context["date_positions"].get(date, [])) - kept_today,
                "expired": expired_count,
                "evicted": evicted_count,
                "buffer_indices": buffer_indices,
                **metrics,
            }

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

    def _ts_run_selected_strategy(self) -> None:
        if not self._ts_dates:
            return
        params = self._get_ts_strategy_params()
        context = self._prepare_ts_simulation_context(params)
        if context is None:
            self._ts_metrics_var.set("Not enough data")
            return

        strategy_name = self._ts_strategy_var.get()
        self.status_var.set(f"Running {strategy_name}...")
        self._ts_progress_var.set(0)
        self._ts_sim_button.config(state="disabled")
        self._ts_run_all_button.config(state="disabled")
        self.root.update_idletasks()

        result = self._ts_simulate_strategy(
            strategy_name, params, context,
            progress_callback=lambda frac: (self._ts_progress_var.set(frac * 100.0), self.root.update_idletasks()),
        )

        self._ts_sim_button.config(state="normal")
        self._ts_run_all_button.config(state="normal")
        self._ts_progress_var.set(100)

        self._ts_results = [r for r in self._ts_results if r["strategy"] != strategy_name]
        self._ts_results.append(result)
        self._refresh_ts_comparison_table()
        self.status_var.set(f"Completed: {strategy_name} | Track {result['mean_tracking_score']:.1f}")

    def _ts_run_all_strategies(self) -> None:
        if not self._ts_dates:
            return
        params = self._get_ts_strategy_params()
        context = self._prepare_ts_simulation_context(params)
        if context is None:
            self._ts_metrics_var.set("Not enough data")
            return

        self.status_var.set("Running all strategies...")
        self._ts_progress_var.set(0)
        self._ts_sim_button.config(state="disabled")
        self._ts_run_all_button.config(state="disabled")
        self.root.update_idletasks()

        strategy_names = list(BUFFER_STRATEGIES.keys())
        results = []
        for idx, strategy_name in enumerate(strategy_names):
            base = idx / max(len(strategy_names), 1)
            span = 1.0 / max(len(strategy_names), 1)
            result = self._ts_simulate_strategy(
                strategy_name, params, context,
                progress_callback=lambda frac, b=base, s=span: (
                    self._ts_progress_var.set((b + frac * s) * 100.0),
                    self.root.update_idletasks(),
                ),
            )
            results.append(result)

        self._ts_results = results
        self._refresh_ts_comparison_table()
        self._ts_sim_button.config(state="normal")
        self._ts_run_all_button.config(state="normal")
        self._ts_progress_var.set(100)
        if self._ts_results:
            best = max(self._ts_results, key=lambda r: r["mean_tracking_score"])
            self.status_var.set(f"Best strategy: {best['strategy']} | Track {best['mean_tracking_score']:.1f}")

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
