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

        # ── Time-series mode state ──
        self._mode_var = tk.StringVar(value="normal")  # "normal" or "timeseries"
        self._ts_color_vars: dict[str, tk.BooleanVar] = {
            v: tk.BooleanVar(value=(i == 0)) for i, v in enumerate(self.color_values)
        }
        self._ts_dates: list[str] = []  # sorted date list for selected colors
        self._ts_date_idx: int = 0  # current step index
        self._ts_view_mode_var = tk.StringVar(value="step")  # "step" or "overview"
        self._ts_show_prev_var = tk.BooleanVar(value=True)
        self._ts_strategy_var = tk.StringVar(value="Min Distance")
        self._ts_param_entries: dict[str, tk.StringVar] = {}
        self._ts_param_widgets: list[tk.Widget] = []
        self._ts_sim_result: dict | None = None  # date-sequential sim result
        self._ts_panel_widgets: list[tk.Widget] = []  # for dynamic cleanup
        self._ts_date_label_var = tk.StringVar(value="")
        self._ts_metrics_var = tk.StringVar(value="")
        self._ts_right_mode_var = tk.StringVar(value="all")  # "all" or "step"
        self._ts_right_date_idx: int = 0
        self._ts_right_date_label_var = tk.StringVar(value="")

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

    def _on_hover_left(self, event):
        if self.left_ax is None or event.inaxes != self.left_ax:
            self._hide_annotation()
            return
        for scatter, gdf, _ in self.left_scatter_lookup:
            ok, info = scatter.contains(event)
            if ok and info.get("ind", []):
                self._show_annotation(gdf.iloc[int(info["ind"][0])], event)
                return
        self._hide_annotation()

    def _on_click_left(self, event):
        """Click on a point to open image preview."""
        if self.left_ax is None or event.inaxes != self.left_ax:
            return
        for scatter, gdf, _ in self.left_scatter_lookup:
            ok, info = scatter.contains(event)
            if ok and info.get("ind", []):
                row = gdf.iloc[int(info["ind"][0])]
                self._open_image(row)
                return

    def _open_image(self, row) -> None:
        """Open an image file in a new window with metadata."""
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

        win = tk.Toplevel(self.root)
        win.title(os.path.basename(path))
        win.geometry(f"{img.size[0]+20}x{img.size[1]+100}")

        tk_img = ImageTk.PhotoImage(img)
        label = tk.Label(win, image=tk_img)
        label.image = tk_img  # prevent GC
        label.pack(padx=5, pady=5)

        meta_text = (
            f"path  : {path}\n"
            f"type  : {row['type']}\n"
            f"side  : {row['side']}\n"
            f"color : {row['color']}"
        )
        tk.Label(win, text=meta_text, font=("Consolas", 9), fg="#444444",
                 justify="left", anchor="w").pack(fill=tk.X, padx=10, pady=(0, 8))

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
        is_3d = self.dimension_var.get() == "3D"
        method = self.method_var.get()
        n_dim = 3 if is_3d else 2

        self.right_ax = self.right_figure.add_subplot(111, projection="3d" if is_3d else None)
        ax = self.right_ax
        ax.set_xlabel("C1")
        ax.set_ylabel("C2")
        if is_3d:
            ax.set_zlabel("C3")

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

            if is_3d and n_dim >= 3:
                ax.scatter(emb[:, 0], emb[:, 1], emb[:, 2], depthshade=True, **kw)
            else:
                ax.scatter(emb[:, 0], emb[:, 1], **kw)

        ax.set_title(
            f"Simulation Comparison ({len(active_datasets)} datasets, {total_points:,} pts)",
            fontsize=11, pad=12,
        )
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, fontsize=7)
        self.right_figure.tight_layout()
        self.right_canvas.draw()


    # ── TIME-SERIES mode panel builder ──────────────────────────

    def _build_ts_panel(self) -> None:
        """Build the time-series analysis mode panel (initially hidden)."""
        tp = self._ts_panel

        # Method / Dimension (shared with normal mode but duplicated for clarity)
        row_md = ttk.Frame(tp)
        row_md.pack(fill=tk.X, pady=(0, 8))
        mf = ttk.LabelFrame(row_md, text="Method", padding=6)
        mf.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        ttk.Radiobutton(mf, text="PCA", value="PCA", variable=self.method_var).pack(anchor="w")
        umap_state = "normal" if HAS_UMAP else "disabled"
        ttk.Radiobutton(mf, text="UMAP", value="UMAP", variable=self.method_var,
                        state=umap_state).pack(anchor="w")
        df_ = ttk.LabelFrame(row_md, text="Dim", padding=6)
        df_.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Radiobutton(df_, text="2D", value="2D", variable=self.dimension_var).pack(anchor="w")
        ttk.Radiobutton(df_, text="3D", value="3D", variable=self.dimension_var).pack(anchor="w")

        # Redraw button
        ttk.Button(tp, text="★ Redraw (recalculate)", command=self._ts_redraw).pack(fill=tk.X, pady=(0, 8))

        # Color selection (multi-select checkboxes)
        color_box = ttk.LabelFrame(tp, text="🎨 Color Selection (multi)", padding=8)
        color_box.pack(fill=tk.X, pady=(0, 8))
        for cv in self.color_values:
            ttk.Checkbutton(color_box, text=cv, variable=self._ts_color_vars[cv],
                            command=self._update_ts_dates).pack(anchor="w")
        csel_row = ttk.Frame(color_box)
        csel_row.pack(fill=tk.X, pady=(4, 0))
        ttk.Button(csel_row, text="All", command=lambda: [v.set(True) for v in self._ts_color_vars.values()] or self._update_ts_dates()).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)
        ttk.Button(csel_row, text="Clear", command=lambda: [v.set(False) for v in self._ts_color_vars.values()] or self._update_ts_dates()).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=1)

        # Date info
        self._ts_date_info_frame = ttk.LabelFrame(tp, text="📅 Dates Found", padding=8)
        self._ts_date_info_frame.pack(fill=tk.X, pady=(0, 8))
        self._ts_date_list_label = ttk.Label(
            self._ts_date_info_frame, text="Color를 선택하세요",
            wraplength=290, justify="left", foreground="gray"
        )
        self._ts_date_list_label.pack(anchor="w")

        # View mode
        vm_box = ttk.LabelFrame(tp, text="📺 View Mode", padding=8)
        vm_box.pack(fill=tk.X, pady=(0, 8))
        ttk.Radiobutton(vm_box, text="Step-by-Step", value="step",
                        variable=self._ts_view_mode_var,
                        command=self._ts_refresh_views).pack(anchor="w")
        ttk.Radiobutton(vm_box, text="All Dates Overview", value="overview",
                        variable=self._ts_view_mode_var,
                        command=self._ts_refresh_views).pack(anchor="w")

        # Step navigation
        nav_box = ttk.LabelFrame(tp, text="🔄 Navigation", padding=8)
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

        # Simulation controls — strategy selector
        sim_box = ttk.LabelFrame(tp, text="🎯 Date-Sequential Simulation", padding=8)
        sim_box.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(sim_box, text="Strategy:", font=("Arial", 10, "bold")).pack(anchor="w")
        ts_strategy_combo = ttk.Combobox(
            sim_box, textvariable=self._ts_strategy_var,
            values=list(STRATEGIES.keys()), state="readonly", width=28
        )
        ts_strategy_combo.pack(fill=tk.X, pady=(2, 4))
        ts_strategy_combo.bind("<<ComboboxSelected>>", lambda e: self._update_ts_strategy_params())
        self._ts_param_frame = ttk.Frame(sim_box)
        self._ts_param_frame.pack(fill=tk.X, pady=(0, 6))

        self._ts_sim_button = ttk.Button(sim_box, text="▶ Run Date-Sequential Sim",
                                          command=self._ts_run_simulation)
        self._ts_sim_button.pack(fill=tk.X, pady=(0, 4))
        self._ts_progress_var = tk.DoubleVar(value=0)
        ttk.Progressbar(sim_box, variable=self._ts_progress_var, maximum=100).pack(fill=tk.X, pady=(0, 6))

        # Right view mode — step through sim results or show all
        rv_box = ttk.LabelFrame(tp, text="🔬 Right View", padding=8)
        rv_box.pack(fill=tk.X, pady=(0, 8))
        ttk.Radiobutton(rv_box, text="Show All Dates", value="all",
                        variable=self._ts_right_mode_var,
                        command=self._ts_draw_right).pack(anchor="w")
        ttk.Radiobutton(rv_box, text="Step-by-Step", value="step",
                        variable=self._ts_right_mode_var,
                        command=self._ts_draw_right).pack(anchor="w")
        rv_nav = ttk.Frame(rv_box)
        rv_nav.pack(fill=tk.X, pady=(4, 0))
        ttk.Button(rv_nav, text="◀", width=4, command=self._ts_right_prev).pack(side=tk.LEFT, padx=2)
        ttk.Label(rv_nav, textvariable=self._ts_right_date_label_var,
                  font=("Consolas", 9), anchor="center").pack(side=tk.LEFT, expand=True, fill=tk.X)
        ttk.Button(rv_nav, text="▶", width=4, command=self._ts_right_next).pack(side=tk.RIGHT, padx=2)

        # Metrics
        metrics_box = ttk.LabelFrame(tp, text="📊 Tracking Metrics", padding=8)
        metrics_box.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(metrics_box, textvariable=self._ts_metrics_var,
                  wraplength=290, justify="left", font=("Consolas", 9)).pack(fill=tk.X)

        # Status
        ts_status = ttk.LabelFrame(tp, text="Status", padding=8)
        ts_status.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(ts_status, textvariable=self.status_var,
                  wraplength=300, justify="left").pack(fill=tk.X)

        self._update_ts_strategy_params()

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
        """Update the date list based on selected colors (multi-select)."""
        sel_colors = {k for k, v in self._ts_color_vars.items() if v.get()}
        if not sel_colors:
            self._ts_dates = []
            self._ts_date_list_label.config(text="Color를 선택하세요", foreground="gray")
            self._ts_date_label_var.set("—")
            return
        mask = self.df["color"].astype(str).isin(sel_colors)
        dates = sorted(self.df[mask]["side"].astype(str).unique().tolist())
        self._ts_dates = dates
        self._ts_date_idx = 0

        color_str = ", ".join(sorted(sel_colors))
        if dates:
            self._ts_date_list_label.config(
                text=f"[{color_str}] {len(dates)} dates: {', '.join(dates)}",
                foreground="black"
            )
            self._ts_date_label_var.set(f"{dates[0]}  (1/{len(dates)})")
        else:
            self._ts_date_list_label.config(text="해당 color에 데이터 없음", foreground="red")
            self._ts_date_label_var.set("—")

        self._ts_sim_result = None
        self._ts_metrics_var.set("")
        self._ts_refresh_views()

    def _ts_prev_date(self) -> None:
        if not self._ts_dates:
            return
        self._ts_date_idx = max(0, self._ts_date_idx - 1)
        self._ts_date_label_var.set(
            f"{self._ts_dates[self._ts_date_idx]}  ({self._ts_date_idx+1}/{len(self._ts_dates)})"
        )
        self._ts_refresh_views()

    def _ts_next_date(self) -> None:
        if not self._ts_dates:
            return
        self._ts_date_idx = min(len(self._ts_dates) - 1, self._ts_date_idx + 1)
        self._ts_date_label_var.set(
            f"{self._ts_dates[self._ts_date_idx]}  ({self._ts_date_idx+1}/{len(self._ts_dates)})"
        )
        self._ts_refresh_views()

    def _ts_redraw(self) -> None:
        """Redraw for time-series mode (recompute PCA/UMAP, then refresh views)."""
        self._umap_cache.clear()
        self._ts_refresh_views()

    def _update_ts_strategy_params(self) -> None:
        """Re-build strategy param inputs for time-series sim."""
        for w in self._ts_param_widgets:
            w.destroy()
        self._ts_param_widgets.clear()
        self._ts_param_entries.clear()
        strategy = self._ts_strategy_var.get()
        info = STRATEGIES.get(strategy, {})
        for key, pinfo in info.get("params", {}).items():
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
        """Parse current TS strategy params."""
        strategy = self._ts_strategy_var.get()
        info = STRATEGIES.get(strategy, {})
        result = {}
        for key, pinfo in info.get("params", {}).items():
            try:
                val = float(self._ts_param_entries[key].get())
                val = max(pinfo["min"], min(pinfo["max"], val))
            except (ValueError, KeyError):
                val = pinfo["default"]
            if key == "k":
                val = int(val)
            result[key] = val
        return result

    def _ts_right_prev(self) -> None:
        if not self._ts_sim_result or not self._ts_dates:
            return
        self._ts_right_date_idx = max(0, self._ts_right_date_idx - 1)
        d = self._ts_dates[self._ts_right_date_idx]
        self._ts_right_date_label_var.set(f"{d}  ({self._ts_right_date_idx+1}/{len(self._ts_dates)})")
        self._ts_draw_right()

    def _ts_right_next(self) -> None:
        if not self._ts_sim_result or not self._ts_dates:
            return
        self._ts_right_date_idx = min(len(self._ts_dates) - 1, self._ts_right_date_idx + 1)
        d = self._ts_dates[self._ts_right_date_idx]
        self._ts_right_date_label_var.set(f"{d}  ({self._ts_right_date_idx+1}/{len(self._ts_dates)})")
        self._ts_draw_right()

    def _get_selected_colors(self) -> set[str]:
        return {k for k, v in self._ts_color_vars.items() if v.get()}

    def _ts_refresh_views(self) -> None:
        """Refresh both left and right views in time-series mode."""
        if self._mode_var.get() != "timeseries" or not self._ts_dates:
            return
        self._ts_draw_left()
        self._ts_draw_right()

    # ── Time-series LEFT view ─────────────────────────────────

    def _ts_draw_left(self) -> None:
        """Draw left view for time-series mode."""
        sel_colors = self._get_selected_colors()
        if not sel_colors or not self._ts_dates:
            return

        is_3d = self.dimension_var.get() == "3D"
        n_dim = 3 if is_3d else 2
        method = self.method_var.get()
        view_mode = self._ts_view_mode_var.get()

        # Get all data for selected colors
        sel_colors = self._get_selected_colors()
        if not sel_colors:
            return
        color_mask = self.df["color"].astype(str).isin(sel_colors)
        color_df = self.df[color_mask]

        if len(color_df) < 2:
            self._draw_error("Not enough data for this color")
            return

        # Compute embedding on all data for this color
        try:
            if method == "UMAP" and n_dim in self._umap_cache:
                emb = self._umap_cache[n_dim][color_mask.values]
            else:
                emb = compute_embedding(
                    color_df, self.vector_columns,
                    method=method, n_dim=n_dim, standardize=self.standardize,
                )
                if method == "UMAP":
                    # Cache at full df level
                    full_emb = np.zeros((len(self.df), n_dim))
                    full_emb[color_mask.values] = emb
                    self._umap_cache[n_dim] = full_emb
        except Exception as exc:
            self._draw_error(f"Embedding error: {exc}")
            return

        color_df = color_df.copy()
        color_df["C1"] = emb[:, 0]
        color_df["C2"] = emb[:, 1]
        color_df["C3"] = emb[:, 2] if n_dim >= 3 else 0.0

        # Draw
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
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, fontsize=7)
        self.left_figure.tight_layout()
        self._connect_left_hover()
        self.left_canvas.draw()

    def _ts_draw_left_overview(self, ax, color_df: pd.DataFrame, is_3d: bool) -> None:
        """Draw all dates with gradient colors (cool→warm)."""
        n_dates = len(self._ts_dates)
        cmap = plt.get_cmap("coolwarm")

        ax.set_title(f"All Dates Overview — {', '.join(sorted(self._get_selected_colors()))} ({n_dates} dates)",
                     fontsize=11, pad=12)

        for i, date in enumerate(self._ts_dates):
            date_df = color_df[color_df["side"].astype(str) == date]
            if len(date_df) == 0:
                continue
            c = cmap(i / max(n_dates - 1, 1))
            lab = f"{date} ({len(date_df)})"
            date_df = date_df.reset_index(drop=True)
            kw = dict(s=self.style.point_size, alpha=0.7, c=[c], label=lab, picker=True)
            if is_3d:
                sc = ax.scatter(date_df["C1"], date_df["C2"], date_df["C3"], depthshade=True, **kw)
            else:
                sc = ax.scatter(date_df["C1"], date_df["C2"], **kw)
            self.left_scatter_lookup.append((sc, date_df, is_3d))

        self.status_var.set(f"📅 Overview: {n_dates} dates, {len(color_df):,} pts")

    def _ts_draw_left_step(self, ax, color_df: pd.DataFrame, is_3d: bool) -> None:
        """Draw step-by-step view: current date highlighted, previous dates gray."""
        if not self._ts_dates:
            return
        current_date = self._ts_dates[self._ts_date_idx]
        show_prev = self._ts_show_prev_var.get()

        ax.set_title(f"Step: {current_date} — {', '.join(sorted(self._get_selected_colors()))}", fontsize=11, pad=12)

        # Draw previous dates in gray if enabled
        if show_prev and self._ts_date_idx > 0:
            prev_dates = self._ts_dates[:self._ts_date_idx]
            prev_mask = color_df["side"].astype(str).isin(prev_dates)
            prev_df = color_df[prev_mask]
            if len(prev_df) > 0:
                prev_df = prev_df.reset_index(drop=True)
                kw = dict(s=self.style.point_size * 0.6, alpha=0.15,
                         c=[(0.5, 0.5, 0.5, 0.3)], label=f"prev ({len(prev_df)})")
                if is_3d:
                    sc = ax.scatter(prev_df["C1"], prev_df["C2"], prev_df["C3"], depthshade=True, **kw)
                else:
                    sc = ax.scatter(prev_df["C1"], prev_df["C2"], **kw)
                self.left_scatter_lookup.append((sc, prev_df, is_3d))

        # Draw current date in bright color
        cur_mask = color_df["side"].astype(str) == current_date
        cur_df = color_df[cur_mask]
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
            f"📅 Step {self._ts_date_idx+1}/{len(self._ts_dates)}: "
            f"{current_date} ({len(cur_df):,} pts)"
        )

    # ── Time-series RIGHT view (simulation results) ───────────

    def _ts_draw_right(self) -> None:
        """Draw right view: simulation results if available."""
        self.right_figure.clf()
        is_3d = self.dimension_var.get() == "3D"
        n_dim = 3 if is_3d else 2
        method = self.method_var.get()

        self.right_ax = self.right_figure.add_subplot(111, projection="3d" if is_3d else None)
        ax = self.right_ax
        ax.set_xlabel("C1"); ax.set_ylabel("C2")
        if is_3d:
            ax.set_zlabel("C3")

        if not self._ts_sim_result:
            ax.set_title("Date-Sequential Simulation", fontsize=11, pad=12)
            msg = "시뮬레이션을 실행하세요"
            if is_3d:
                ax.text2D(0.5, 0.5, msg, transform=ax.transAxes,
                          ha="center", va="center", fontsize=11, color="gray")
            else:
                ax.text(0.5, 0.5, msg, transform=ax.transAxes,
                        ha="center", va="center", fontsize=11, color="gray")
            self.right_canvas.draw()
            return

        # Get color data + embedding
        sel_colors = self._get_selected_colors()
        color_mask = self.df["color"].astype(str).isin(sel_colors)
        color_df = self.df[color_mask]
        try:
            if method == "UMAP" and n_dim in self._umap_cache:
                emb = self._umap_cache[n_dim][color_mask.values]
            else:
                emb = compute_embedding(
                    color_df, self.vector_columns,
                    method=method, n_dim=n_dim, standardize=self.standardize,
                )
        except Exception:
            self.right_canvas.draw()
            return

        color_df = color_df.copy()
        color_df["C1"] = emb[:, 0]
        color_df["C2"] = emb[:, 1]
        color_df["C3"] = emb[:, 2] if n_dim >= 3 else 0.0

        sim = self._ts_sim_result
        right_mode = self._ts_right_mode_var.get()
        cmap = plt.get_cmap("coolwarm")
        n_dates = len(sim["per_date"])

        if right_mode == "step":
            # Step-by-step right view — show accepted up to current right date
            rd_idx = min(self._ts_right_date_idx, n_dates - 1)
            dates_list = list(sim["per_date"].keys())
            current_rd = dates_list[rd_idx] if dates_list else ""

            # Show gray for previous dates' accepted pts
            total_shown = 0
            for i, (date, info) in enumerate(sim["per_date"].items()):
                if i > rd_idx:
                    break
                accepted_idx = info["accepted_indices"]
                if not accepted_idx:
                    continue
                date_mask2 = color_df.index.isin(accepted_idx)
                acc_df = color_df[date_mask2].reset_index(drop=True)
                if len(acc_df) == 0:
                    continue
                total_shown += len(acc_df)
                if i < rd_idx:
                    kw = dict(s=self.style.point_size * 0.7, alpha=0.2,
                             c=[(0.5, 0.5, 0.5, 0.3)])
                else:
                    c = cmap(i / max(n_dates - 1, 1))
                    kw = dict(s=self.style.point_size * 1.3, alpha=0.9, c=[c],
                             label=f"{date}: {info['accepted']}/{info['total']} ({info['rate']:.0%})",
                             edgecolors="white", linewidths=0.5)
                if is_3d:
                    ax.scatter(acc_df["C1"], acc_df["C2"], acc_df["C3"], depthshade=True, **kw)
                else:
                    ax.scatter(acc_df["C1"], acc_df["C2"], **kw)

            ax.set_title(f"Sim Step: {current_rd} ({total_shown} pts)",
                         fontsize=11, pad=12)
        else:
            # Show all dates
            total_accepted = 0
            for i, (date, info) in enumerate(sim["per_date"].items()):
                accepted_idx = info["accepted_indices"]
                if not accepted_idx:
                    continue
                date_mask2 = color_df.index.isin(accepted_idx)
                acc_df = color_df[date_mask2].reset_index(drop=True)
                if len(acc_df) == 0:
                    continue
                total_accepted += len(acc_df)
                c = cmap(i / max(n_dates - 1, 1))
                lab = f"{date}: {info['accepted']}/{info['total']} ({info['rate']:.0%})"
                kw = dict(s=self.style.point_size * 1.2, alpha=0.8, c=[c], label=lab,
                         edgecolors="white", linewidths=0.3)
                if is_3d:
                    ax.scatter(acc_df["C1"], acc_df["C2"], acc_df["C3"], depthshade=True, **kw)
                else:
                    ax.scatter(acc_df["C1"], acc_df["C2"], **kw)

            strat = sim.get("strategy", "")
            ax.set_title(f"Sim [{strat}]: {total_accepted} accepted",
                         fontsize=11, pad=12)

        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, fontsize=7)
        self.right_figure.tight_layout()
        self.right_canvas.draw()

    # ── Time-series simulation ────────────────────────────────

    def _ts_run_simulation(self) -> None:
        """Run date-sequential simulation using selected strategy."""
        if not self._ts_dates:
            return

        sel_colors = self._get_selected_colors()
        if not sel_colors:
            return

        strategy_name = self._ts_strategy_var.get()
        params = self._get_ts_strategy_params()

        color_mask = self.df["color"].astype(str).isin(sel_colors)
        color_df = self.df[color_mask]

        if len(color_df) < 2:
            self._ts_metrics_var.set("Not enough data")
            return

        self._ts_sim_button.config(state="disabled")
        self._ts_progress_var.set(0)
        self.root.update_idletasks()

        vectors = color_df[self.vector_columns].to_numpy(dtype=np.float64)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        normalized = vectors / norms

        idx_to_pos = {idx: pos for pos, idx in enumerate(color_df.index)}

        accepted_vecs: list[np.ndarray] = []
        accepted_indices: list[int] = []
        per_date: dict = {}
        total_processed = 0
        total_all = len(color_df)

        for date in self._ts_dates:
            date_mask = color_df["side"].astype(str) == date
            date_indices = color_df[date_mask].index.tolist()
            date_accepted = []

            for idx in date_indices:
                pos = idx_to_pos[idx]
                vec = normalized[pos]
                total_processed += 1

                if not accepted_vecs:
                    accepted_vecs.append(vec)
                    accepted_indices.append(idx)
                    date_accepted.append(idx)
                else:
                    if StreamingSimulator._should_accept(vec, accepted_vecs, strategy_name, params):
                        accepted_vecs.append(vec)
                        accepted_indices.append(idx)
                        date_accepted.append(idx)

                if total_processed % 100 == 0:
                    self._ts_progress_var.set(100.0 * total_processed / total_all)
                    self.root.update_idletasks()

            n_date = len(date_indices)
            n_acc = len(date_accepted)
            per_date[date] = {
                "total": n_date,
                "accepted": n_acc,
                "rate": n_acc / max(n_date, 1),
                "accepted_indices": date_accepted,
            }

        self._ts_progress_var.set(100)
        self._ts_sim_button.config(state="normal")

        # Compute metrics
        n_accepted = len(accepted_indices)
        if n_accepted >= 2:
            acc_arr = np.array(accepted_vecs)
            sample_n = min(200, n_accepted)
            if sample_n < n_accepted:
                sample_idx = np.random.choice(n_accepted, sample_n, replace=False)
                sample = acc_arr[sample_idx]
            else:
                sample = acc_arr
            sims = sample @ sample.T
            np.fill_diagonal(sims, -1)
            nn_dists = 1.0 - np.max(sims, axis=1)
            min_nn = float(np.min(nn_dists))
            mean_nn = float(np.mean(nn_dists))
        else:
            min_nn = 0.0
            mean_nn = 0.0

        self._ts_sim_result = {
            "strategy": strategy_name,
            "params": params,
            "per_date": per_date,
            "total_accepted": n_accepted,
            "total_processed": total_processed,
            "min_nn_dist": min_nn,
            "mean_nn_dist": mean_nn,
        }

        self._ts_right_date_idx = 0
        if self._ts_dates:
            self._ts_right_date_label_var.set(
                f"{self._ts_dates[0]}  (1/{len(self._ts_dates)})"
            )

        # Format metrics text
        lines = [f"═══ {strategy_name} ═══\n"]
        for k, v in params.items():
            lines.append(f"  {k}={v}")
        lines.append(f"\n{'Date':<10} {'In':>5} {'Accept':>6} {'Rate':>6}")
        lines.append("─" * 30)
        for date, info in per_date.items():
            lines.append(f"{date:<10} {info['total']:>5} {info['accepted']:>6} {info['rate']:>5.0%}")
        lines.append("─" * 30)
        lines.append(f"Total: {n_accepted}/{total_processed} ({n_accepted/max(total_processed,1):.1%})")
        lines.append(f"Min NN Dist:  {min_nn:.6f}")
        lines.append(f"Mean NN Dist: {mean_nn:.6f}")

        self._ts_metrics_var.set("\n".join(lines))
        self._ts_draw_right()


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
