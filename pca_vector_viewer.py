#!/usr/bin/env python3
"""
CSV vector viewer with PCA / UMAP 2D/3D visualization.

Supports two CSV schemas:
  1. "vector" single column: type, side, color, path, vector
     (header has 5 columns, data rows have 260 comma-separated fields)
  2. Separate vector columns: type, side, color, path, v_0, ..., v_255

On startup the user can optionally filter by side and downsample.
In the GUI, dimensionality reduction (PCA or UMAP) is recalculated
every time the user clicks "Redraw", using only the currently
filtered subset.

Example:
    python pca_vector_viewer.py --csv /path/to/vectors.csv
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import tkinter as tk
from dataclasses import dataclass
from tkinter import filedialog, messagebox, ttk

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cosine as cosine_distance

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


# ─────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize 256-dim vectors from CSV using PCA/UMAP in 2D/3D."
    )
    parser.add_argument("--csv", type=str, default="", help="Path to input CSV file.")
    parser.add_argument("--point-size", type=float, default=26.0, help="Scatter point size.")
    parser.add_argument("--alpha", type=float, default=0.8, help="Scatter point alpha.")
    parser.add_argument("--no-standardize", action="store_true", help="Disable standardization.")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────
# CSV scanning & loading  (NO dimensionality reduction here)
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
    """Load CSV, validate, filter, downsample. Returns (df, vector_columns, skipped).

    Unlike before, this does NOT run PCA/UMAP — raw vectors are preserved.
    """
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
# Embedding computation  (called on every Redraw)
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
# GUI Viewer
# ─────────────────────────────────────────────────────────────

@dataclass
class PlotStyle:
    point_size: float
    alpha: float


class PCAVectorViewer:
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

        self.side_palette = build_palette(self.side_values)
        self.color_palette = build_palette(self.color_values)

        self.figure = plt.Figure(figsize=(9, 7), dpi=100, facecolor="white")

        self.ax = None
        self.annotation = None
        self.scatter_lookup: list[tuple[object, pd.DataFrame, bool]] = []
        self.hover_cid = None
        self.leave_cid = None
        self.canvas = None
        self.canvas_widget = None

        # UMAP cache: computed once on full dataset, reused on filter changes
        # Keys: n_dim (2 or 3) -> np.ndarray of shape (len(df), n_dim)
        self._umap_cache: dict[int, np.ndarray] = {}

        # Click mode: "viewer" or "distance"
        self._click_mode_var = tk.StringVar(value="viewer")

        # Distance measurement state
        self._selected_points: list[pd.Series] = []  # up to 2 selected rows
        self._selected_markers: list = []  # matplotlib marker artists
        self._distance_line = None  # dashed line artist
        self._distance_var = tk.StringVar(value="두 점을 클릭하여 거리를 측정하세요")

        self._build_ui()
        self._bind_filter_events()
        # Defer first draw
        self.root.after(200, self.redraw)

    # ── UI ───────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self.root.title("PCA / UMAP Vector Viewer")
        self.root.geometry("1500x900")

        outer = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        outer.pack(fill=tk.BOTH, expand=True)

        control_outer = ttk.Frame(outer, padding=0)
        self.plot_frame = ttk.Frame(outer, padding=8)
        outer.add(control_outer, weight=1)
        outer.add(self.plot_frame, weight=4)

        # Scrollable control panel
        control_canvas = tk.Canvas(control_outer, highlightthickness=0)
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
        ttk.Label(control_frame, text="Vector Viewer", font=("Arial", 14, "bold")).pack(anchor="w", pady=(0, 8))
        ttk.Label(control_frame, text=f"CSV: {self.csv_path}", wraplength=340, justify="left").pack(anchor="w", pady=(0, 6))
        summary = (
            f"Total rows (valid): {self.total_loaded:,}\n"
            f"Skipped rows: {self.skipped:,}\n"
            f"Sides: {len(self.side_values)}, Colors: {len(self.color_values)}"
        )
        ttk.Label(control_frame, text=summary, justify="left").pack(anchor="w", pady=(0, 12))

        # ── Method (PCA / UMAP) ──
        method_box = ttk.LabelFrame(control_frame, text="Method", padding=8)
        method_box.pack(fill=tk.X, pady=(0, 10))
        ttk.Radiobutton(method_box, text="PCA", value="PCA", variable=self.method_var).pack(anchor="w")
        umap_label = "UMAP" if HAS_UMAP else "UMAP (not installed)"
        umap_state = "normal" if HAS_UMAP else "disabled"
        ttk.Radiobutton(method_box, text=umap_label, value="UMAP", variable=self.method_var,
                        state=umap_state).pack(anchor="w")

        # ── Dimension ──
        dim_box = ttk.LabelFrame(control_frame, text="Dimension", padding=8)
        dim_box.pack(fill=tk.X, pady=(0, 10))
        ttk.Radiobutton(dim_box, text="2D", value="2D", variable=self.dimension_var).pack(anchor="w")
        ttk.Radiobutton(dim_box, text="3D", value="3D", variable=self.dimension_var).pack(anchor="w")

        # ── Color Mode ──
        cmode_box = ttk.LabelFrame(control_frame, text="Color Mapping", padding=8)
        cmode_box.pack(fill=tk.X, pady=(0, 10))
        ttk.Radiobutton(cmode_box, text="Color by color", value="color", variable=self.color_mode_var).pack(anchor="w")
        ttk.Radiobutton(cmode_box, text="Color by side", value="side", variable=self.color_mode_var).pack(anchor="w")

        # ── Path Filter ──
        search_box = ttk.LabelFrame(control_frame, text="Path Filter", padding=8)
        search_box.pack(fill=tk.X, pady=(0, 10))
        ttk.Entry(search_box, textvariable=self.path_filter_var).pack(fill=tk.X)
        ttk.Label(search_box, text="Only show rows where path contains this text.",
                  wraplength=320, justify="left").pack(anchor="w", pady=(6, 0))

        # ── Click Mode ──
        mode_box = ttk.LabelFrame(control_frame, text="🖱 Click Mode", padding=8)
        mode_box.pack(fill=tk.X, pady=(0, 10))
        ttk.Radiobutton(mode_box, text="🖼 Image Viewer", value="viewer",
                        variable=self._click_mode_var,
                        command=self._on_mode_change).pack(anchor="w")
        ttk.Radiobutton(mode_box, text="📏 Distance Measurement", value="distance",
                        variable=self._click_mode_var,
                        command=self._on_mode_change).pack(anchor="w")
        self._mode_hint_label = ttk.Label(
            mode_box, text="클릭하면 이미지를 엽니다.",
            wraplength=320, justify="left", foreground="gray"
        )
        self._mode_hint_label.pack(anchor="w", pady=(6, 0))

        # ── Buttons ──
        btn_box = ttk.Frame(control_frame)
        btn_box.pack(fill=tk.X, pady=(0, 10))

        redraw_btn = ttk.Button(btn_box, text="★ Redraw (recalculate)", command=self.redraw)
        redraw_btn.pack(fill=tk.X, pady=(0, 6))

        ttk.Label(btn_box, text="PCA: Redraw로 재계산\nUMAP: 필터 변경 시 자동 업데이트",
                  wraplength=320, justify="left", foreground="gray").pack(anchor="w", pady=(0, 6))

        ttk.Button(btn_box, text="Select all sides", command=self._select_all_sides).pack(fill=tk.X, pady=1)
        ttk.Button(btn_box, text="Clear all sides", command=self._clear_all_sides).pack(fill=tk.X, pady=1)
        ttk.Button(btn_box, text="Select all colors", command=self._select_all_colors).pack(fill=tk.X, pady=1)
        ttk.Button(btn_box, text="Clear all colors", command=self._clear_all_colors).pack(fill=tk.X, pady=1)

        # ── Side Filter ──
        side_box = ttk.LabelFrame(control_frame, text="Side Filter", padding=8)
        side_box.pack(fill=tk.X, pady=(0, 10))
        for v in self.side_values:
            ttk.Checkbutton(side_box, text=v, variable=self.side_vars[v]).pack(anchor="w")

        # ── Color Filter ──
        color_box = ttk.LabelFrame(control_frame, text="Color Filter", padding=8)
        color_box.pack(fill=tk.X, pady=(0, 10))
        for v in self.color_values:
            ttk.Checkbutton(color_box, text=v, variable=self.color_vars[v]).pack(anchor="w")

        # ── Distance Result Panel ──
        self._dist_frame = ttk.LabelFrame(control_frame, text="📏 Distance Result", padding=8)
        self._dist_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(self._dist_frame, textvariable=self._distance_var, wraplength=340,
                  justify="left", font=("Consolas", 9)).pack(fill=tk.X, pady=(0, 4))
        ttk.Button(self._dist_frame, text="Clear Selection", command=self._clear_distance_selection).pack(fill=tk.X)
        # Initially hidden (viewer mode)
        self._dist_frame.pack_forget()

        # ── Status ──
        status_frame = ttk.LabelFrame(control_frame, text="Current View", padding=8)
        status_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(status_frame, textvariable=self.status_var, wraplength=340, justify="left").pack(fill=tk.X)

        # matplotlib canvas + toolbar
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

        # Navigation toolbar (zoom, pan, home, save)
        toolbar = NavigationToolbar2Tk(self.canvas, self.plot_frame)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)

    # ── helpers ──────────────────────────────────────────────

    def _bind_filter_events(self):
        """When filters change and UMAP is active, auto-update the plot."""
        def _on_filter_change(*_):
            if self.method_var.get() == "UMAP" and self._umap_cache:
                self._draw_from_umap_cache()

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

    # ── hover (tkinter tooltip window) ────────────────────────

    def _connect_hover(self):
        if self.hover_cid is not None:
            self.canvas.mpl_disconnect(self.hover_cid)
        if self.leave_cid is not None:
            self.canvas.mpl_disconnect(self.leave_cid)
        if hasattr(self, '_click_cid') and self._click_cid is not None:
            self.canvas.mpl_disconnect(self._click_cid)
        self.hover_cid = self.canvas.mpl_connect("motion_notify_event", self._on_hover)
        self.leave_cid = self.canvas.mpl_connect("figure_leave_event", self._on_leave)
        self._click_cid = self.canvas.mpl_connect("button_press_event", self._on_click)

    def _build_annotation(self):
        """Create a tkinter Toplevel tooltip (hidden by default)."""
        if hasattr(self, "_tooltip") and self._tooltip is not None:
            self._tooltip.destroy()
        self._tooltip = tk.Toplevel(self.root)
        self._tooltip.wm_overrideredirect(True)  # no title bar
        self._tooltip.withdraw()  # start hidden

        self._tooltip_label = tk.Label(
            self._tooltip,
            text="",
            justify="left",
            background="#ffffee",
            foreground="#333333",
            relief="solid",
            borderwidth=1,
            font=("Consolas", 9),
            padx=8,
            pady=6,
        )
        self._tooltip_label.pack()
        self.annotation = True  # flag so callers don't break

    def _show_annotation(self, row, event):
        if not hasattr(self, "_tooltip") or self._tooltip is None:
            return

        path_str = str(row["path"])
        text = (
            f"type : {row['type']}\n"
            f"side : {row['side']}\n"
            f"color: {row['color']}\n"
            f"path : {path_str}"
        )
        self._tooltip_label.config(text=text)

        # Position near the mouse cursor
        x_root = self.root.winfo_pointerx() + 18
        y_root = self.root.winfo_pointery() + 18
        self._tooltip.wm_geometry(f"+{x_root}+{y_root}")
        self._tooltip.deiconify()
        self._tooltip.lift()

    def _hide_annotation(self):
        if hasattr(self, "_tooltip") and self._tooltip is not None:
            self._tooltip.withdraw()

    def _on_hover(self, event):
        if self.ax is None or event.inaxes != self.ax:
            self._hide_annotation()
            return
        for scatter, gdf, _ in self.scatter_lookup:
            ok, info = scatter.contains(event)
            if ok and info.get("ind", []):
                self._show_annotation(gdf.iloc[int(info["ind"][0])], event)
                return
        self._hide_annotation()

    def _on_leave(self, _):
        self._hide_annotation()

    def _on_click(self, event):
        """Click on a point — behavior depends on current click mode."""
        if self.ax is None or event.inaxes != self.ax:
            return

        mode = self._click_mode_var.get()

        for scatter, gdf, _ in self.scatter_lookup:
            ok, info = scatter.contains(event)
            if ok and info.get("ind", []):
                row = gdf.iloc[int(info["ind"][0])]
                if mode == "distance":
                    self._select_point_for_distance(row, event)
                else:
                    self._open_image(row)
                return

    def _on_mode_change(self) -> None:
        """Called when click mode radio button changes."""
        mode = self._click_mode_var.get()
        if mode == "viewer":
            self._mode_hint_label.config(text="클릭하면 이미지를 엽니다.")
            self._dist_frame.pack_forget()
            self._clear_distance_selection()
        else:
            self._mode_hint_label.config(text="두 점을 순서대로 클릭하면\n코사인 거리를 계산합니다.")
            # Show distance panel (insert before Status)
            self._dist_frame.pack(fill=tk.X, pady=(0, 10))

    def _open_image(self, row) -> None:
        """Open an image file in a new window with metadata."""
        path = str(row["path"])

        if not os.path.isfile(path):
            messagebox.showinfo("File not found", f"Image file not found:\n{path}")
            return

        if not HAS_PIL:
            messagebox.showinfo("Pillow missing", f"Install Pillow to view images:\npip install Pillow\n\nPath: {path}")
            return

        try:
            img = Image.open(path)
        except Exception as exc:
            messagebox.showerror("Image error", f"Cannot open image:\n{path}\n\n{exc}")
            return

        # Resize if too large, keeping aspect ratio
        max_size = 800
        w, h = img.size
        if w > max_size or h > max_size:
            ratio = min(max_size / w, max_size / h)
            img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)

        win = tk.Toplevel(self.root)
        win.title(os.path.basename(path))
        win.geometry(f"{img.size[0]+20}x{img.size[1]+100}")

        # Image
        tk_img = ImageTk.PhotoImage(img)
        label = tk.Label(win, image=tk_img)
        label.image = tk_img  # prevent GC
        label.pack(padx=5, pady=5)

        # Metadata below the image
        meta_text = (
            f"path  : {path}\n"
            f"type  : {row['type']}\n"
            f"side  : {row['side']}\n"
            f"color : {row['color']}"
        )
        tk.Label(win, text=meta_text, font=("Consolas", 9), fg="#444444",
                 justify="left", anchor="w").pack(fill=tk.X, padx=10, pady=(0, 8))

    # ── Distance measurement ─────────────────────────────────

    def _select_point_for_distance(self, row: pd.Series, event) -> None:
        """Select a point for cosine distance measurement (Shift+Click)."""
        if len(self._selected_points) >= 2:
            # Reset: start new selection
            self._clear_distance_selection()

        self._selected_points.append(row)
        is_3d = self.dimension_var.get() == "3D"

        # Draw a red marker on the selected point
        if "C1" in row.index and "C2" in row.index:
            if is_3d and "C3" in row.index:
                marker = self.ax.scatter(
                    [row["C1"]], [row["C2"]], [row["C3"]],
                    s=200, c="red", marker="*", zorder=10,
                    edgecolors="darkred", linewidths=1.5,
                )
            else:
                marker = self.ax.scatter(
                    [row["C1"]], [row["C2"]],
                    s=200, c="red", marker="*", zorder=10,
                    edgecolors="darkred", linewidths=1.5,
                )
            self._selected_markers.append(marker)
            self.canvas.draw()

        n = len(self._selected_points)
        if n == 1:
            self._distance_var.set(
                f"Point A 선택됨\n"
                f"  type: {row['type']}\n"
                f"  side: {row['side']}\n"
                f"  color: {row['color']}\n"
                f"  path: {os.path.basename(str(row['path']))}\n\n"
                f"→ Shift+Click으로 Point B를 선택하세요"
            )
        elif n == 2:
            self._compute_and_show_distance()

    def _compute_and_show_distance(self) -> None:
        """Compute cosine distance between two selected points using original vectors."""
        row_a = self._selected_points[0]
        row_b = self._selected_points[1]

        vec_a = row_a[self.vector_columns].to_numpy(dtype=np.float64)
        vec_b = row_b[self.vector_columns].to_numpy(dtype=np.float64)

        # Cosine distance (1 - cosine_similarity), range [0, 2]
        cos_dist = cosine_distance(vec_a, vec_b)
        cos_sim = 1.0 - cos_dist

        # Euclidean distance
        euc_dist = np.linalg.norm(vec_a - vec_b)

        # Draw line between the two points
        is_3d = self.dimension_var.get() == "3D"
        if "C1" in row_a.index and "C1" in row_b.index:
            xs = [row_a["C1"], row_b["C1"]]
            ys = [row_a["C2"], row_b["C2"]]
            if is_3d and "C3" in row_a.index:
                zs = [row_a["C3"], row_b["C3"]]
                self.ax.plot(xs, ys, zs, color="red", linewidth=1.5,
                             linestyle="--", alpha=0.7, zorder=9)
            else:
                self.ax.plot(xs, ys, color="red", linewidth=1.5,
                             linestyle="--", alpha=0.7, zorder=9)
            self.canvas.draw()

        self._distance_var.set(
            f"═══ Distance Result ═══\n"
            f"\n"
            f"Cosine Distance : {cos_dist:.6f}\n"
            f"Cosine Similarity: {cos_sim:.6f}\n"
            f"Euclidean Distance: {euc_dist:.4f}\n"
            f"\n"
            f"── Point A ──\n"
            f"  type: {row_a['type']}\n"
            f"  side: {row_a['side']}, color: {row_a['color']}\n"
            f"  {os.path.basename(str(row_a['path']))}\n"
            f"\n"
            f"── Point B ──\n"
            f"  type: {row_b['type']}\n"
            f"  side: {row_b['side']}, color: {row_b['color']}\n"
            f"  {os.path.basename(str(row_b['path']))}"
        )

    def _clear_distance_selection(self) -> None:
        """Clear selected points and remove markers."""
        for marker in self._selected_markers:
            try:
                marker.remove()
            except Exception:
                pass
        self._selected_markers.clear()
        self._selected_points.clear()
        self._distance_var.set("두 점을 클릭하여 거리를 측정하세요")
        if self.canvas:
            self.canvas.draw()

    # ── Redraw (core method) ─────────────────────────────────

    def redraw(self) -> None:
        """Main redraw: recompute embedding then draw.

        - PCA: always recomputes on the filtered subset.
        - UMAP: computes on ALL data once, caches the result,
          then subsequent filter changes just use the cache.
        """
        # Clear distance selection on redraw
        self._selected_points.clear()
        self._selected_markers.clear()
        self._distance_var.set("두 점을 클릭하여 거리를 측정하세요")

        method = self.method_var.get()

        if method == "UMAP":
            # Compute UMAP on full dataset if not cached yet
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
            # PCA: recompute on the filtered subset
            self._draw_pca()

    def _draw_error(self, msg: str) -> None:
        self.figure.clf()
        is_3d = self.dimension_var.get() == "3D"
        self.ax = self.figure.add_subplot(111, projection="3d" if is_3d else None)
        if is_3d:
            self.ax.text2D(0.05, 0.95, msg, transform=self.ax.transAxes, fontsize=9, color="red")
        else:
            self.ax.text(0.05, 0.95, msg, transform=self.ax.transAxes, fontsize=9, color="red")
        self.status_var.set(msg)
        self.canvas.draw()

    def _draw_from_umap_cache(self) -> None:
        """Draw using cached UMAP embedding, filtering by current selections."""
        is_3d = self.dimension_var.get() == "3D"
        n_dim = 3 if is_3d else 2

        # If this dimension hasn't been cached yet, trigger full compute
        if n_dim not in self._umap_cache:
            self.redraw()
            return

        mask = self._get_filtered_indices()
        filtered = self.df[mask].copy()
        emb = self._umap_cache[n_dim][mask]

        filtered["C1"] = emb[:, 0]
        filtered["C2"] = emb[:, 1]
        filtered["C3"] = emb[:, 2] if n_dim >= 3 else 0.0

        self._draw_scatter(filtered, "UMAP", is_3d)

    def _draw_pca(self) -> None:
        """Compute PCA on the filtered subset and draw."""
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

        self._draw_scatter(filtered, "PCA", is_3d)

    def _draw_scatter(self, filtered: pd.DataFrame, method: str, is_3d: bool) -> None:
        """Common scatter plot drawing logic."""
        self.figure.clf()
        self.scatter_lookup = []

        self.ax = self.figure.add_subplot(111, projection="3d" if is_3d else None)
        ax = self.ax
        ax.set_xlabel("C1")
        ax.set_ylabel("C2")
        if is_3d:
            ax.set_zlabel("C3")

        self._build_annotation()
        visible_count = len(filtered)

        if visible_count < 2:
            ax.set_title(f"{method} — not enough points ({visible_count})", fontsize=12, pad=16)
            msg = "Need at least 2 data points."
            if is_3d:
                ax.text2D(0.05, 0.95, msg, transform=ax.transAxes)
            else:
                ax.text(0.05, 0.95, msg, transform=ax.transAxes)
            self.status_var.set(f"★ Displayed: {visible_count}\nTotal: {self.total_loaded:,}")
            self._connect_hover()
            self.canvas.draw()
            return

        ax.set_title(
            f"{method} ({('3D' if is_3d else '2D')})  —  {visible_count:,} points",
            pad=16, fontsize=12,
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
            self.scatter_lookup.append((sc, gdf, is_3d))

        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, fontsize=9)
        self.figure.tight_layout()

        self.status_var.set(
            f"★ Displayed: {visible_count:,}  ({method})\n"
            f"Total valid: {self.total_loaded:,}  |  Skipped: {self.skipped:,}\n"
            f"Sides: {filtered['side'].astype(str).value_counts().to_dict()}\n"
            f"Colors: {filtered['color'].astype(str).value_counts().to_dict()}"
        )
        self._connect_hover()
        self.canvas.draw()


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
    PCAVectorViewer(
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
