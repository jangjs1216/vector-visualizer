# Vector Visualizer

CSV vector viewer with **PCA / UMAP** 2D/3D visualization.

## Features

- **PCA & UMAP** dimensionality reduction (2D / 3D)
- Interactive scatter plot with **hover tooltips** and **click-to-open image**
- Side / Color filtering with checkboxes
- Path text filter
- Downsampling support at startup
- UMAP caching for fast filter updates

## CSV Schemas

Supports two CSV formats:

1. **"vector" single column**: `type, side, color, path, vector`
   (header has 5 columns, data rows have 260 comma-separated fields)
2. **Separate vector columns**: `type, side, color, path, v_0, ..., v_255`

## Requirements

```bash
pip install numpy pandas matplotlib scikit-learn Pillow umap-learn
```

## Usage

```bash
python pca_vector_viewer.py --csv /path/to/vectors.csv
```

### Options

| Argument | Default | Description |
|---|---|---|
| `--csv` | *(file dialog)* | Path to input CSV file |
| `--point-size` | `26.0` | Scatter point size |
| `--alpha` | `0.8` | Scatter point alpha |
| `--no-standardize` | `False` | Disable standardization |
