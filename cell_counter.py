"""
Cell Counter - Pacagen Assignment 1
====================================
Counts bright blue cells in fluorescence microscopy images using:
  1. Resize to 800 px     -> ~26x speedup on 4128x2808 source images
  2. High-pass filtering  -> removes large dim background blobs
  3. Watershed            -> splits touching/clustered cells
  4. Area filter          -> 20-3000 px² at full res, scaled automatically

Speed notes
-----------
Source images are 4128x2808 (~11.6 MP).
Resizing to 800 px before all operations gives ~26x fewer pixels,
reducing per-image time from ~2 min to ~3-8 s.
Annotation is written with cv2 instead of matplotlib (~10x faster).

Usage:
    python cell_counter.py --input assignment_1/images/ --output results/

Output:
    results/
        annotated/          <- side-by-side JPEGs per input image
        cell_counts.csv     <- well name + count + image path
        plate_heatmap.png   <- heatmap of counts across plate
"""

import cv2
import numpy as np
import re
import csv
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path


# ─────────────────────────────────────────
#  Resolution settings
# ─────────────────────────────────────────

MAX_PROCESS_DIM  = 800    # longest edge for detection  (~26x speedup on 4 k images)
ANNOTATE_MAX_DIM = 1200   # longest edge for saved annotations


def _resize(img_bgr, max_dim):
    """Return (resized_img, scale).  scale <= 1.0."""
    h, w = img_bgr.shape[:2]
    scale = min(max_dim / max(h, w), 1.0)
    if scale < 1.0:
        nw, nh = int(w * scale), int(h * scale)
        return cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA), scale
    return img_bgr.copy(), 1.0


# ─────────────────────────────────────────
#  Core Detection
# ─────────────────────────────────────────

def detect_cells(img_bgr, **_kwargs):
    """
    Detect cells in a fluorescence image.

    Steps
    -----
    1. Resize to MAX_PROCESS_DIM for speed.
    2. High-pass filter (blue channel) — suppresses large smooth blobs.
    3. Binary threshold + morphological opening.
    4. Distance transform → local maxima seeds.
    5. Watershed — splits touching clusters.
    6. Area filter (scaled to processed resolution).
    7. Return centres in *original* image pixel coordinates.

    Parameters
    ----------
    img_bgr : BGR uint8 array  (any resolution)
    **_kwargs : ignored (for API compatibility)

    Returns
    -------
    centers : list of (cx, cy, area)   — original image coords
    debug   : dict of intermediate arrays
    """
    # ── 1. Resize ─────────────────────────────────────────────────────────
    img_small, scale = _resize(img_bgr, MAX_PROCESS_DIM)
    b, g, r = cv2.split(img_small)
    b_float = b.astype(np.float32)

    # ── 2. High-pass filter ───────────────────────────────────────────────
    b_lowpass  = cv2.GaussianBlur(b_float, (51, 51), 0)
    b_highpass = np.clip(b_float - b_lowpass + 128, 0, 255).astype(np.uint8)

    # ── 3. Threshold + morphological opening ─────────────────────────────
    _, hp_thresh = cv2.threshold(b_highpass, 148, 255, cv2.THRESH_BINARY)
    kernel       = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    hp_clean     = cv2.morphologyEx(hp_thresh, cv2.MORPH_OPEN, kernel)

    # ── 4. Distance transform → local maxima seeds ───────────────────────
    dist = cv2.distanceTransform(hp_clean, cv2.DIST_L2, 5)

    # r_kernel and distance threshold scaled to processed resolution
    r_kernel  = max(2, int(round(8 * scale)))          # ≈ 2 at 800 px
    dist_thr  = max(1.0, 4.0 * scale)                  # ≈ 1.0 at 800 px
    pk        = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                          (r_kernel * 2 - 1, r_kernel * 2 - 1))
    dist_dil  = cv2.dilate(dist, pk)
    local_max = ((dist == dist_dil) & (dist > dist_thr)).astype(np.uint8)

    sk             = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    local_max_dil  = cv2.dilate(local_max, sk)
    _, seed_labels = cv2.connectedComponents(local_max_dil)

    # ── 5. Watershed ──────────────────────────────────────────────────────
    markers            = seed_labels.copy().astype(np.int32)
    markers[hp_clean == 0] = 0
    ws = cv2.watershed(cv2.merge([b, b, b]), markers)

    # ── 6. Collect centroids, filter by scaled area ───────────────────────
    scale2   = scale * scale
    area_min = max(3,   int(20   * scale2))   # ≈  1 → floor at 3
    area_max = max(150, int(3000 * scale2))   # ≈ 113 → floor at 150

    unique_labels = np.unique(ws)
    unique_labels = unique_labels[unique_labels > 1]

    centers = []
    for lbl in unique_labels:
        mask = ws == lbl
        area = int(mask.sum())
        if area < area_min or area > area_max:
            continue
        ys, xs = np.where(mask)
        # ── 7. Scale back to original image coordinates ───────────────────
        cx = float(xs.mean()) / scale
        cy = float(ys.mean()) / scale
        centers.append((cx, cy, area))

    debug = dict(
        highpass=b_highpass, thresh=hp_thresh, cleaned=hp_clean,
        dist=dist, watershed=ws, scale=scale,
        # kept for API compatibility with app.py
        well_mask=None, well_contour=None, well_warning=None,
    )
    return centers, debug


# ─────────────────────────────────────────
#  Visualisation  (cv2-based for speed)
# ─────────────────────────────────────────

def annotate_image(img_bgr, centers, well_contour=None, draw_scale=1.0):
    """
    Draw lime-green circles on cells.
    draw_scale: multiply cx/cy by this to convert to display pixel coords.
    Returns a BGR array.
    """
    vis = img_bgr.copy()
    r = max(4, int(12 * draw_scale))
    t = max(1, int(2  * draw_scale))
    for cx, cy, _ in centers:
        cv2.circle(vis, (int(cx * draw_scale), int(cy * draw_scale)), r,
                   (0, 255, 0), t)
    return vis


def save_annotated(img_bgr, centers, out_path, well_name, debug=None):
    """
    Save a side-by-side original / annotated JPEG using cv2 (~10x faster
    than matplotlib).  The image is first shrunk to ANNOTATE_MAX_DIM.
    """
    well_warning = (debug or {}).get('well_warning')

    # Resize to display resolution
    orig_disp, disp_scale = _resize(img_bgr, ANNOTATE_MAX_DIM)

    # Draw annotations at display scale (centres are in original coords)
    annot_disp = annotate_image(orig_disp, centers, draw_scale=disp_scale)

    # Side-by-side composite
    composite = np.hstack([orig_disp, annot_disp])

    # Text overlay
    label = f"Well: {well_name}  |  {len(centers)} cells"
    if well_warning:
        label += f"  |  {well_warning}"
    cv2.putText(composite, label, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imwrite(str(out_path), composite, [cv2.IMWRITE_JPEG_QUALITY, 88])


# ─────────────────────────────────────────
#  Plate-map helpers
# ─────────────────────────────────────────

ROWS = list("ABCDEFGHIJKLMNOP")   # A-P  (16 rows)
COLS = list(range(1, 25))          # 1-24 (24 cols)


def well_to_rc(well_name):
    """'B6' -> (row_idx, col_idx)  0-indexed."""
    m = re.match(r'^([A-P])(\d+)$', well_name.strip().upper())
    if not m:
        return None, None
    return ROWS.index(m.group(1)), int(m.group(2)) - 1


def build_heatmap(counts_dict, out_path):
    """Save a 384-well plate heatmap PNG annotated with counts."""
    grid = np.full((16, 24), np.nan)
    for well, cnt in counts_dict.items():
        r, c = well_to_rc(well)
        if r is not None:
            grid[r, c] = cnt

    fig, ax = plt.subplots(figsize=(18, 8))
    im = ax.imshow(grid, aspect='auto', cmap='YlOrRd',
                   vmin=np.nanmin(grid), vmax=np.nanmax(grid))
    plt.colorbar(im, ax=ax, label='Cell count')

    ax.set_xticks(range(24)); ax.set_xticklabels(range(1, 25), fontsize=7)
    ax.set_yticks(range(16)); ax.set_yticklabels(ROWS, fontsize=8)
    ax.set_xlabel("Column"); ax.set_ylabel("Row")
    ax.set_title("Cell Count Heatmap — 384-well Plate",
                 fontsize=14, fontweight='bold')

    for r in range(16):
        for c in range(24):
            if not np.isnan(grid[r, c]):
                ax.text(c, r, str(int(grid[r, c])),
                        ha='center', va='center', fontsize=5, color='black')

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Heatmap saved -> {out_path}")


# ─────────────────────────────────────────
#  Batch processing
# ─────────────────────────────────────────

def parse_well_name(filename):
    """'B6_01_FL3_merge.jpg' -> 'B6'"""
    m = re.match(r'^([A-P]\d+)_', Path(filename).name, re.IGNORECASE)
    return m.group(1).upper() if m else Path(filename).stem


def process_folder(input_dir, output_dir, debug_well=False):
    input_dir  = Path(input_dir)
    output_dir = Path(output_dir)
    annot_dir  = output_dir / "annotated"
    annot_dir.mkdir(parents=True, exist_ok=True)

    extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}
    image_files = sorted([f for f in input_dir.iterdir()
                          if f.suffix.lower() in extensions])
    if not image_files:
        print(f"No images found in {input_dir}")
        return

    print(f"Found {len(image_files)} images  "
          f"(processing at {MAX_PROCESS_DIM} px, annotating at {ANNOTATE_MAX_DIM} px)")

    counts_dict = {}
    csv_rows    = []

    for i, img_path in enumerate(image_files, 1):
        well = parse_well_name(img_path.name)
        print(f"  [{i:3d}/{len(image_files)}]  {img_path.name}  (well={well})",
              end='', flush=True)

        img = cv2.imread(str(img_path))
        if img is None:
            print("  <- FAILED to load, skipping")
            continue

        centers, debug = detect_cells(img)
        count = len(centers)
        counts_dict[well] = count
        print(f"  ->  {count} cells")

        annot_path = annot_dir / (img_path.stem + "_annotated.jpg")
        save_annotated(img, centers, str(annot_path), well, debug=debug)

        csv_rows.append({
            'well':       well,
            'cell_count': count,
            'image_file': img_path.name,
            'annotated':  annot_path.name,
        })

    # Write CSV
    csv_path = output_dir / "cell_counts.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f,
                    fieldnames=['well', 'cell_count', 'image_file', 'annotated'])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\nCSV saved -> {csv_path}")

    # Heatmap
    if counts_dict:
        build_heatmap(counts_dict, str(output_dir / "plate_heatmap.png"))

    print(f"\nDone. {len(csv_rows)} wells processed.")
    print(f"Results in: {output_dir.resolve()}")


# ─────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Count fluorescent cells in microscopy images.')
    parser.add_argument('--input',  '-i', default='assignment_1/images',
                        help='Input image folder (default: assignment_1/images/)')
    parser.add_argument('--output', '-o', default='results',
                        help='Output folder    (default: results/)')
    args = parser.parse_args()

    process_folder(args.input, args.output)
