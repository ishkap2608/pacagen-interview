"""
Pacagen Cell Analysis Suite — Streamlit App
============================================
Two tabs:
  🔬  Cell Counter     — upload images, run the counter, download results
  📊  Plate Analysis   — detect plate effects in readings.xlsx and
                         in Assignment-1 counter results
"""

import io
import re
import sys
import os
import csv
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats as scipy_stats

sys.path.insert(0, str(Path(__file__).parent))
from cell_counter import detect_cells, annotate_image

# ─────────────────────────────────────────────────────────────────────────────
#  Page config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Pacagen Cell Analysis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    .stTabs [data-baseweb="tab"] { font-size: 1.05rem; font-weight: 600; }
    .metric-box {
        background: #f8f9fa; border-radius: 8px; padding: 14px 18px;
        border-left: 4px solid #0068c9;
    }
    .verdict-yes {
        background: #fff3cd; border-left: 4px solid #ffc107;
        border-radius: 8px; padding: 14px 18px;
    }
    .verdict-no {
        background: #d4edda; border-left: 4px solid #28a745;
        border-radius: 8px; padding: 14px 18px;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────────────────

ALL_ROWS   = list("ABCDEFGHIJKLMNOP")
ALL_COLS   = list(range(1, 25))
WATER_ROWS = ["B", "I", "O"]
VALID_COLS = list(range(2, 23))        # cols 2-22; outer 1 & 24 are empty, col 23 ignored
INNER_ROWS = [r for r in ALL_ROWS if r not in ("A", "P")]   # B-O

BASE_DIR   = Path(__file__).parent
A1_PLATEMAP_PATH  = BASE_DIR / "assignment_1" / "Plate_map.xlsx"
A2_READINGS_PATH  = BASE_DIR / "assignment_2" / "readings.xlsx"
A1_RESULTS_PATH   = BASE_DIR / "results" / "cell_counts.csv"

# ─────────────────────────────────────────────────────────────────────────────
#  Plate parsing helpers
# ─────────────────────────────────────────────────────────────────────────────

def parse_plate_grid(source) -> pd.DataFrame:
    """
    Read the first sheet of a readings-style xlsx (raw layout, no header)
    into a clean 16×24 DataFrame indexed by row letter (A-P), cols 1-24.
    """
    df_raw = pd.read_excel(source, sheet_name=0, header=None)
    grid = pd.DataFrame(np.nan, index=ALL_ROWS, columns=ALL_COLS, dtype=float)
    for i, row_letter in enumerate(ALL_ROWS):
        for j, col_num in enumerate(ALL_COLS):
            try:
                val = df_raw.iloc[i + 1, j + 1]
                grid.loc[row_letter, col_num] = float(val)
            except (TypeError, ValueError, IndexError):
                pass
    return grid


def parse_platemap_labels(source) -> pd.DataFrame:
    """
    Parse a plate-map xlsx into a 16×24 DataFrame of string labels
    (e.g. 'Molecule-A 20 µM', 'Water', or empty string).
    """
    df_raw = pd.read_excel(source, sheet_name=0, header=None)
    labels = pd.DataFrame("", index=ALL_ROWS, columns=ALL_COLS, dtype=str)

    for i, row_letter in enumerate(ALL_ROWS):
        raw_row = df_raw.iloc[i + 1]
        col_idx = 1   # plate col 1
        j = 1         # df column index
        while j < len(raw_row):
            val = raw_row.iloc[j]
            if isinstance(val, str) and val.strip().lower() == "water":
                # Fill entire row with Water (triplicate groups)
                for c in ALL_COLS[1:-1]:   # cols 2-23
                    labels.loc[row_letter, c] = "Water"
                break
            elif isinstance(val, str) and val.strip().startswith("Molecule"):
                mol_name = val.strip()
                conc = raw_row.iloc[j + 1] if j + 1 < len(raw_row) else ""
                try:
                    conc_str = f"{float(conc):g} µM"
                except (TypeError, ValueError):
                    conc_str = str(conc)
                label = f"{mol_name}\n{conc_str}"
                # Assign to next 3 columns (triplicate)
                plate_col = j   # raw df col j → plate col j (1-indexed offset)
                for k in range(3):
                    pc = j + k   # df col index → plate col number
                    if 1 <= pc <= 24:
                        labels.loc[row_letter, pc] = label
                j += 4  # skip name + conc + 1 blank
                continue
            j += 1
    return labels


def cell_counts_to_grid(csv_path) -> pd.DataFrame:
    """Load cell_counts.csv into a 16×24 float grid."""
    df = pd.read_csv(csv_path)
    grid = pd.DataFrame(np.nan, index=ALL_ROWS, columns=ALL_COLS, dtype=float)
    m = re.compile(r"^([A-P])(\d+)$", re.IGNORECASE)
    for _, row in df.iterrows():
        hit = m.match(str(row.get("well", "")).strip().upper())
        if hit:
            r, c = hit.group(1), int(hit.group(2))
            if c in ALL_COLS:
                grid.loc[r, c] = float(row.get("cell_count", np.nan))
    return grid


# ─────────────────────────────────────────────────────────────────────────────
#  Plate-effect analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyse_plate_effect(grid: pd.DataFrame, platemap_labels: pd.DataFrame | None = None):
    """
    Compute plate-effect statistics from a 16×24 reading grid.
    Returns a dict with all computed metrics.
    """
    data = grid.loc[INNER_ROWS, VALID_COLS].astype(float)

    # ── Water-control rows ────────────────────────────────────────────────
    water_stats = {}
    for row in WATER_ROWS:
        if row not in data.index:
            continue
        vals = data.loc[row].dropna().values
        if len(vals) == 0:
            continue
        water_stats[row] = {
            "mean": float(np.mean(vals)),
            "std":  float(np.std(vals)),
            "cv":   float(np.std(vals) / np.mean(vals) * 100) if np.mean(vals) else 0,
            "left_mean":  float(data.loc[row, 2:7].mean()),
            "right_mean": float(data.loc[row, 17:22].mean()),
        }

    # ── Row / column means ────────────────────────────────────────────────
    row_means = data.mean(axis=1).dropna().to_dict()
    col_means = data.mean(axis=0).dropna().to_dict()

    # ── Edge vs centre ────────────────────────────────────────────────────
    edge_rows   = ["B", "C", "N", "O"]
    center_rows = ["F", "G", "H", "I", "J", "K"]
    edge_cols   = [2, 3, 21, 22]
    center_cols = list(range(10, 15))

    edge_vals   = data.loc[[r for r in edge_rows   if r in data.index]].values.flatten()
    center_vals = data.loc[[r for r in center_rows if r in data.index], center_cols].values.flatten()
    edge_vals   = edge_vals[~np.isnan(edge_vals)]
    center_vals = center_vals[~np.isnan(center_vals)]

    edge_mean   = float(np.mean(edge_vals))   if len(edge_vals)   else np.nan
    center_mean = float(np.mean(center_vals)) if len(center_vals) else np.nan

    # ── Top-to-bottom gradient (water B vs O) ─────────────────────────────
    b_mean = water_stats.get("B", {}).get("mean", np.nan)
    o_mean = water_stats.get("O", {}).get("mean", np.nan)
    top_bottom_pct = (b_mean - o_mean) / o_mean * 100 if (not np.isnan(b_mean) and o_mean) else np.nan

    # ── Left-to-right gradient (within water rows) ────────────────────────
    left_vals  = [water_stats[r]["left_mean"]  for r in WATER_ROWS if r in water_stats]
    right_vals = [water_stats[r]["right_mean"] for r in WATER_ROWS if r in water_stats]
    left_mean  = float(np.mean(left_vals))  if left_vals  else np.nan
    right_mean = float(np.mean(right_vals)) if right_vals else np.nan
    lr_pct = (right_mean - left_mean) / left_mean * 100 if (not np.isnan(left_mean) and left_mean) else np.nan

    # ── Linear gradient test on row means (slope significance) ────────────
    rm_items   = [(ALL_ROWS.index(r), v) for r, v in row_means.items()]
    row_idx    = np.array([x[0] for x in rm_items])
    row_vals_a = np.array([x[1] for x in rm_items])
    if len(row_idx) > 2:
        slope, intercept, r_val, p_val, _ = scipy_stats.linregress(row_idx, row_vals_a)
    else:
        slope, p_val, r_val = 0.0, 1.0, 0.0

    # ── Verdict ───────────────────────────────────────────────────────────
    thresholds_exceeded = [
        abs(top_bottom_pct) > 3 if not np.isnan(top_bottom_pct) else False,
        abs(lr_pct) > 3          if not np.isnan(lr_pct)         else False,
        p_val < 0.05,
    ]
    plate_effect_detected = sum(thresholds_exceeded) >= 2

    return {
        "water_stats":            water_stats,
        "row_means":              row_means,
        "col_means":              col_means,
        "edge_mean":              edge_mean,
        "center_mean":            center_mean,
        "edge_vs_center_pct":     (edge_mean - center_mean) / center_mean * 100
                                  if (center_mean and not np.isnan(center_mean)) else np.nan,
        "top_bottom_pct":         top_bottom_pct,
        "left_right_pct":         lr_pct,
        "row_slope":              float(slope),
        "row_gradient_p":         float(p_val),
        "row_gradient_r2":        float(r_val ** 2),
        "plate_effect_detected":  plate_effect_detected,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Plotly chart helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_heatmap(grid: pd.DataFrame,
                 title: str,
                 labels_df: pd.DataFrame | None = None,
                 show_values: bool = True) -> go.Figure:
    """
    Interactive 16×24 heatmap with hover tooltips and in-cell value labels.

    Hover fix: use integer x (1–24) and integer y (0–15) so Plotly maps
    customdata[ri][ci] to exactly the cell at (y=ri, x=ci+1) with no offset.
    """
    z          = grid.values.astype(float)
    customdata = np.full_like(z, "", dtype=object)   # rich hover text
    cell_text  = np.full_like(z, "", dtype=object)   # short in-cell label

    for ri, row in enumerate(ALL_ROWS):
        for ci, col in enumerate(ALL_COLS):
            val   = z[ri, ci]
            label = ""
            if labels_df is not None:
                raw   = labels_df.loc[row, col]
                label = str(raw) if (raw and str(raw) not in ("", "nan")) else ""
            if np.isnan(val):
                customdata[ri, ci] = f"<b>Well {row}{col}</b><br>{label}<br>Count: —"
                cell_text[ri, ci]  = ""
            else:
                customdata[ri, ci] = (
                    f"<b>Well {row}{col}</b><br>{label}<br>"
                    f"Count: {val:,.0f}"
                )
                cell_text[ri, ci] = f"{val/1e6:.2f}M" if val >= 500_000 else str(int(val))

    # Use integer axes so position == value (no categorical-string ambiguity).
    # x: 1..24, y: 0..15  →  z[ri][ci] lives at (y=ri, x=ci+1)
    fig = go.Figure(go.Heatmap(
        z=z,
        x=list(range(1, 25)),   # integers 1-24
        y=list(range(16)),       # integers 0-15
        customdata=customdata,
        hovertemplate="%{customdata}<extra></extra>",
        colorscale="Plasma",
        colorbar=dict(title="Cell count"),
    ))

    # In-cell value annotations aligned to the integer axes
    if show_values:
        annotations = []
        for ri in range(16):
            for ci in range(24):
                t = cell_text[ri, ci]
                if t:
                    annotations.append(dict(
                        x=ci + 1,   # matches integer x axis (1-24)
                        y=ri,       # matches integer y axis (0-15)
                        text=t,
                        xref="x", yref="y",
                        showarrow=False,
                        font=dict(size=9, color="white"),
                        align="center",
                    ))
        fig.update_layout(annotations=annotations)

    # Water-row highlight boxes
    for row_letter in WATER_ROWS:
        ri = ALL_ROWS.index(row_letter)
        fig.add_shape(type="rect",
                      x0=0.5, x1=24.5,   # matches integer x range 1-24
                      y0=ri - 0.5, y1=ri + 0.5,
                      line=dict(color="cyan", width=1.5),
                      fillcolor="rgba(0,0,0,0)")

    fig.update_layout(
        title=title,
        xaxis=dict(
            title="Column",
            tickmode="array",
            tickvals=list(range(1, 25)),
            ticktext=[str(c) for c in ALL_COLS],
        ),
        yaxis=dict(
            title="Row",
            tickmode="array",
            tickvals=list(range(16)),
            ticktext=ALL_ROWS,
            autorange="reversed",
        ),
        height=500,
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig


def make_row_col_bar(stats_dict: dict, axis_label: str, title: str,
                     water_keys=None) -> go.Figure:
    keys   = list(stats_dict.keys())
    values = list(stats_dict.values())
    colors = [
        "#e07b54" if (water_keys and k in water_keys) else "#4c78a8"
        for k in keys
    ]
    fig = go.Figure(go.Bar(
        x=[str(k) for k in keys],
        y=values,
        marker_color=colors,
        hovertemplate=f"{axis_label} %{{x}}<br>Mean count: %{{y:,.0f}}<extra></extra>",
    ))
    grand_mean = np.nanmean(values)
    fig.add_hline(y=grand_mean, line_dash="dash", line_color="red",
                  annotation_text=f"Grand mean: {grand_mean:,.0f}")
    fig.update_layout(title=title, xaxis_title=axis_label,
                      yaxis_title="Mean cell count",
                      height=320, margin=dict(l=40, r=40, t=50, b=40))
    return fig


def make_water_comparison(water_stats: dict) -> go.Figure:
    rows  = list(water_stats.keys())
    means = [water_stats[r]["mean"] for r in rows]
    cvs   = [water_stats[r]["cv"]   for r in rows]
    lefts  = [water_stats[r]["left_mean"]  for r in rows]
    rights = [water_stats[r]["right_mean"] for r in rows]

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Water-row means (top / mid / bottom)",
                                        "Left half vs Right half per water row"))

    fig.add_trace(go.Bar(name="Row mean", x=rows, y=means,
                         error_y=dict(type="data",
                                      array=[water_stats[r]["std"] for r in rows],
                                      visible=True),
                         marker_color=["#e07b54", "#54a0e0", "#54c07b"]),
                  row=1, col=1)

    for i, r in enumerate(rows):
        fig.add_trace(go.Bar(name=f"Row {r} Left",  x=[r], y=[lefts[i]],
                             marker_color="#4c78a8", showlegend=(i == 0)),
                      row=1, col=2)
        fig.add_trace(go.Bar(name=f"Row {r} Right", x=[r], y=[rights[i]],
                             marker_color="#f58518", showlegend=(i == 0)),
                      row=1, col=2)

    fig.update_layout(barmode="group", height=330,
                      margin=dict(l=40, r=40, t=60, b=40))
    return fig


# ─────────────────────────────────────────────────────────────────────────────
#  Shared plate analysis renderer
# ─────────────────────────────────────────────────────────────────────────────

def render_plate_analysis(grid: pd.DataFrame,
                          platemap_labels: pd.DataFrame | None,
                          dataset_name: str):
    res = analyse_plate_effect(grid, platemap_labels)

    # ── Verdict banner ────────────────────────────────────────────────────
    if res["plate_effect_detected"]:
        st.markdown(
            "### 🟡 Plate Effect: **DETECTED**\n"
            "Statistical evidence (gradient across rows/columns and/or water-control "
            "variation) indicates a significant plate effect.",
            unsafe_allow_html=False,
        )
    else:
        st.markdown(
            "### 🟢 Plate Effect: **NOT DETECTED**\n"
            "No statistically significant spatial gradient found.",
            unsafe_allow_html=False,
        )

    # ── Key metrics row ───────────────────────────────────────────────────
    ws = res["water_stats"]
    b_mean = ws.get("B", {}).get("mean", np.nan)
    i_mean = ws.get("I", {}).get("mean", np.nan)
    o_mean = ws.get("O", {}).get("mean", np.nan)

    cols = st.columns(5)
    def metric(col, label, value, delta=None, delta_suffix=""):
        with col:
            if isinstance(value, float) and not np.isnan(value):
                formatted = f"{value/1e6:.3f}M" if value >= 100_000 else f"{value:,.0f}"
                col.metric(label, formatted,
                           delta=f"{delta:+.1f}%{delta_suffix}" if delta is not None else None)
            else:
                col.metric(label, "N/A")

    metric(cols[0], "Water row B (mean)", b_mean)
    metric(cols[1], "Water row I (mean)", i_mean)
    metric(cols[2], "Water row O (mean)", o_mean)
    cols[3].metric("Top→Bottom gradient",
                   f"{res['top_bottom_pct']:+.1f}%"
                   if not np.isnan(res.get('top_bottom_pct', np.nan)) else "N/A")
    cols[4].metric("Left→Right gradient (water)",
                   f"{res['left_right_pct']:+.1f}%"
                   if not np.isnan(res.get('left_right_pct', np.nan)) else "N/A")

    st.divider()

    # ── Heatmap ───────────────────────────────────────────────────────────
    st.plotly_chart(
        make_heatmap(grid, f"{dataset_name} — 384-well heatmap (cyan = water rows)",
                     platemap_labels),
        use_container_width=True,
    )

    # ── Row / column means ────────────────────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            make_row_col_bar(res["row_means"], "Row", "Mean count per row",
                             water_keys=WATER_ROWS),
            use_container_width=True,
        )
    with c2:
        st.plotly_chart(
            make_row_col_bar(res["col_means"], "Column", "Mean count per column"),
            use_container_width=True,
        )

    # ── Water-row detail ──────────────────────────────────────────────────
    if ws:
        st.plotly_chart(make_water_comparison(ws), use_container_width=True)

        with st.expander("Water-control statistics table"):
            rows_list = []
            for r, s in ws.items():
                rows_list.append({
                    "Row": r,
                    "Mean": f"{s['mean']:,.0f}",
                    "Std": f"{s['std']:,.0f}",
                    "CV (%)": f"{s['cv']:.1f}",
                    "Left mean (cols 2-7)": f"{s['left_mean']:,.0f}",
                    "Right mean (cols 17-22)": f"{s['right_mean']:,.0f}",
                })
            st.dataframe(pd.DataFrame(rows_list), use_container_width=True, hide_index=True)

    # ── Edge vs centre ────────────────────────────────────────────────────
    e_vs_c = res.get("edge_vs_center_pct", np.nan)
    if not np.isnan(e_vs_c):
        st.info(
            f"**Edge vs Centre:** edge wells average "
            f"{res['edge_mean']:,.0f}  vs centre wells {res['center_mean']:,.0f}  "
            f"({e_vs_c:+.1f}%)"
        )

    # ── Statistics summary ────────────────────────────────────────────────
    with st.expander("Statistical details"):
        st.write(f"**Row gradient** (linear regression):  "
                 f"slope = {res['row_slope']:.1f} counts/row, "
                 f"R² = {res['row_gradient_r2']:.3f}, "
                 f"p = {res['row_gradient_p']:.4f}")
        st.write("A p-value < 0.05 and positive/negative slope confirms a "
                 "statistically significant spatial gradient.")


# ─────────────────────────────────────────────────────────────────────────────
#  Main UI
# ─────────────────────────────────────────────────────────────────────────────

st.title("🔬 Pacagen Cell Analysis Suite")
st.caption("Automated fluorescence cell counting and plate-effect detection")

tab_counter, tab_plate = st.tabs(["🔬  Cell Counter", "📊  Plate Effect Analysis"])


# ════════════════════════════════════════════════════════════════════════════
#  Tab 1 — Cell Counter
# ════════════════════════════════════════════════════════════════════════════

with tab_counter:
    st.header("Cell Counter")
    st.write(
        "Upload one or more fluorescence microscopy images.  "
        "The counter will detect bright blue cells, apply well-boundary masking, "
        "and return per-image counts."
    )

    uploaded_images = st.file_uploader(
        "Upload images (JPG / PNG / TIFF)",
        type=["jpg", "jpeg", "png", "tif", "tiff"],
        accept_multiple_files=True,
    )

    if uploaded_images:
        run_btn = st.button("▶  Run Cell Counter", type="primary")

        if run_btn:
            results_rows = []
            progress = st.progress(0, text="Starting…")

            for idx, uploaded in enumerate(uploaded_images):
                progress.progress(
                    (idx) / len(uploaded_images),
                    text=f"Processing {uploaded.name}  ({idx+1}/{len(uploaded_images)})"
                )

                # Decode image
                file_bytes = np.frombuffer(uploaded.read(), np.uint8)
                img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                if img_bgr is None:
                    st.warning(f"Could not read {uploaded.name}, skipping.")
                    continue

                well = re.match(r"^([A-P]\d+)_", uploaded.name, re.IGNORECASE)
                well_name = well.group(1).upper() if well else Path(uploaded.name).stem

                centers, debug = detect_cells(img_bgr)
                count          = len(centers)

                # Resize to display size for both original and annotated
                from cell_counter import _resize, ANNOTATE_MAX_DIM
                disp_bgr, disp_scale = _resize(img_bgr, ANNOTATE_MAX_DIM)
                orig_rgb  = cv2.cvtColor(disp_bgr, cv2.COLOR_BGR2RGB)
                annot_rgb = annotate_image(disp_bgr, centers,
                                           well_contour=debug.get("well_contour"),
                                           draw_scale=disp_scale)

                results_rows.append({
                    "well":       well_name,
                    "cell_count": count,
                    "filename":   uploaded.name,
                    "orig_img":   orig_rgb,
                    "annot_img":  annot_rgb,
                    "warning":    debug.get("well_warning") or "",
                })

            progress.progress(1.0, text="Done!")
            st.session_state["counter_results"] = results_rows

    # ── Show results ──────────────────────────────────────────────────────
    if "counter_results" in st.session_state and st.session_state["counter_results"]:
        rows = st.session_state["counter_results"]
        st.success(f"Processed **{len(rows)}** image(s)")

        # Summary table
        table_df = pd.DataFrame([
            {
                "Well":       r["well"],
                "Cell Count": r["cell_count"],
                "Filename":   r["filename"],
            }
            for r in rows
        ])

        # ── Readings table ─────────────────────────────────────────────────
        st.subheader("Cell counts per well")
        col_l, col_r = st.columns([3, 1])
        with col_l:
            st.dataframe(
                table_df.style.background_gradient(
                    subset=["Cell Count"], cmap="YlOrRd"
                ),
                use_container_width=True,
                hide_index=True,
            )
        with col_r:
            st.metric("Total cells", f"{table_df['Cell Count'].sum():,}")
            st.metric("Mean / well",  f"{table_df['Cell Count'].mean():.0f}")
            st.metric("Min / well",   f"{table_df['Cell Count'].min():,}")
            st.metric("Max / well",   f"{table_df['Cell Count'].max():,}")

        # Download CSV
        csv_buf = io.StringIO()
        table_df.drop(columns=["Warning"]).to_csv(csv_buf, index=False)
        st.download_button(
            "⬇  Download results CSV",
            data=csv_buf.getvalue().encode(),
            file_name="cell_counts.csv",
            mime="text/csv",
        )

        # ── Image viewer ──────────────────────────────────────────────────
        st.divider()
        st.subheader("Image viewer")

        well_options = [f"{r['well']}  ({r['cell_count']} cells)" for r in rows]
        sel_idx = st.selectbox(
            "Select a well to inspect",
            options=range(len(rows)),
            format_func=lambda i: well_options[i],
            key="img_viewer_sel",
        )

        sel       = rows[sel_idx]
        orig_rgb  = sel["orig_img"]
        annot_rgb = sel["annot_img"]

        view_mode = st.radio(
            "View",
            ["Annotated (circles)", "Original", "Side by side"],
            horizontal=True,
            key="img_viewer_mode",
        )

        if view_mode == "Annotated (circles)":
            st.image(annot_rgb,
                     caption=f"Well {sel['well']}  —  {sel['cell_count']} cells detected",
                     use_container_width=True)
        elif view_mode == "Original":
            st.image(orig_rgb,
                     caption=f"Well {sel['well']}  —  original image",
                     use_container_width=True)
        else:
            c1, c2 = st.columns(2)
            with c1:
                st.image(orig_rgb,
                         caption="Original",
                         use_container_width=True)
            with c2:
                st.image(annot_rgb,
                         caption=f"Annotated — {sel['cell_count']} cells",
                         use_container_width=True)

        # ── Thumbnail strip ────────────────────────────────────────────────
        with st.expander(f"All {len(rows)} images (thumbnails)"):
            ncols = min(4, len(rows))
            thumb_cols = st.columns(ncols)
            for i, r in enumerate(rows):
                with thumb_cols[i % ncols]:
                    st.image(r["annot_img"],
                             caption=f"{r['well']}  {r['cell_count']}",
                             use_container_width=True)
    elif uploaded_images:
        st.info("Click **Run Cell Counter** to process the uploaded images.")

    # ── Default Assignment 1 results ──────────────────────────────────────
    st.divider()
    st.subheader("Default results — Assignment 1 images")
    st.caption(
        "Pre-computed results from the Assignment 1 image set.  "
        "These load automatically from `results/cell_counts.csv` "
        "and `results/annotated/`."
    )

    A1_CSV   = BASE_DIR / "results" / "cell_counts.csv"
    A1_ANNOT = BASE_DIR / "results" / "annotated"
    A1_IMGS  = BASE_DIR / "assignment_1" / "images"

    if not A1_CSV.exists():
        st.info(
            "No cached results yet.  "
            "Run `python cell_counter.py` to generate them, then refresh this page."
        )
    else:
        # Load CSV
        a1_df = pd.read_csv(A1_CSV)

        # ── Summary table ──────────────────────────────────────────────────
        c_l, c_r = st.columns([3, 1])
        with c_l:
            st.dataframe(
                a1_df[["well", "cell_count", "image_file"]].rename(columns={
                    "well": "Well", "cell_count": "Cell Count", "image_file": "Filename"
                }).style.background_gradient(subset=["Cell Count"], cmap="YlOrRd"),
                use_container_width=True,
                hide_index=True,
            )
        with c_r:
            st.metric("Total cells", f"{a1_df['cell_count'].sum():,}")
            st.metric("Mean / well",  f"{a1_df['cell_count'].mean():.0f}")
            st.metric("Min / well",   f"{a1_df['cell_count'].min():,}")
            st.metric("Max / well",   f"{a1_df['cell_count'].max():,}")

        csv_dl = io.StringIO()
        a1_df[["well", "cell_count", "image_file"]].to_csv(csv_dl, index=False)
        st.download_button(
            "⬇  Download Assignment 1 CSV",
            data=csv_dl.getvalue().encode(),
            file_name="assignment1_cell_counts.csv",
            mime="text/csv",
            key="dl_a1_csv",
        )

        # ── Image viewer ───────────────────────────────────────────────────
        st.markdown("#### Image viewer")

        well_list = a1_df["well"].tolist()
        count_list = a1_df["cell_count"].tolist()
        file_list  = a1_df["image_file"].tolist() if "image_file" in a1_df.columns else [""] * len(well_list)
        annot_list = a1_df["annotated"].tolist()  if "annotated"   in a1_df.columns else [""] * len(well_list)

        a1_sel_idx = st.selectbox(
            "Select a well",
            options=range(len(well_list)),
            format_func=lambda i: f"{well_list[i]}  ({count_list[i]} cells)",
            key="a1_viewer_sel",
        )

        a1_view = st.radio(
            "View",
            ["Annotated (circles)", "Original", "Side by side"],
            horizontal=True,
            key="a1_viewer_mode",
        )

        def _load_a1_image(folder, filename, max_dim=1200):
            """Load image from disk, resize, return RGB array or None."""
            if not filename:
                return None
            p = Path(folder) / filename
            if not p.exists():
                return None
            img = cv2.imread(str(p))
            if img is None:
                return None
            from cell_counter import _resize
            img_small, _ = _resize(img, max_dim)
            return cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)

        sel_well  = well_list[a1_sel_idx]
        sel_count = count_list[a1_sel_idx]
        sel_orig_file  = file_list[a1_sel_idx]
        sel_annot_file = annot_list[a1_sel_idx]

        # Cache images in session state to avoid re-reading on every widget change
        orig_cache_key  = f"a1_orig_{sel_orig_file}"
        annot_cache_key = f"a1_annot_{sel_annot_file}"

        if orig_cache_key not in st.session_state:
            st.session_state[orig_cache_key]  = _load_a1_image(A1_IMGS,  sel_orig_file)
        if annot_cache_key not in st.session_state:
            st.session_state[annot_cache_key] = _load_a1_image(A1_ANNOT, sel_annot_file)

        orig_img  = st.session_state[orig_cache_key]
        annot_img = st.session_state[annot_cache_key]

        # Fall back gracefully if a file is missing
        missing = orig_img is None and annot_img is None
        if missing:
            st.warning(f"Image files not found for well {sel_well}. "
                       "Make sure `assignment_1/images/` and `results/annotated/` exist.")
        else:
            display_orig  = orig_img  if orig_img  is not None else annot_img
            display_annot = annot_img if annot_img is not None else orig_img

            if a1_view == "Annotated (circles)":
                st.image(display_annot,
                         caption=f"Well {sel_well}  —  {sel_count} cells detected",
                         use_container_width=True)
            elif a1_view == "Original":
                st.image(display_orig,
                         caption=f"Well {sel_well}  —  original image",
                         use_container_width=True)
            else:
                c1, c2 = st.columns(2)
                with c1:
                    st.image(display_orig,
                             caption="Original",
                             use_container_width=True)
                with c2:
                    st.image(display_annot,
                             caption=f"Annotated — {sel_count} cells",
                             use_container_width=True)

        # Thumbnail strip
        with st.expander(f"All {len(well_list)} wells (thumbnails)"):
            ncols = 4
            tcols = st.columns(ncols)
            for i, (w, cnt, af) in enumerate(zip(well_list, count_list, annot_list)):
                cache_k = f"a1_thumb_{af}"
                if cache_k not in st.session_state:
                    st.session_state[cache_k] = _load_a1_image(A1_ANNOT, af, max_dim=400)
                img_t = st.session_state[cache_k]
                if img_t is not None:
                    with tcols[i % ncols]:
                        st.image(img_t, caption=f"{w}  {cnt}", use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
#  Tab 2 — Plate Effect Analysis
# ════════════════════════════════════════════════════════════════════════════

with tab_plate:
    st.header("Plate Effect Analysis")
    st.write(
        "Spatial gradients across a 384-well plate indicate a *plate effect* — "
        "systematic bias that skews every measurement.  "
        "Water-control rows (B, I, O) provide the cleanest signal because "
        "any variation in those rows is purely technical, not biological."
    )

    sub_a2, sub_a1 = st.tabs(["Assignment 2 — readings.xlsx", "Assignment 1 — Cell Counter Results"])

    # ── Assignment 2 ──────────────────────────────────────────────────────
    with sub_a2:
        st.subheader("Readings dataset (readings.xlsx)")

        # File loader — default or upload
        use_default_a2 = A2_READINGS_PATH.exists()
        if use_default_a2:
            st.caption(f"Using default file: `{A2_READINGS_PATH.name}`")
            readings_source = A2_READINGS_PATH
        else:
            readings_upload = st.file_uploader(
                "Upload readings.xlsx", type=["xlsx"], key="a2_upload"
            )
            readings_source = readings_upload

        if readings_source:
            with st.spinner("Parsing readings…"):
                grid_a2 = parse_plate_grid(readings_source)

            # Load plate-map labels for tooltip annotation
            pm_labels_a2 = None
            if use_default_a2:
                # Plate map is embedded in the same file (rows 18-34 of Sheet1)
                # Parse from the bottom half of the same xlsx
                try:
                    df_raw = pd.read_excel(A2_READINGS_PATH, sheet_name=0, header=None)
                    # Second grid starts at row index 18 (0-based)
                    pm_labels_a2 = pd.DataFrame("", index=ALL_ROWS, columns=ALL_COLS, dtype=str)
                    offset = 18   # row offset where plate-map starts
                    for i, row_letter in enumerate(ALL_ROWS):
                        raw_row = df_raw.iloc[offset + i + 1]
                        j = 1
                        while j < len(raw_row):
                            val = raw_row.iloc[j]
                            if isinstance(val, str) and val.strip().lower() == "water":
                                for c in range(2, 23):
                                    pm_labels_a2.loc[row_letter, c] = "Water"
                                break
                            elif isinstance(val, str) and val.strip().startswith("Molecule"):
                                mol = val.strip()
                                conc = raw_row.iloc[j + 1] if j + 1 < len(raw_row) else ""
                                try:
                                    cs = f"{float(conc):g} µM"
                                except (TypeError, ValueError):
                                    cs = str(conc)
                                for k in range(3):
                                    pc = j + k
                                    if 1 <= pc <= 24:
                                        pm_labels_a2.loc[row_letter, pc] = f"{mol}\n{cs}"
                                j += 4
                                continue
                            j += 1
                except Exception:
                    pm_labels_a2 = None

            render_plate_analysis(grid_a2, pm_labels_a2, "Assignment 2")
        else:
            st.info("Upload `readings.xlsx` to begin analysis.")

    # ── Assignment 1 ──────────────────────────────────────────────────────
    with sub_a1:
        st.subheader("Assignment 1 — Cell Counter Results")

        # Priority order for data source:
        #   1. Cached file at results/cell_counts.csv  (auto-loaded, no action needed)
        #   2. Manual upload (if cache doesn't exist yet)
        csv_source    = None
        source_label  = None

        if A1_RESULTS_PATH.exists():
            csv_source   = A1_RESULTS_PATH
            source_label = f"Auto-loaded from cache: `{A1_RESULTS_PATH}`"
        else:
            st.info(
                "No cached results found at `results/cell_counts.csv`.  "
                "Either run the cell counter (`python cell_counter.py`) "
                "or upload an existing CSV below."
            )
            csv_upload = st.file_uploader(
                "Upload cell_counts.csv", type=["csv"], key="a1_csv"
            )
            if csv_upload:
                csv_source   = csv_upload
                source_label = f"Uploaded: `{csv_upload.name}`"

        if csv_source is not None:
            if source_label:
                st.caption(source_label)

            # Plate map labels (auto-loaded from assignment_1/Plate_map.xlsx)
            pm_labels_a1 = None
            if A1_PLATEMAP_PATH.exists():
                try:
                    pm_labels_a1 = parse_platemap_labels(A1_PLATEMAP_PATH)
                except Exception:
                    pm_labels_a1 = None

            with st.spinner("Parsing results…"):
                grid_a1 = cell_counts_to_grid(csv_source)

            n_wells = int((~grid_a1.isna()).values.sum())
            st.caption(f"{n_wells} wells loaded")
            render_plate_analysis(grid_a1, pm_labels_a1, "Assignment 1")
