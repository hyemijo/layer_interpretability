"""
Layerwise Cross-Lingual Difference Visualization
=================================================
Compares hidden states from two ICL settings:
  (1) English demonstrations + target-language query
  (2) Target-language demonstrations + target-language query

Cross-lingual difference per layer = cosine distance between the two settings'
hidden states, averaged over num_samples validation queries.

Expects three parallel CSV files (same row index = same content in different languages):
  --train_tgt   trainset in target language
  --train_en    trainset in English
  --val_tgt     validation set in target language  (query source)

Demonstrations are randomly sampled from the trainset; queries come from the
validation set. Prompts are formatted with the model's chat template.

Requirements:
    pip install transformers torch pandas matplotlib seaborn tqdm

Usage:
    python visualize_crosslingual_diff.py \
        --model_name  "meta-llama/Meta-Llama-3-8B-Instruct" \
        --train_tgt   "data/train_zh.csv" \
        --train_en    "data/train_en.csv" \
        --val_tgt     "data/val_zh.csv" \
        --target_lang "zh" \
        --num_shots   3 \
        --num_samples 50 \
        --output_dir  "./output"

Re-plotting from saved .npy files (skip model forward pass):
    python visualize_crosslingual_diff.py \
        --skip_extraction \
        --target_lang "zh" \
        --output_dir  "./output"
"""

import argparse
import os
import random
import warnings
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


# ──────────────────────────────────────────────
# 1.  Data helpers
# ──────────────────────────────────────────────

REQUIRED_COLS = {"instruction", "input", "id", "output", "lang_q", "lang_a"}


def _load_csv(path: str, label: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"[{label}] CSV '{path}' is missing columns: {missing}")
    df["input"] = df["input"].fillna("")
    return df.reset_index(drop=True)


def load_splits(
    train_tgt_path: str,
    train_en_path: str,
    val_tgt_path: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_tgt = _load_csv(train_tgt_path, "train_tgt")
    train_en  = _load_csv(train_en_path,  "train_en")
    val_tgt   = _load_csv(val_tgt_path,   "val_tgt")

    if len(train_tgt) != len(train_en):
        raise ValueError(
            f"train_tgt ({len(train_tgt)} rows) and train_en ({len(train_en)} rows) "
            "must be the same length (parallel files)."
        )

    print(f"[i] train_tgt rows : {len(train_tgt)}")
    print(f"[i] train_en  rows : {len(train_en)}")
    print(f"[i] val_tgt   rows : {len(val_tgt)}")
    return train_tgt, train_en, val_tgt


def build_chat_messages(demonstrations: List[Dict], query: Dict) -> List[Dict]:
    messages: List[Dict] = []
    for demo in demonstrations:
        user_content = demo["instruction"].strip()
        if demo.get("input", "").strip():
            user_content += "\n\n" + demo["input"].strip()
        messages.append({"role": "user",      "content": user_content})
        messages.append({"role": "assistant", "content": demo["output"].strip()})

    q_content = query["instruction"].strip()
    if query.get("input", "").strip():
        q_content += "\n\n" + query["input"].strip()
    messages.append({"role": "user", "content": q_content})
    return messages


def apply_template(
    tokenizer,
    messages: List[Dict],
    add_generation_prompt: bool = True,
) -> str:
    if tokenizer.chat_template is not None:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    warnings.warn(
        "Tokenizer has no chat_template — falling back to Alpaca-style formatting.",
        UserWarning,
        stacklevel=2,
    )
    lines = []
    for msg in messages:
        role, text = msg["role"], msg["content"]
        if role == "user":
            lines.append(f"### Instruction:\n{text}\n")
        elif role == "assistant":
            lines.append(f"### Response:\n{text}\n")
    lines.append("### Response:")
    return "\n".join(lines)


# ──────────────────────────────────────────────
# 2.  Hidden-state extraction
# ──────────────────────────────────────────────

@torch.no_grad()
def get_layerwise_hidden_states_batch(
    model,
    tokenizer,
    prompts: List[str],
    device: str,
    max_length: int = 2048,
) -> np.ndarray:
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(device)

    tokenizer.padding_side = original_padding_side

    outputs = model(**inputs, output_hidden_states=True)

    batch_layer_reps = []
    for layer_hs in outputs.hidden_states:
        last_tok = layer_hs[:, -1, :].cpu().float().numpy()
        batch_layer_reps.append(last_tok)

    stacked = np.stack(batch_layer_reps, axis=0)
    return stacked.transpose(1, 0, 2)


def cosine_distance_batch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-10)
    b_norm = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-10)
    return 1.0 - np.einsum("lh,lh->l", a_norm, b_norm)


# ──────────────────────────────────────────────
# 3.  Main extraction loop
# ──────────────────────────────────────────────

def _select_query_indices(n_val: int, num_samples: int, seed: int) -> List[int]:
    rng = random.Random(seed)
    return rng.sample(range(n_val), min(num_samples, n_val))


def _select_demo_indices(n_train: int, num_shots: int, seed: int, query_pos: int) -> List[int]:
    per_query_seed = seed * 1_000_003 + query_pos
    rng = random.Random(per_query_seed)
    return rng.sample(range(n_train), min(num_shots, n_train))


def compute_crosslingual_diffs(
    model,
    tokenizer,
    train_tgt: pd.DataFrame,
    train_en: pd.DataFrame,
    val_tgt: pd.DataFrame,
    target_lang: str,
    num_shots: int,
    num_samples: int,
    device: str,
    seed: int = 42,
    max_length: int = 2048,
    batch_size: int = 8,
) -> tuple:
    query_positions = _select_query_indices(len(val_tgt), num_samples, seed)
    print(f"[i] Query positions (first 5): {query_positions[:5]} …")

    prompts_bare: List[str] = []
    prompts_en:   List[str] = []
    prompts_tgt:  List[str] = []

    for i, query_pos in enumerate(query_positions):
        query_row    = val_tgt.iloc[query_pos].to_dict()
        demo_indices = _select_demo_indices(len(train_tgt), num_shots, seed, query_pos)

        tgt_demos = [train_tgt.iloc[j].to_dict() for j in demo_indices]
        en_demos  = [train_en.iloc[j].to_dict()  for j in demo_indices]
        if i == 0:
            print(tgt_demos[0], tgt_demos[-1])
            print(en_demos[0], en_demos[-1])
            print("------------------")

        prompts_bare.append(apply_template(tokenizer, build_chat_messages([], query_row)))
        prompts_en.append(  apply_template(tokenizer, build_chat_messages(en_demos,  query_row)))
        prompts_tgt.append( apply_template(tokenizer, build_chat_messages(tgt_demos, query_row)))

    all_hs_bare: List[np.ndarray] = []
    all_hs_en:   List[np.ndarray] = []
    all_hs_tgt:  List[np.ndarray] = []

    n = len(prompts_en)
    for start in tqdm(range(0, n, batch_size), desc="Batched forward passes"):
        sl = slice(start, start + batch_size)

        hs_bare_b = get_layerwise_hidden_states_batch(
            model, tokenizer, prompts_bare[sl], device, max_length)
        hs_en_b   = get_layerwise_hidden_states_batch(
            model, tokenizer, prompts_en[sl],   device, max_length)
        hs_tgt_b  = get_layerwise_hidden_states_batch(
            model, tokenizer, prompts_tgt[sl],  device, max_length)

        all_hs_bare.extend(hs_bare_b)
        all_hs_en.extend(hs_en_b)
        all_hs_tgt.extend(hs_tgt_b)

    cross_diffs:   List[np.ndarray] = []
    icl_en_diffs:  List[np.ndarray] = []
    icl_tgt_diffs: List[np.ndarray] = []

    for hs_bare, hs_en, hs_tgt in zip(all_hs_bare, all_hs_en, all_hs_tgt):
        assert (len(hs_bare) == len(hs_en)) and (len(hs_en) == len(hs_tgt))
        cross_diffs.append(  cosine_distance_batch(hs_en, hs_tgt))
        icl_en_diffs.append( cosine_distance_batch(hs_bare, hs_en))
        icl_tgt_diffs.append(cosine_distance_batch(hs_bare, hs_tgt))

    return (
        np.stack(cross_diffs,   axis=0),
        np.stack(icl_en_diffs,  axis=0),
        np.stack(icl_tgt_diffs, axis=0),
    )


# ──────────────────────────────────────────────
# 4.  Save / load helpers
# ──────────────────────────────────────────────

def _save_matrix(mat: np.ndarray, path: str, label: str):
    np.save(path, mat)
    print(f"[✓] {label} matrix → {path}")

    stats_path = path.replace(".npy", "_stats.csv")
    num_layers  = mat.shape[1]
    pd.DataFrame({
        "layer":  np.arange(num_layers),
        "mean":   mat.mean(axis=0),
        "std":    mat.std(axis=0),
        "min":    mat.min(axis=0),
        "max":    mat.max(axis=0),
        "median": np.median(mat, axis=0),
    }).to_csv(stats_path, index=False)
    print(f"[✓] {label} stats   → {stats_path}")


def save_results(
    diff_matrix: np.ndarray,
    output_dir: str,
    target_lang: str,
    icl_en_matrix:  Optional[np.ndarray] = None,
    icl_tgt_matrix: Optional[np.ndarray] = None,
):
    _save_matrix(
        diff_matrix,
        os.path.join(output_dir, f"diff_matrix_{target_lang}.npy"),
        "Cross-lingual diff",
    )
    if icl_en_matrix is not None:
        _save_matrix(
            icl_en_matrix,
            os.path.join(output_dir, f"icl_en_matrix_{target_lang}.npy"),
            "ICL-EN-vs-bare diff",
        )
    if icl_tgt_matrix is not None:
        _save_matrix(
            icl_tgt_matrix,
            os.path.join(output_dir, f"icl_tgt_matrix_{target_lang}.npy"),
            "ICL-TGT-vs-bare diff",
        )


def load_results(
    output_dir: str,
    target_lang: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    paths = {
        "diff":    os.path.join(output_dir, f"diff_matrix_{target_lang}.npy"),
        "icl_en":  os.path.join(output_dir, f"icl_en_matrix_{target_lang}.npy"),
        "icl_tgt": os.path.join(output_dir, f"icl_tgt_matrix_{target_lang}.npy"),
    }
    missing = [p for p in paths.values() if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(
            "Cannot load saved matrices — run extraction first:\n  " +
            "\n  ".join(missing)
        )

    diff_matrix    = np.load(paths["diff"])
    icl_en_matrix  = np.load(paths["icl_en"])
    icl_tgt_matrix = np.load(paths["icl_tgt"])

    print(f"[i] Loaded diff_matrix    : {diff_matrix.shape}  ← {paths['diff']}")
    print(f"[i] Loaded icl_en_matrix  : {icl_en_matrix.shape}  ← {paths['icl_en']}")
    print(f"[i] Loaded icl_tgt_matrix : {icl_tgt_matrix.shape}  ← {paths['icl_tgt']}")
    return diff_matrix, icl_en_matrix, icl_tgt_matrix


# ──────────────────────────────────────────────
# 5.  Plotting helpers
# ──────────────────────────────────────────────

_LANG_COLORS = [
    "#2166ac",  # blue
    "#d6604d",  # red-orange
    "#4dac26",  # green
    "#762a83",  # purple
    "#e08214",  # amber
    "#1a9641",  # dark green
    "#92c5de",  # light blue
]

_STYLE_PARAMS = {
    "font.family":          "serif",
    "font.serif":           ["DejaVu Serif", "Times New Roman", "Palatino", "serif"],
    "mathtext.fontset":     "dejavuserif",
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    "axes.linewidth":       0.8,
    "xtick.direction":      "out",
    "ytick.direction":      "out",
    "xtick.major.size":     3.5,
    "ytick.major.size":     3.5,
    "xtick.minor.visible":  False,
    "ytick.minor.visible":  False,
    "legend.framealpha":    0.93,
    "legend.edgecolor":     "0.75",
    "legend.fontsize":      8.5,
    "legend.handlelength":  1.6,
    "axes.labelsize":       10,
    "axes.titlesize":       10,
    "xtick.labelsize":      8.5,
    "ytick.labelsize":      8.5,
    "lines.linewidth":      1.6,
}

# Fraction of the current y-span added above the data to make room for labels.
_YLIM_HEADROOM = 0.22


# ---------------------------------------------------------------------------
# AnnotationManager – collision-aware peak label placement
# ---------------------------------------------------------------------------

class AnnotationManager:
    """
    Places peak annotation boxes so they do not overlap each other, the
    legend, or the data curves.

    All geometry is tracked in *axes-fraction* coordinates [0, 1]².
    Call :meth:`place` for each peak in drawing order; each successful
    placement is registered so later calls avoid it.

    Parameters
    ----------
    ax          : target Axes (y-limits must be final before the first call)
    legend_loc  : same string passed to ``ax.legend(loc=…)``
    n_legend_rows : approximate number of legend rows (scales reserved height)
    """

    # Default box size in axes-fraction units.
    # These are estimates for a label like "L12: 0.1234" at 7.5 pt font.
    # _DEFAULT_BW: float = 0.22
    # _DEFAULT_BH: float = 0.10
    _DEFAULT_BW: float = 0.16
    _DEFAULT_BH: float = 0.075

    # Minimum clear gap around every reserved region.
    _GAP: float = 0.014

    def __init__(
        self,
        ax: "plt.Axes",
        legend_loc: str = "upper left",
        n_legend_rows: int = 1,
    ) -> None:
        self.ax = ax
        self._reserved: list[tuple[float, float, float, float]] = []
        self._reserve_legend(legend_loc, n_legend_rows)

    # ── coordinate helpers ───────────────────────────────────────────────

    def _d2f(self, xd: float, yd: float) -> tuple[float, float]:
        """Data → axes-fraction."""
        xl, xh = self.ax.get_xlim()
        yl, yh = self.ax.get_ylim()
        return (xd - xl) / (xh - xl), (yd - yl) / (yh - yl)

    def _f2d(self, xf: float, yf: float) -> tuple[float, float]:
        """Axes-fraction → data."""
        xl, xh = self.ax.get_xlim()
        yl, yh = self.ax.get_ylim()
        return xl + xf * (xh - xl), yl + yf * (yh - yl)

    # ── reservation helpers ──────────────────────────────────────────────

    def _reserve(self, x0: float, y0: float, x1: float, y1: float) -> None:
        self._reserved.append((
            max(0.0, float(x0)), max(0.0, float(y0)),
            min(1.0, float(x1)), min(1.0, float(y1)),
        ))

    def _reserve_legend(self, loc: str, n_rows: int) -> None:
        # Scale height with number of legend rows (≈5.5% per row, capped at 55%).
        h = min(0.06 + 0.055 * n_rows, 0.55)
        w = 0.46
        locs: dict[str, tuple[float, float, float, float]] = {
            "upper left":   (0.0,       1.0 - h, w,         1.0),
            "upper right":  (1.0 - w,   1.0 - h, 1.0,       1.0),
            "upper center": (0.5 - w/2, 1.0 - h, 0.5 + w/2, 1.0),
            "lower left":   (0.0,       0.0,      w,         h  ),
            "lower right":  (1.0 - w,   0.0,      1.0,       h  ),
        }
        self._reserve(*locs.get(loc, locs["upper left"]))

    # ── overlap tests ────────────────────────────────────────────────────

    def _overlaps_reserved(self, cx: float, cy: float, bw: float, bh: float) -> bool:
        g = self._GAP
        x0, y0 = cx - bw / 2 - g, cy - bh / 2 - g
        x1, y1 = cx + bw / 2 + g, cy + bh / 2 + g
        for rx0, ry0, rx1, ry1 in self._reserved:
            if x0 < rx1 and x1 > rx0 and y0 < ry1 and y1 > ry0:
                return True
        return False

    def _overlaps_curve(
        self,
        cx: float, cy: float, bw: float, bh: float,
        layers: np.ndarray, mean: np.ndarray,
    ) -> bool:
        """True if the candidate box footprint intersects any segment of *mean*."""
        xd0, yd0 = self._f2d(cx - bw / 2, cy - bh / 2)
        xd1, yd1 = self._f2d(cx + bw / 2, cy + bh / 2)
        mask = (layers >= xd0) & (layers <= xd1)
        if not np.any(mask):
            return False
        vals = mean[mask]
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            return False
        vmin, vmax = float(vals.min()), float(vals.max())
        # Extra vertical margin = 30 % of box height in data units.
        margin_d = (yd1 - yd0) * 0.30
        return not (yd1 + margin_d < vmin or yd0 - margin_d > vmax)

    # ── main placement method ────────────────────────────────────────────
    def place(
        self,
        peak_x: float,
        peak_y: float,
        layers: np.ndarray,
        mean: np.ndarray,
        bw: Optional[float] = None,
        bh: Optional[float] = None,
    ) -> tuple[float, float]:
    
        bw = bw if bw is not None else self._DEFAULT_BW
        bh = bh if bh is not None else self._DEFAULT_BH
    
        pxf, pyf = self._d2f(peak_x, peak_y)
    
        # Candidate offsets around the peak (axes-fraction units)
        offsets = [
            ( 0.10,  0.08),
            (-0.10,  0.08),
            ( 0.10, -0.08),
            (-0.10, -0.08),
            ( 0.00,  0.12),
            ( 0.00, -0.12),
            ( 0.14,  0.00),
            (-0.14,  0.00),
        ]
    
        edge = 0.03
    
        best = None
        best_score = np.inf
    
        for dx, dy in offsets:
            cxf = pxf + dx
            cyf = pyf + dy
    
            # keep inside axes
            cxf = np.clip(cxf, edge + bw / 2, 1.0 - edge - bw / 2)
            cyf = np.clip(cyf, edge + bh / 2, 1.0 - edge - bh / 2)
    
            if self._overlaps_reserved(cxf, cyf, bw, bh):
                continue
    
            if self._overlaps_curve(cxf, cyf, bw, bh, layers, mean):
                continue
    
            # Prefer SHORT arrows
            dist = np.hypot(dx, dy)
    
            if dist < best_score:
                best_score = dist
                best = (cxf, cyf)
    
        if best is None:
            # fallback: place slightly above peak
            cxf = np.clip(
                pxf,
                edge + bw / 2,
                1.0 - edge - bw / 2,
            )
            cyf = np.clip(
                pyf + 0.12,
                edge + bh / 2,
                1.0 - edge - bh / 2,
            )
            best = (cxf, cyf)
    
        cxf, cyf = best
    
        self._reserve(
            cxf - bw / 2,
            cyf - bh / 2,
            cxf + bw / 2,
            cyf + bh / 2,
        )
    
        return self._f2d(cxf, cyf)


    # def place(
    #     self,
    #     peak_x: float,
    #     peak_y: float,
    #     layers: np.ndarray,
    #     mean: np.ndarray,
    #     bw: Optional[float] = None,
    #     bh: Optional[float] = None,
    # ) -> tuple[float, float]:
    #     """
    #     Find the best centre position (in *data* coordinates) for an annotation
    #     box of axes-fraction size (*bw*, *bh*) and register it.

    #     The scoring function rewards:
    #     - distance from the peak marker (labels away from peaks read more easily)
    #     - distance from the axes centre (edge-anchored labels look cleaner)
    #     """
    #     bw = bw if bw is not None else self._DEFAULT_BW
    #     bh = bh if bh is not None else self._DEFAULT_BH
    #     pxf, pyf = self._d2f(peak_x, peak_y)

    #     edge = 0.03  # minimum margin from axes border
    #     n    = 11    # grid resolution per axis
    #     xs   = np.linspace(edge + bw / 2, 1.0 - edge - bw / 2, n)
    #     ys   = np.linspace(edge + bh / 2, 1.0 - edge - bh / 2, n)

    #     best: Optional[tuple[float, float]] = None
    #     best_score = -np.inf

    #     for cxf in xs:
    #         for cyf in ys:
    #             if self._overlaps_reserved(cxf, cyf, bw, bh):
    #                 continue
    #             if self._overlaps_curve(cxf, cyf, bw, bh, layers, mean):
    #                 continue
    #             dist_peak   = np.hypot(cxf - pxf, cyf - pyf)
    #             dist_centre = np.hypot(cxf - 0.5, cyf - 0.5)
    #             score = dist_peak + 0.25 * dist_centre
    #             if score > best_score:
    #                 best_score = score
    #                 best = (cxf, cyf)

    #     if best is None:
    #         # Fallback: corner farthest from the peak (placement guaranteed).
    #         corners = [
    #             (edge + bw / 2,       1.0 - edge - bh / 2),   # top-left
    #             (1.0 - edge - bw / 2, 1.0 - edge - bh / 2),   # top-right
    #             (edge + bw / 2,       edge + bh / 2),           # bottom-left
    #             (1.0 - edge - bw / 2, edge + bh / 2),           # bottom-right
    #         ]
    #         best = max(corners, key=lambda c: np.hypot(c[0] - pxf, c[1] - pyf))

    #     cxf, cyf = best
    #     self._reserve(cxf - bw / 2, cyf - bh / 2, cxf + bw / 2, cyf + bh / 2)
    #     return self._f2d(cxf, cyf)


# ---------------------------------------------------------------------------
# Drawing primitives
# ---------------------------------------------------------------------------

def _expand_ylim(ax: "plt.Axes", headroom: float = _YLIM_HEADROOM) -> None:
    """Add vertical headroom above the data so annotation labels have room."""
    y_lo, y_hi = ax.get_ylim()
    ax.set_ylim(y_lo, y_hi + (y_hi - y_lo) * headroom)


def _draw_mean_std(
    ax: "plt.Axes",
    layers: np.ndarray,
    diff_matrix: np.ndarray,
    color: str,
    label: str,
    alpha_band: float = 0.14,
    linewidth: float = 1.7,
    marker: str = "o",
    markersize: float = 3.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Plot mean ± 1 σ band; return (mean, std)."""
    mean = diff_matrix.mean(axis=0)
    std  = diff_matrix.std(axis=0)
    ax.fill_between(layers, mean - std, mean + std,
                    color=color, alpha=alpha_band, linewidth=0)
    ax.plot(layers, mean,
            color=color, linewidth=linewidth,
            marker=marker, markersize=markersize,
            markerfacecolor="white", markeredgecolor=color, markeredgewidth=1.2,
            label=label, zorder=4)
    return mean, std


def _annotate_peak(
    ax: "plt.Axes",
    layers: np.ndarray,
    mean: np.ndarray,
    color: str,
    ann_mgr: "AnnotationManager",
) -> None:
    """
    Mark the peak of *mean* with a star, draw partial crosshairs anchored
    precisely at the peak data-point, and place a non-overlapping label box
    managed by *ann_mgr*.

    Notes
    -----
    * ``np.nanargmax`` is used so NaN values (e.g. in masked ratio curves)
      are silently skipped.
    * Crosshairs are drawn as explicit line segments from the axis edges
      to the exact peak coordinates — not as full-span axvline/axhline —
      to guarantee pixel-perfect alignment with the star marker.
    * The annotation box position is decided by ``AnnotationManager.place``,
      which avoids overlap with the legend, existing labels, and the data curve.
    """
    finite_mean = np.where(np.isfinite(mean), mean, np.nan)
    if np.all(np.isnan(finite_mean)):
        return  # nothing to annotate

    peak_idx = int(np.nanargmax(finite_mean))
    peak_x   = float(layers[peak_idx])
    peak_y   = float(finite_mean[peak_idx])

    xl, xh = ax.get_xlim()
    yl, _  = ax.get_ylim()

    # ── star marker at the exact peak ────────────────────────────────────
    ax.plot(peak_x, peak_y,
            marker="*", markersize=11,
            color=color,
            markeredgecolor="white", markeredgewidth=0.7,
            zorder=6)

    # ── partial crosshairs: left-edge→peak (horizontal) and
    #    bottom-edge→peak (vertical).  Using explicit segments instead of
    #    axvline/axhline ensures the lines terminate exactly at the peak. ──
    ax.plot([xl, peak_x], [peak_y, peak_y],
            color=color, linewidth=0.55, linestyle="--", alpha=0.40, zorder=2)
    ax.plot([peak_x, peak_x], [yl, peak_y],
            color=color, linewidth=0.55, linestyle="--", alpha=0.40, zorder=2)

    # ── non-overlapping label ─────────────────────────────────────────────
    text_x, text_y = ann_mgr.place(peak_x, peak_y, layers, finite_mean)

    ax.annotate(
        f"L{int(peak_x)}: {peak_y:.4f}",
        xy=(peak_x, peak_y),
        xytext=(text_x, text_y),
        fontsize=7.5,
        color=color,
        fontweight="bold",
        ha="center", va="center",
        bbox=dict(
            boxstyle="round,pad=0.30",
            facecolor="white",
            edgecolor=color,
            linewidth=0.8,
            alpha=0.95,
        ),
        arrowprops=dict(
            arrowstyle="-",
            color=color,
            lw=0.7,
            alpha=0.8,
        ),
        
        # arrowprops=dict(
        #     arrowstyle="-|>",
        #     color=color,
        #     lw=0.8,
        #     mutation_scale=8,
        #     connectionstyle="arc3,rad=0.08", # 0.18
        # ),
        # zorder=9,
    )


def _compute_ratio(
    mean_tgt: np.ndarray,
    mean_en: np.ndarray,
    eps: float = 1e-6,
    mask_threshold: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    valid_mask = mean_en >= mask_threshold
    ratio = np.where(valid_mask, mean_tgt / (mean_en + eps), np.nan)
    return ratio, valid_mask


def _draw_ratio(
    ax: "plt.Axes",
    layers: np.ndarray,
    ratio: np.ndarray,
    valid_mask: np.ndarray,
    color: str,
    label: str,
    linewidth: float = 1.6,
    markersize: float = 3.0,
) -> None:
    ax.axhline(1.0, color="#aaaaaa", linewidth=0.8, linestyle="--", zorder=1)

    if not valid_mask.all():
        for lyr in layers[~valid_mask]:
            ax.axvspan(lyr - 0.5, lyr + 0.5,
                       color="#eeeeee", alpha=0.6, linewidth=0, zorder=0)

    ax.plot(layers, ratio,
            color=color, linewidth=linewidth,
            marker="D", markersize=markersize,
            markerfacecolor="white", markeredgecolor=color, markeredgewidth=1.1,
            label=label, zorder=4)


def _setup_ax(ax: "plt.Axes", num_layers: int, ylabel: str, max_ticks: int = 16) -> None:
    """Apply consistent axis styling."""
    ax.set_facecolor("white")
    ax.yaxis.grid(True, linewidth=0.45, color="#dddddd", zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlim(0.5, num_layers + 0.5)
    ax.set_xlabel("Layer index")
    ax.set_ylabel(ylabel)

    try:
        fig        = ax.get_figure()
        width_inch = ax.get_position().width * fig.get_figwidth()
        pixels     = width_inch * fig.dpi
    except Exception:
        pixels = 400.0

    ticks_by_pixels = max(1, int(pixels / 45))
    n_ticks         = min(max_ticks, ticks_by_pixels)
    raw_interval    = num_layers / n_ticks

    for nice in [1, 2, 4, 5, 8, 10, 16, 20, 25, 32, 50]:
        if nice >= raw_interval:
            interval = nice
            break
    else:
        interval = int(raw_interval) + 1

    ax.xaxis.set_major_locator(ticker.MultipleLocator(interval))

    expected_labels = num_layers // interval
    rot = 45 if expected_labels > 12 else 0
    ax.tick_params(axis="x", labelrotation=rot, labelsize=8.5)


def _save_panel(fig: "plt.Figure", ax: "plt.Axes", path: str) -> None:
    all_axes   = fig.get_axes()
    visibility = [a.get_visible() for a in all_axes]

    for a in all_axes:
        if a is not ax:
            a.set_visible(False)

    original_pos = ax.get_position()
    ax.set_position([0.12, 0.12, 0.82, 0.76])

    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")

    ax.set_position(original_pos)
    for a, vis in zip(all_axes, visibility):
        a.set_visible(vis)

    print(f"[✓] Panel saved → {path}")


# ──────────────────────────────────────────────
# 6.  Plot functions
# ──────────────────────────────────────────────

def plot_crosslingual_diff(
    diff_matrix: np.ndarray,
    target_lang: str,
    num_shots: int,
    output_path: str,
    model_name: str = "",
) -> None:
    num_layers = diff_matrix.shape[1]
    layers     = np.arange(num_layers)
    color      = _LANG_COLORS[0]

    with plt.rc_context(_STYLE_PARAMS):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        fig.patch.set_facecolor("white")
        _setup_ax(ax, num_layers, "Cosine distance  (EN-shot vs. TGT-shot)")

        mean, _ = _draw_mean_std(
            ax, layers, diff_matrix, color,
            label=f"{target_lang.upper()} demos vs. EN demos  (n={diff_matrix.shape[0]})",
        )

        ax.legend(loc="upper left")
        _expand_ylim(ax)

        ann_mgr = AnnotationManager(ax, legend_loc="upper left", n_legend_rows=1)
        _annotate_peak(ax, layers, mean, color, ann_mgr)

        model_tag = model_name.split("/")[-1] if model_name else ""
        ax.set_title(
            f"Layerwise cross-lingual hidden-state difference\n"
            f"{model_tag}  |  {num_shots}-shot  |  lang: {target_lang.upper()}",
            fontsize=10, pad=6,
        )

        fig.tight_layout()
        fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
        print(f"[✓] Single-language plot saved → {output_path}")
        plt.close(fig)


def plot_multilang_merged(
    lang_matrix_pairs: List[tuple],
    num_shots: int,
    output_path: str,
    model_name: str = "",
    figsize: tuple = (9, 5),
) -> None:
    if not lang_matrix_pairs:
        raise ValueError("lang_matrix_pairs is empty.")

    num_layers = lang_matrix_pairs[0][1].shape[1]
    layers     = np.arange(num_layers)

    with plt.rc_context(_STYLE_PARAMS):
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor("white")
        _setup_ax(ax, num_layers, "Cosine distance  (EN-shot vs. TGT-shot)")

        means = []
        for idx, (lang, diff_matrix) in enumerate(lang_matrix_pairs):
            if diff_matrix.shape[1] != num_layers:
                raise ValueError(
                    f"Language '{lang}' has {diff_matrix.shape[1]} layers; "
                    f"expected {num_layers}."
                )
            color = _LANG_COLORS[idx % len(_LANG_COLORS)]
            mean, _ = _draw_mean_std(
                ax, layers, diff_matrix, color,
                label=f"{lang.upper()}  (n={diff_matrix.shape[0]})",
                linewidth=1.7, markersize=3.0,
            )
            means.append(mean)

        n_langs = len(lang_matrix_pairs)
        ax.legend(loc="upper left", ncol=min(4, n_langs))
        _expand_ylim(ax)

        ann_mgr = AnnotationManager(ax, legend_loc="upper left", n_legend_rows=n_langs)
        for idx, ((lang, _), mean) in enumerate(zip(lang_matrix_pairs, means)):
            color = _LANG_COLORS[idx % len(_LANG_COLORS)]
            _annotate_peak(ax, layers, mean, color, ann_mgr)

        model_tag  = model_name.split("/")[-1] if model_name else ""
        lang_codes = ", ".join(l.upper() for l, _ in lang_matrix_pairs)
        ax.set_title(
            f"Layerwise cross-lingual hidden-state difference\n"
            f"{model_tag}  |  {num_shots}-shot  |  languages: {lang_codes}",
            fontsize=10, pad=6,
        )

        fig.tight_layout()
        fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
        print(f"[✓] Multi-language merged plot saved → {output_path}")
        plt.close(fig)


def plot_multilang_from_files(
    npy_paths: Dict[str, str],
    num_shots: int,
    output_path: str,
    model_name: str = "",
    figsize: tuple = (9, 5),
) -> None:
    pairs = []
    for lang, path in npy_paths.items():
        mat = np.load(path)
        print(f"[i] Loaded {lang}: {mat.shape}  ← {path}")
        pairs.append((lang, mat))
    plot_multilang_merged(
        lang_matrix_pairs=pairs,
        num_shots=num_shots,
        output_path=output_path,
        model_name=model_name,
        figsize=figsize,
    )


def plot_icl_vs_bare(
    icl_en_matrix: np.ndarray,
    icl_tgt_matrix: np.ndarray,
    target_lang: str,
    num_shots: int,
    output_path: str,
    model_name: str = "",
) -> None:
    assert icl_en_matrix.shape == icl_tgt_matrix.shape

    num_layers  = icl_en_matrix.shape[1]
    layers      = np.arange(num_layers)
    color_en    = _LANG_COLORS[0]
    color_tgt   = _LANG_COLORS[1]
    color_delta = "#555555"
    color_ratio = "#762a83"
    model_tag   = model_name.split("/")[-1] if model_name else ""

    with plt.rc_context(_STYLE_PARAMS):
        fig, (ax_abs, ax_diff, ax_ratio) = plt.subplots(
            1, 3, figsize=(18, 5),
            gridspec_kw={"wspace": 0.32},
        )
        fig.patch.set_facecolor("white")

        _setup_ax(ax_abs,   num_layers, "Cosine distance  (bare vs. ICL)")
        _setup_ax(ax_diff,  num_layers, "Δ cosine distance  (TGT-shot − EN-shot)")
        _setup_ax(ax_ratio, num_layers, "Ratio  (TGT-shot / EN-shot)")

        # ── Panel 1: absolute shift ───────────────────────────────────────
        m_en, _ = _draw_mean_std(
            ax_abs, layers, icl_en_matrix, color_en,
            label=f"Bare → EN-shot  (n={icl_en_matrix.shape[0]})",
        )
        m_tgt, _ = _draw_mean_std(
            ax_abs, layers, icl_tgt_matrix, color_tgt,
            label=f"Bare → {target_lang.upper()}-shot",
        )
        ax_abs.legend(loc="upper left")
        _expand_ylim(ax_abs)
        mgr_abs = AnnotationManager(ax_abs, legend_loc="upper left", n_legend_rows=2)
        _annotate_peak(ax_abs, layers, m_en,  color_en,  mgr_abs)
        _annotate_peak(ax_abs, layers, m_tgt, color_tgt, mgr_abs)
        ax_abs.set_title(
            f"Bare vs. ICL hidden-state shift\n"
            f"{model_tag}  |  {num_shots}-shot  |  {target_lang.upper()}",
            fontsize=10, pad=6,
        )

        # ── Panel 2: absolute difference TGT − EN ────────────────────────
        ax_diff.axhline(0, color="#aaaaaa", linewidth=0.8, linestyle="--", zorder=1)

        delta      = icl_tgt_matrix - icl_en_matrix
        mean_delta = delta.mean(axis=0)
        std_delta  = delta.std(axis=0)
        ax_diff.fill_between(layers,
                             mean_delta - std_delta, mean_delta + std_delta,
                             color=color_delta, alpha=0.12, linewidth=0)
        ax_diff.plot(layers, mean_delta,
                     color=color_delta, linewidth=1.6,
                     marker="s", markersize=3.0,
                     markerfacecolor="white", markeredgecolor=color_delta,
                     markeredgewidth=1.1,
                     label=f"TGT − EN  (n={delta.shape[0]})", zorder=4)
        ax_diff.legend(loc="upper left")
        _expand_ylim(ax_diff)
        mgr_diff = AnnotationManager(ax_diff, legend_loc="upper left", n_legend_rows=1)
        _annotate_peak(ax_diff, layers, mean_delta, color_delta, mgr_diff)
        ax_diff.set_title(
            f"Absolute difference  (TGT − EN)\n"
            f"{model_tag}  |  {num_shots}-shot  |  {target_lang.upper()}",
            fontsize=10, pad=6,
        )

        # ── Panel 3: ratio TGT / EN ───────────────────────────────────────
        mean_en  = icl_en_matrix.mean(axis=0)
        mean_tgt = icl_tgt_matrix.mean(axis=0)
        ratio, valid_mask = _compute_ratio(mean_tgt, mean_en)
        _draw_ratio(
            ax_ratio, layers, ratio, valid_mask,
            color=color_ratio,
            label=f"TGT / EN  (n={icl_en_matrix.shape[0]})",
        )
        ax_ratio.legend(loc="upper left")
        _expand_ylim(ax_ratio)
        if np.any(valid_mask):
            mgr_ratio = AnnotationManager(ax_ratio, legend_loc="upper left", n_legend_rows=1)
            _annotate_peak(ax_ratio, layers, ratio, color_ratio, mgr_ratio)
        ax_ratio.set_title(
            f"Ratio  (TGT / EN)\n"
            f"{model_tag}  |  {num_shots}-shot  |  {target_lang.upper()}\n"
            f"grey bands = EN near zero (unreliable)",
            fontsize=10, pad=6,
        )

        fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
        print(f"[✓] ICL-vs-bare plot saved → {output_path}")

        base, ext = os.path.splitext(output_path)
        _save_panel(fig, ax_abs,   f"{base}_panel1_abs{ext}")
        _save_panel(fig, ax_diff,  f"{base}_panel2_diff{ext}")
        _save_panel(fig, ax_ratio, f"{base}_panel3_ratio{ext}")

        plt.close(fig)


def plot_icl_vs_bare_multilang(
    lang_en_pairs:  List[tuple],
    lang_tgt_pairs: List[tuple],
    num_shots: int,
    output_path: str,
    model_name: str = "",
    figsize: tuple = (22, 5.5),
) -> None:
    assert len(lang_en_pairs) == len(lang_tgt_pairs)

    num_layers = lang_en_pairs[0][1].shape[1]
    layers     = np.arange(num_layers)
    model_tag  = model_name.split("/")[-1] if model_name else ""
    n_langs    = len(lang_en_pairs)

    with plt.rc_context(_STYLE_PARAMS):
        fig, (ax_abs, ax_diff, ax_ratio) = plt.subplots(
            1, 3, figsize=figsize,
            gridspec_kw={"wspace": 0.32},
        )
        fig.patch.set_facecolor("white")

        _setup_ax(ax_abs,   num_layers, "Cosine distance  (bare vs. ICL)")
        _setup_ax(ax_diff,  num_layers, "Δ cosine distance  (TGT-shot − EN-shot)")
        _setup_ax(ax_ratio, num_layers, "Ratio  (TGT-shot / EN-shot)")

        ax_diff.axhline(0,    color="#aaaaaa", linewidth=0.8, linestyle="--", zorder=1)
        ax_ratio.axhline(1.0, color="#aaaaaa", linewidth=0.8, linestyle="--", zorder=1)

        # Collected for deferred annotation (must annotate after y-limits are final)
        abs_data:    list[tuple] = []
        delta_means: list[tuple] = []
        ratios_data: list[tuple] = []

        # ── First pass: draw all curves ───────────────────────────────────
        for idx, ((lang, mat_en), (_, mat_tgt)) in enumerate(
            zip(lang_en_pairs, lang_tgt_pairs)
        ):
            if mat_en.shape[1] != num_layers or mat_tgt.shape[1] != num_layers:
                raise ValueError(f"Language '{lang}' matrix has unexpected layer count.")
            color    = _LANG_COLORS[idx % len(_LANG_COLORS)]
            n        = mat_en.shape[0]
            mean_en  = mat_en.mean(axis=0)
            std_en   = mat_en.std(axis=0)
            mean_tgt = mat_tgt.mean(axis=0)
            std_tgt  = mat_tgt.std(axis=0)

            # Panel 1: absolute shift
            for mean, std, ls, mk in [
                (mean_en,  std_en,  "-",  "o"),
                (mean_tgt, std_tgt, "--", "D"),
            ]:
                ax_abs.fill_between(layers, mean - std, mean + std,
                                    color=color, alpha=0.07, linewidth=0)
                lbl_suffix = "EN" if ls == "-" else "TGT"
                ax_abs.plot(layers, mean,
                            color=color, linewidth=1.5, linestyle=ls,
                            marker=mk, markersize=2.5,
                            markerfacecolor="white", markeredgecolor=color,
                            markeredgewidth=0.9,
                            label=f"{lang.upper()} {lbl_suffix}  (n={n})", zorder=4)
            abs_data.append((color, mean_en, mean_tgt))

            # Panel 2: TGT − EN delta
            delta      = mat_tgt - mat_en
            mean_delta = delta.mean(axis=0)
            std_delta  = delta.std(axis=0)
            ax_diff.fill_between(layers,
                                 mean_delta - std_delta, mean_delta + std_delta,
                                 color=color, alpha=0.10, linewidth=0)
            ax_diff.plot(layers, mean_delta,
                         color=color, linewidth=1.6,
                         marker="s", markersize=2.8,
                         markerfacecolor="white", markeredgecolor=color,
                         markeredgewidth=1.0,
                         label=f"{lang.upper()}  (n={n})", zorder=4)
            delta_means.append((color, mean_delta))

            # Panel 3: ratio
            ratio, valid_mask = _compute_ratio(mean_tgt, mean_en)
            _draw_ratio(
                ax_ratio, layers, ratio, valid_mask,
                color=color,
                label=f"{lang.upper()}  (n={n})",
            )
            ratios_data.append((color, ratio, valid_mask))

        # ── Legends + y-limit expansion (must precede AnnotationManager) ──
        ax_abs.legend(
            loc="upper left",
            ncol=max(1, n_langs // 2), fontsize=7,
        )
        ax_diff.legend(
            loc="upper left",
            ncol=max(1, n_langs // 4), fontsize=8,
        )
        ax_ratio.legend(
            loc="upper left",
            ncol=max(1, n_langs // 4), fontsize=8,
        )

        for ax in (ax_abs, ax_diff, ax_ratio):
            _expand_ylim(ax)

        # ── AnnotationManagers (created after final y-limits) ─────────────
        # Panel 1 has 2 curves per language; panel 2 and 3 have 1 each.
        mgr_abs   = AnnotationManager(ax_abs,   legend_loc="upper left", n_legend_rows=2 * n_langs)
        mgr_diff  = AnnotationManager(ax_diff,  legend_loc="upper left", n_legend_rows=n_langs)
        mgr_ratio = AnnotationManager(ax_ratio, legend_loc="upper left", n_legend_rows=n_langs)

        # ── Second pass: annotate ─────────────────────────────────────────
        for color, mean_en, mean_tgt in abs_data:
            _annotate_peak(ax_abs, layers, mean_en,  color, mgr_abs)
            _annotate_peak(ax_abs, layers, mean_tgt, color, mgr_abs)

        for color, mean_delta in delta_means:
            _annotate_peak(ax_diff, layers, mean_delta, color, mgr_diff)

        for color, ratio, valid_mask in ratios_data:
            if np.any(valid_mask):
                _annotate_peak(ax_ratio, layers, ratio, color, mgr_ratio)

        # ── Titles ────────────────────────────────────────────────────────
        lang_codes = ", ".join(l.upper() for l, _ in lang_en_pairs)

        ax_abs.set_title(
            f"Raw ICL shift  (Bare → EN/TGT)\n"
            f"{model_tag}  |  {num_shots}-shot  |  {lang_codes}\n"
            "solid = EN-shot,  dashed = TGT-shot",
            fontsize=10, pad=6,
        )
        ax_diff.set_title(
            f"Absolute difference  (TGT − EN)\n"
            f"{model_tag}  |  {num_shots}-shot  |  {lang_codes}",
            fontsize=10, pad=6,
        )
        ax_ratio.set_title(
            f"Ratio  (TGT / EN)\n"
            f"{model_tag}  |  {num_shots}-shot  |  {lang_codes}\n"
            "grey bands = EN near zero (unreliable)",
            fontsize=10, pad=6,
        )

        fig.tight_layout()
        fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
        print(f"[✓] Multi-language ICL-vs-bare merged plot saved → {output_path}")

        base, ext = os.path.splitext(output_path)
        _save_panel(fig, ax_abs,   f"{base}_panel1_abs{ext}")
        _save_panel(fig, ax_diff,  f"{base}_panel2_diff{ext}")
        _save_panel(fig, ax_ratio, f"{base}_panel3_ratio{ext}")

        plt.close(fig)


def plot_icl_vs_bare_multilang_from_files(
    npy_dir: str,
    langs: List[str],
    num_shots: int,
    output_path: str,
    model_name: str = "",
    figsize: tuple = (13, 5.5),
) -> None:
    lang_en_pairs:  List[tuple] = []
    lang_tgt_pairs: List[tuple] = []

    for lang in langs:
        en_path  = os.path.join(npy_dir, f"icl_en_matrix_{lang}.npy")
        tgt_path = os.path.join(npy_dir, f"icl_tgt_matrix_{lang}.npy")
        for p in (en_path, tgt_path):
            if not os.path.isfile(p):
                raise FileNotFoundError(f"Missing: {p}  (run extraction first)")
        mat_en  = np.load(en_path)
        mat_tgt = np.load(tgt_path)
        print(f"[i] Loaded {lang} EN : {mat_en.shape}  ← {en_path}")
        print(f"[i] Loaded {lang} TGT: {mat_tgt.shape}  ← {tgt_path}")
        lang_en_pairs.append( (lang, mat_en))
        lang_tgt_pairs.append((lang, mat_tgt))

    plot_icl_vs_bare_multilang(
        lang_en_pairs  = lang_en_pairs,
        lang_tgt_pairs = lang_tgt_pairs,
        num_shots      = num_shots,
        output_path    = output_path,
        model_name     = model_name,
        figsize        = figsize,
    )


# ──────────────────────────────────────────────
# 7.  CLI entry point
# ──────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Layerwise cross-lingual hidden-state difference visualizer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g = p.add_argument_group("Data files (parallel CSVs) — required for extraction")
    g.add_argument("--train_tgt", type=str, default=None)
    g.add_argument("--train_en",  type=str, default=None)
    g.add_argument("--val_tgt",   type=str, default=None)

    p.add_argument("--model_name",   type=str,
                   default="meta-llama/Meta-Llama-3-8B-Instruct")
    p.add_argument("--load_in_8bit", action="store_true")
    p.add_argument("--target_lang",  type=str, default="zh")
    p.add_argument("--num_shots",    type=int, default=3)
    p.add_argument("--num_samples",  type=int, default=50)
    p.add_argument("--batch_size",   type=int, default=8)
    p.add_argument("--max_length",   type=int, default=2048)
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--output_dir",   type=str, default="./output")
    p.add_argument("--device",       type=str, default=None)

    p.add_argument(
        "--skip_extraction", action="store_true",
        help=(
            "Load diff matrices from --output_dir instead of running model "
            "forward passes. Requires save_results() to have been called in a "
            "prior run with the same --output_dir and --target_lang."
        ),
    )

    mg = p.add_argument_group("Multi-language cross-lingual merged plot")
    mg.add_argument("--merge_langs",    type=str, nargs="+", default=None)
    mg.add_argument("--merge_npy_dir",  type=str, default=None)
    mg.add_argument("--merge_output",   type=str, default=None)

    mi = p.add_argument_group("Multi-language ICL-vs-bare merged plot")
    mi.add_argument("--merge_icl_langs",   type=str, nargs="+", default=None)
    mi.add_argument("--merge_icl_npy_dir", type=str, default=None)
    mi.add_argument("--merge_icl_output",  type=str, default=None)

    return p.parse_args()


def _resolve_device(requested: Optional[str]) -> str:
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.merge_langs:
        npy_dir   = args.merge_npy_dir or args.output_dir
        npy_paths = {
            lang: os.path.join(npy_dir, f"diff_matrix_{lang}.npy")
            for lang in args.merge_langs
        }
        missing = [p for p in npy_paths.values() if not os.path.isfile(p)]
        if missing:
            raise FileNotFoundError(
                "Missing .npy files (run extraction first):\n  " +
                "\n  ".join(missing)
            )
        merge_out = args.merge_output or os.path.join(
            args.output_dir,
            "merged_" + "_".join(args.merge_langs) + ".png",
        )
        plot_multilang_from_files(
            npy_paths=npy_paths, num_shots=args.num_shots,
            output_path=merge_out, model_name=args.model_name,
        )

    if args.merge_icl_langs:
        npy_dir   = args.merge_icl_npy_dir or args.output_dir
        merge_out = args.merge_icl_output or os.path.join(
            args.output_dir,
            "icl_vs_bare_merged_" + "_".join(args.merge_icl_langs) + ".png",
        )
        plot_icl_vs_bare_multilang_from_files(
            npy_dir=npy_dir, langs=args.merge_icl_langs,
            num_shots=args.num_shots, output_path=merge_out,
            model_name=args.model_name,
        )

    if args.merge_langs or args.merge_icl_langs:
        return

    if args.skip_extraction:
        print("[i] --skip_extraction set: loading saved matrices …")
        diff_matrix, icl_en_matrix, icl_tgt_matrix = load_results(
            output_dir=args.output_dir,
            target_lang=args.target_lang,
        )
    else:
        if not (args.train_tgt and args.train_en and args.val_tgt):
            raise ValueError(
                "Provide --train_tgt, --train_en, and --val_tgt for extraction, "
                "or use --skip_extraction / --merge_langs / --merge_icl_langs "
                "to plot from saved files."
            )

        device = _resolve_device(args.device)
        print(f"[i] Using device: {device}")

        print(f"[i] Loading tokenizer: {args.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if tokenizer.chat_template is None:
            warnings.warn(
                f"Model '{args.model_name}' has no chat_template. "
                "Are you sure this is an instruction-tuned model? "
                "Falling back to Alpaca-style formatting.",
                UserWarning,
            )

        model_kwargs: Dict = {
            "torch_dtype": "auto",
            "device_map":  "auto" if device == "cuda" else None,
        }
        if args.load_in_8bit:
            model_kwargs["load_in_8bit"] = True

        print(f"[i] Loading model: {args.model_name}")
        model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
        if device not in ("cuda",):
            model = model.to(device)
        model.eval()

        print("[i] Loading datasets …")
        train_tgt, train_en, val_tgt = load_splits(
            train_tgt_path=args.train_tgt,
            train_en_path=args.train_en,
            val_tgt_path=args.val_tgt,
        )

        diff_matrix, icl_en_matrix, icl_tgt_matrix = compute_crosslingual_diffs(
            model=model, tokenizer=tokenizer,
            train_tgt=train_tgt, train_en=train_en, val_tgt=val_tgt,
            target_lang=args.target_lang, num_shots=args.num_shots,
            num_samples=args.num_samples, device=device,
            seed=args.seed, max_length=args.max_length, batch_size=args.batch_size,
        )

        save_results(
            diff_matrix=diff_matrix, output_dir=args.output_dir,
            target_lang=args.target_lang,
            icl_en_matrix=icl_en_matrix, icl_tgt_matrix=icl_tgt_matrix,
        )

    print(f"[i] diff_matrix shape     : {diff_matrix.shape}")
    print(f"[i] icl_en_matrix shape   : {icl_en_matrix.shape}")
    print(f"[i] icl_tgt_matrix shape  : {icl_tgt_matrix.shape}")

    plot_crosslingual_diff(
        diff_matrix=diff_matrix, target_lang=args.target_lang,
        num_shots=args.num_shots,
        output_path=os.path.join(
            args.output_dir,
            f"crosslingual_diff_{args.target_lang}_{args.num_shots}shot.png",
        ),
        model_name=args.model_name,
    )

    plot_icl_vs_bare(
        icl_en_matrix=icl_en_matrix, icl_tgt_matrix=icl_tgt_matrix,
        target_lang=args.target_lang, num_shots=args.num_shots,
        output_path=os.path.join(
            args.output_dir,
            f"icl_vs_bare_{args.target_lang}_{args.num_shots}shot.png",
        ),
        model_name=args.model_name,
    )

    print("\n[✓] Done.")


if __name__ == "__main__":
    main()