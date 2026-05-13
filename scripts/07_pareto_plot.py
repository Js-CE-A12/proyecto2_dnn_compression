"""
Fase 7 - Curva de Pareto y tablas de resultados

Genera:
  results/figures/pareto.png        - AUC-ROC vs latencia (Pareto frontier)
  results/figures/ablation_table.png - tabla ablacion E0-E5
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

RESULTS_DIR = Path("results")
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

EVAL_FILE     = RESULTS_DIR / "metrics_eval.json"
ABLATION_FILE = RESULTS_DIR / "ablation_metrics.json"


# ---------------------------------------------------------------------------
# Pareto frontier
# ---------------------------------------------------------------------------

def pareto_frontier(points: list) -> list:
    """Retorna indices de puntos en la frontera de Pareto (max AUC, min lat)."""
    dominated = [False] * len(points)
    for i, (lat_i, auc_i) in enumerate(points):
        for j, (lat_j, auc_j) in enumerate(points):
            if i == j:
                continue
            if lat_j <= lat_i and auc_j >= auc_i and (lat_j < lat_i or auc_j > auc_i):
                dominated[i] = True
                break
    return [i for i, d in enumerate(dominated) if not d]


def make_pareto_plot(ablation: dict) -> None:
    eids   = ["E0", "E1", "E2", "E3", "E4", "E5"]
    lats   = [ablation[e]["lat_ms"]          for e in eids]
    aucs   = [ablation[e]["test"]["auc_roc"] for e in eids]
    kbs    = [ablation[e]["disk_kb"]         for e in eids]
    labels = [ablation[e]["label"]           for e in eids]
    quants = [ablation[e]["quantized"]       for e in eids]

    COLORS  = ["#1565C0", "#E65100", "#5C6BC0", "#BF360C", "#2E7D32", "#1B5E20"]
    MARKERS = ["o" if not q else "s" for q in quants]
    max_kb  = max(kbs)
    sizes   = [180 + 650 * (k / max_kb) for k in kbs]

    points     = list(zip(lats, aucs))
    pareto_idx = pareto_frontier(points)

    # Two subplots: full view (left) + zoom on dense cluster (right)
    fig, (ax, axz) = plt.subplots(1, 2, figsize=(14, 6),
                                   gridspec_kw={"width_ratios": [3, 2]})
    fig.patch.set_facecolor("#FAFAFA")

    for ax_ in (ax, axz):
        ax_.set_facecolor("#F8F8F8")
        ax_.grid(True, alpha=0.3, linestyle="--")

    # ---- Draw points on both axes ----
    for i, eid in enumerate(eids):
        is_p = i in pareto_idx
        kw   = dict(c=COLORS[i], marker=MARKERS[i], s=sizes[i],
                    zorder=4, alpha=0.88,
                    edgecolors="black" if is_p else "#888",
                    linewidths=2.5 if is_p else 0.8)
        ax.scatter(lats[i],  aucs[i],  **kw)
        axz.scatter(lats[i], aucs[i],  **kw)

    # Pareto line on both
    par_pts = sorted([(lats[i], aucs[i]) for i in pareto_idx])
    if par_pts:
        px, py = zip(*par_pts)
        for ax_ in (ax, axz):
            ax_.plot(px, py, "k--", lw=1.3, alpha=0.4, zorder=2)

    # ---- Annotations on full plot ----
    ann_full = {
        # eid: (offset_pts, ha, va)
        "E0": ((-10, -20), "center", "top"),
        "E1": ((-14, +16), "right",  "bottom"),
        "E2": ((+12, -16), "left",   "top"),
        "E3": ((-14, -16), "right",  "top"),
        "E4": ((+12, +14), "left",   "bottom"),
        "E5": ((-14, +20), "right",  "bottom"),
    }
    for i, eid in enumerate(eids):
        ox, oy = ann_full[eid][0]
        ha, va = ann_full[eid][1], ann_full[eid][2]
        is_p   = i in pareto_idx
        tag    = f"{eid} ★" if eid == "E5" else eid
        bbox   = dict(boxstyle="round,pad=0.25", fc="#C8E6C9",
                      ec="#2E7D32", lw=1.4) if eid == "E5" else None
        ax.annotate(
            f"{tag}\n{labels[i]}\n{kbs[i]:.0f} KB",
            xy=(lats[i], aucs[i]),
            xytext=(ox, oy), textcoords="offset points",
            ha=ha, va=va, fontsize=8,
            fontweight="bold" if is_p else "normal",
            color=COLORS[i],
            bbox=bbox,
            arrowprops=dict(arrowstyle="-|>", color=COLORS[i],
                            lw=1.1, mutation_scale=9),
        )

    # ---- Annotations on zoom plot (only E1-E5 visible) ----
    ann_zoom = {
        "E1": ((+10, +14), "left",  "bottom"),
        "E2": ((+10, -14), "left",  "top"),
        "E3": ((-10, -16), "right", "top"),
        "E4": ((+10, +14), "left",  "bottom"),
        "E5": ((-10, +18), "right", "bottom"),
    }
    for i, eid in enumerate(eids):
        if eid not in ann_zoom:
            continue
        ox, oy = ann_zoom[eid][0]
        ha, va = ann_zoom[eid][1], ann_zoom[eid][2]
        is_p   = i in pareto_idx
        tag    = f"{eid} ★" if eid == "E5" else eid
        bbox   = dict(boxstyle="round,pad=0.25", fc="#C8E6C9",
                      ec="#2E7D32", lw=1.4) if eid == "E5" else None
        axz.annotate(
            f"{tag}  {lats[i]:.3f} ms\nAUC {aucs[i]:.4f}",
            xy=(lats[i], aucs[i]),
            xytext=(ox, oy), textcoords="offset points",
            ha=ha, va=va, fontsize=8.5,
            fontweight="bold" if is_p else "normal",
            color=COLORS[i], bbox=bbox,
            arrowprops=dict(arrowstyle="-|>", color=COLORS[i],
                            lw=1.1, mutation_scale=9),
        )

    # ---- Axis limits ----
    ax.set_xlim(0.07, 0.46)
    ax.set_ylim(0.608, 0.638)
    axz.set_xlim(0.090, 0.215)
    axz.set_ylim(0.626, 0.635)

    # ---- Labels & titles ----
    ax.set_xlabel("Latencia media (ms) — CPU, 1 hilo", fontsize=11)
    ax.set_ylabel("AUC-ROC en test cross-dataset (ISRUC)", fontsize=11)
    ax.set_title("Vista completa (E0–E5)\nBurbuja proporcional al tamano en disco",
                 fontsize=10)

    axz.set_xlabel("Latencia media (ms)", fontsize=11)
    axz.set_ylabel("AUC-ROC", fontsize=11)
    axz.set_title("Zoom — cluster comprimido (E1–E5)", fontsize=10)

    # ---- Shared legend ----
    handles = [
        plt.Line2D([0],[0], marker="o", color="w", markerfacecolor="#555",
                   markersize=9, label="FP32"),
        plt.Line2D([0],[0], marker="s", color="w", markerfacecolor="#555",
                   markersize=9, label="INT8"),
        plt.Line2D([0],[0], color="k", ls="--", lw=1.3, label="Frontera Pareto"),
        mpatches.Patch(fc="#C8E6C9", ec="#2E7D32", label="Optimo (E5 ★)"),
    ]
    fig.legend(handles=handles, fontsize=9, loc="lower center",
               ncol=4, framealpha=0.9, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Curva de Pareto — Compresion DNN para Deteccion de Apnea EEG",
                 fontsize=13, fontweight="bold", y=1.01)

    fig.tight_layout()
    out = FIGURES_DIR / "pareto.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Pareto plot guardado: {out}")


# ---------------------------------------------------------------------------
# Tabla de ablacion
# ---------------------------------------------------------------------------

def make_ablation_table(ablation: dict) -> None:
    ids    = list(ablation.keys())
    labels = [ablation[e]["label"] for e in ids]
    params = [f"{ablation[e]['n_params']:,}" for e in ids]
    kbs    = [f"{ablation[e]['disk_kb']:.1f}" for e in ids]
    lats   = [f"{ablation[e]['lat_ms']:.3f}" for e in ids]
    speedups = [f"{ablation[e]['speedup_vs_baseline']:.2f}x" for e in ids]
    aucs   = [f"{ablation[e]['test']['auc_roc']:.4f}" for e in ids]
    diffs  = []
    for e in ids:
        d = ablation[e]["test"]["delta_auc"]
        diffs.append(f"{'+' if d >= 0 else ''}{d:.4f}")

    col_labels = ["ID", "Configuracion", "Params", "KB", "Lat (ms)", "Speedup", "AUC-ROC", "dAUC"]
    row_data   = list(zip(ids, labels, params, kbs, lats, speedups, aucs, diffs))

    fig, ax = plt.subplots(figsize=(13, 3.5))
    ax.axis("off")

    tbl = ax.table(
        cellText=row_data,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.6)

    # Colores alternos
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#ecf0f1")
        else:
            cell.set_facecolor("white")

    # Resaltar E5 (fila 6)
    for col in range(len(col_labels)):
        tbl[(6, col)].set_facecolor("#2ecc71")

    ax.set_title("Ablacion E0-E5: Pipeline de Compresion DNN", fontsize=12, pad=20)
    out = FIGURES_DIR / "ablation_table.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Tabla ablacion guardada: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Fase 7 - Graficas de resultados")
    print("=" * 60)

    with open(EVAL_FILE)     as f: ev = json.load(f)
    with open(ABLATION_FILE) as f: ablation = json.load(f)

    make_pareto_plot(ablation)
    make_ablation_table(ablation)
    print("Fase 7 completada.")


if __name__ == "__main__":
    main()
