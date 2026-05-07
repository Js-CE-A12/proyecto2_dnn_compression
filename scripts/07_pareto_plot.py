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


def make_pareto_plot(ev: dict) -> None:
    configs = [
        # (label, model, quant, color, marker)
        ("Baseline FP32",        "baseline",  "fp32", "#1f77b4", "o"),
        ("Baseline INT8",        "baseline",  "int8", "#aec7e8", "s"),
        ("p050 nokd FP32",       "p050_nokd", "fp32", "#ff7f0e", "o"),
        ("p050 nokd INT8",       "p050_nokd", "int8", "#ffbb78", "s"),
        ("p050 kd FP32",         "p050_kd",   "fp32", "#2ca02c", "o"),
        ("p050 kd INT8",         "p050_kd",   "int8", "#98df8a", "s"),
        ("p070 nokd FP32",       "p070_nokd", "fp32", "#d62728", "o"),
        ("p070 nokd INT8",       "p070_nokd", "int8", "#ff9896", "s"),
        ("p070 kd FP32",         "p070_kd",   "fp32", "#9467bd", "o"),
        ("p070 kd INT8",         "p070_kd",   "int8", "#c5b0d5", "s"),
    ]

    lats, aucs, labels, colors, markers = [], [], [], [], []
    for label, model, quant, color, marker in configs:
        lats.append(ev[model]["latency"][quant]["mean_ms"])
        aucs.append(ev[model]["metrics"][quant]["test"]["auc_roc"])
        labels.append(label)
        colors.append(color)
        markers.append(marker)

    points = list(zip(lats, aucs))
    pareto_idx = pareto_frontier(points)

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (lat, auc, lbl, col, mrk) in enumerate(
            zip(lats, aucs, labels, colors, markers)):
        is_pareto = i in pareto_idx
        ax.scatter(lat, auc, c=col, marker=mrk,
                   s=120 if is_pareto else 70,
                   zorder=3,
                   edgecolors="black" if is_pareto else "none",
                   linewidths=1.5)
        ax.annotate(lbl, (lat, auc),
                    textcoords="offset points", xytext=(5, 3),
                    fontsize=7, color=col)

    # Conectar puntos Pareto
    par_pts = sorted([(lats[i], aucs[i]) for i in pareto_idx])
    if par_pts:
        px, py = zip(*par_pts)
        ax.plot(px, py, "k--", linewidth=1, alpha=0.5, label="Pareto frontier")

    fp32_patch = mpatches.Patch(color="gray", label="FP32 (circulo)")
    int8_patch = mpatches.Patch(color="lightgray", label="INT8 (cuadrado)")
    pareto_patch = mpatches.Patch(color="black", label="Frontera Pareto")
    ax.legend(handles=[fp32_patch, int8_patch, pareto_patch], fontsize=8)

    ax.set_xlabel("Latencia media por muestra (ms) - CPU", fontsize=11)
    ax.set_ylabel("AUC-ROC en test", fontsize=11)
    ax.set_title("Pareto: AUC-ROC vs Latencia — UCDDB Apnea Detection", fontsize=12)
    ax.grid(True, alpha=0.3)

    out = FIGURES_DIR / "pareto.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
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

    make_pareto_plot(ev)
    make_ablation_table(ablation)
    print("Fase 7 completada.")


if __name__ == "__main__":
    main()
