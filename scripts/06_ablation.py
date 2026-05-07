"""
Fase 6 - Estudio de ablacion E0-E5

E0: Baseline (SleepApneaCNN FP32)
E1: Pruning 50% sin KD (FP32)
E2: Cuantizacion INT8 sola (baseline INT8)
E3: Pruning 50% sin KD + INT8
E4: Pruning 50% + KD (FP32)
E5: Pruning 50% + KD + INT8   <- configuracion optima

Salida: results/ablation_metrics.json
"""

import json
from pathlib import Path

RESULTS_DIR = Path("results")
EVAL_FILE   = RESULTS_DIR / "metrics_eval.json"
OUT_FILE    = RESULTS_DIR / "ablation_metrics.json"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    with open(EVAL_FILE) as f:
        ev = json.load(f)

    def get(model: str, quant: str, split: str, metric: str) -> float:
        return ev[model]["metrics"][quant][split][metric]

    def lat(model: str, quant: str) -> float:
        return ev[model]["latency"][quant]["mean_ms"]

    def kb(model: str, quant: str) -> float:
        key = "fp32" if quant == "fp32" else "int8"
        return ev[model]["disk_kb"][key]

    def params(model: str) -> int:
        return ev[model]["n_params"]

    baseline_auc = get("baseline", "fp32", "test", "auc_roc")

    rows = [
        # (id, label, model, quant)
        ("E0", "Baseline",             "baseline",  "fp32"),
        ("E1", "Pruning 50% (no KD)",  "p050_nokd", "fp32"),
        ("E2", "Quant INT8 sola",      "baseline",  "int8"),
        ("E3", "Pruning 50% + INT8",   "p050_nokd", "int8"),
        ("E4", "Pruning 50% + KD",     "p050_kd",   "fp32"),
        ("E5", "Pruning + KD + INT8",  "p050_kd",   "int8"),
    ]

    ablation = {}
    print("=" * 80)
    print("Ablacion E0-E5")
    print("=" * 80)
    hdr = (f"{'ID':<4} {'Configuracion':<22} {'Params':>8} {'KB':>7} "
           f"{'Lat(ms)':>9} {'AUC':>7} {'dAUC':>7} {'Speedup':>8}")
    print(hdr)
    print("-" * len(hdr))

    e0_lat = lat("baseline", "fp32")
    for eid, label, model, quant in rows:
        n   = params(model)
        k   = kb(model, quant)
        l   = lat(model, quant)
        auc = get(model, quant, "test", "auc_roc")
        f1  = get(model, quant, "test", "f1")
        acc = get(model, quant, "test", "accuracy")
        d_auc   = round(auc - baseline_auc, 4)
        speedup = round(e0_lat / l, 2)

        ablation[eid] = {
            "label":     label,
            "model":     model,
            "quantized": quant == "int8",
            "n_params":  n,
            "disk_kb":   k,
            "lat_ms":    l,
            "speedup_vs_baseline": speedup,
            "test": {
                "accuracy": acc,
                "f1":       f1,
                "auc_roc":  auc,
                "delta_auc": d_auc,
            },
        }
        sign = "+" if d_auc >= 0 else ""
        print(f"{eid:<4} {label:<22} {n:>8,} {k:>7.1f} "
              f"{l:>9.3f} {auc:>7.4f} {sign}{d_auc:>6.4f} {speedup:>7.2f}x")

    with open(OUT_FILE, "w") as f:
        json.dump(ablation, f, indent=2)
    print(f"\nGuardado en {OUT_FILE}")
    print("Fase 6 completada.")


if __name__ == "__main__":
    main()
