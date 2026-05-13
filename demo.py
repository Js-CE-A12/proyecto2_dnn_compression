"""
demo.py — Demostracion rapida del pipeline de compresion DNN
Requiere que el pipeline ya haya sido ejecutado (checkpoints + data procesada).

Ejecutar desde la raiz del proyecto:
    python demo.py

Muestra:
  - Metricas de clasificacion: Accuracy, Macro F1, Cohen Kappa, AUC-ROC
  - Comparacion FP32 vs INT8 con verificacion +/-0.5%
  - Tamano en disco de cada modelo
  - Speedup de latencia
"""

import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, cohen_kappa_score,
)

# ---------------------------------------------------------------------------
# Rutas relativas al script (funciona desde cualquier directorio)
ROOT          = Path(__file__).resolve().parent
CKPT_DIR      = ROOT / "checkpoints"
PROCESSED_DIR = ROOT / "data" / "processed"
ISRUC_DIR     = PROCESSED_DIR / "isruc_subjects"
RESULTS_DIR   = ROOT / "results"
WARMUP        = 20
RUNS          = 200
THRESHOLD_PCT = 0.5          # tolerancia maxima FP32 vs INT8 (%)

# ---------------------------------------------------------------------------

def load_isruc_test():
    Xs, ys = [], []
    for npz in sorted(ISRUC_DIR.glob("subj_*.npz"),
                      key=lambda p: int(p.stem.split("_")[1])):
        d = np.load(str(npz))
        Xs.append(d["X"]); ys.append(d["y_apnea"])
    return np.concatenate(Xs).astype(np.float32), np.concatenate(ys)


def make_session(path: Path) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=opts,
                                providers=["CPUExecutionProvider"])


def predict(sess: ort.InferenceSession,
            x: np.ndarray, batch: int = 128):
    probs, preds = [], []
    for i in range(0, len(x), batch):
        xb  = x[i:i+batch]
        out = sess.run(None, {"eeg": xb})[0]
        e   = np.exp(out - out.max(axis=1, keepdims=True))
        p   = e / e.sum(axis=1, keepdims=True)
        probs.append(p[:, 1])
        preds.append(out.argmax(axis=1))
    return np.concatenate(probs), np.concatenate(preds)


def compute_metrics(y_true, probs, preds) -> dict:
    return {
        "accuracy":   round(float(accuracy_score(y_true, preds)),                  4),
        "macro_f1":   round(float(f1_score(y_true, preds, average="macro",
                                           zero_division=0)),                       4),
        "kappa":      round(float(cohen_kappa_score(y_true, preds)),               4),
        "auc_roc":    round(float(roc_auc_score(y_true, probs)),                   4),
    }


def bench_latency(sess: ort.InferenceSession) -> float:
    dummy = np.random.randn(1, 1, 3000).astype(np.float32)
    for _ in range(WARMUP):
        sess.run(None, {"eeg": dummy})
    t0 = time.perf_counter()
    for _ in range(RUNS):
        sess.run(None, {"eeg": dummy})
    return (time.perf_counter() - t0) / RUNS * 1000   # ms


def disk_kb(path: Path) -> float:
    return round(path.stat().st_size / 1024, 1)


def check_tolerance(name: str, fp32_val: float, int8_val: float):
    diff_pct = abs(fp32_val - int8_val) * 100
    status   = "OK" if diff_pct <= THRESHOLD_PCT else "ALERTA"
    return diff_pct, status


def print_separator(char="=", n=70):
    print(char * n)


# ---------------------------------------------------------------------------
# Modelos a evaluar
# ---------------------------------------------------------------------------
MODELS = [
    {
        "name":  "Baseline",
        "fp32":  CKPT_DIR / "baseline_fp32.onnx",
        "int8":  CKPT_DIR / "baseline_int8.onnx",
    },
    {
        "name":  "P50+KD (optimo)",
        "fp32":  CKPT_DIR / "p050_kd" / "model_fp32.onnx",
        "int8":  CKPT_DIR / "p050_kd" / "model_int8.onnx",
    },
]

# ---------------------------------------------------------------------------

def main():
    print_separator()
    print("  DEMO - Compresion DNN para Deteccion de Apnea del Sueno")
    print("  Universidad Interamericana PR - Bayamon")
    print_separator()

    # Cargar test set ISRUC
    print("\nCargando test set cross-dataset (ISRUC)...")
    x_test, y_test = load_isruc_test()
    print(f"  {len(x_test):,} epocas  |  apnea={y_test.mean():.1%}")

    all_results = {}

    for cfg in MODELS:
        name = cfg["name"]
        fp32_path, int8_path = cfg["fp32"], cfg["int8"]

        if not fp32_path.exists() or not int8_path.exists():
            print(f"\n  [SKIP] {name}: archivos ONNX no encontrados.")
            continue

        print_separator("-")
        print(f"  Modelo: {name}")
        print_separator("-")

        # --- Inferencia ---
        sess_fp32 = make_session(fp32_path)
        sess_int8 = make_session(int8_path)

        probs_fp32, preds_fp32 = predict(sess_fp32, x_test)
        probs_int8, preds_int8 = predict(sess_int8, x_test)

        m_fp32 = compute_metrics(y_test, probs_fp32, preds_fp32)
        m_int8 = compute_metrics(y_test, probs_int8, preds_int8)

        # --- Latencia ---
        lat_fp32 = bench_latency(sess_fp32)
        lat_int8 = bench_latency(sess_int8)
        speedup  = lat_fp32 / lat_int8

        # --- Disco ---
        kb_fp32 = disk_kb(fp32_path)
        kb_int8 = disk_kb(int8_path)

        # --- Imprimir tabla de metricas ---
        metric_names = {
            "accuracy": "Accuracy",
            "macro_f1": "Macro F1",
            "kappa":    "Cohen Kappa",
            "auc_roc":  "AUC-ROC",
        }
        print(f"\n  {'Metrica':<16} {'FP32':>8} {'INT8':>8} {'Diff %':>8}")
        print(f"  {'-'*44}")
        for key, label in metric_names.items():
            diff_pct, _ = check_tolerance(key, m_fp32[key], m_int8[key])
            print(f"  {label:<16} {m_fp32[key]:>8.4f} {m_int8[key]:>8.4f} "
                  f"{diff_pct:>7.3f}%")

        # --- Tamano en disco ---
        print(f"\n  Tamano en disco:")
        print(f"    FP32  : {kb_fp32:>8.1f} KB")
        print(f"    INT8  : {kb_int8:>8.1f} KB")
        print(f"    Ratio : {kb_fp32/kb_int8:>8.2f}x reduccion")

        # --- Latencia ---
        print(f"\n  Latencia CPU (1 muestra, {RUNS} runs):")
        print(f"    FP32  : {lat_fp32:>8.3f} ms")
        print(f"    INT8  : {lat_int8:>8.3f} ms")
        print(f"    Speedup: {speedup:>7.2f}x")

        all_results[name] = {
            "fp32": m_fp32, "int8": m_int8,
            "lat_fp32_ms": round(lat_fp32, 3),
            "lat_int8_ms": round(lat_int8, 3),
            "speedup":     round(speedup, 2),
            "disk_kb":     {"fp32": kb_fp32, "int8": kb_int8,
                            "ratio": round(kb_fp32 / kb_int8, 2)},
        }

    # --- Resumen comparativo final ---
    print_separator()
    print("  RESUMEN COMPARATIVO (test cross-dataset ISRUC)")
    print_separator()
    hdr = f"  {'Modelo':<20} {'Precision':>10} {'Macro F1':>10} {'Kappa':>8} {'AUC':>8} {'KB INT8':>9} {'Speedup':>9}"
    print(hdr)
    print(f"  {'-'*(len(hdr)-2)}")

    for name, r in all_results.items():
        i = r["int8"]
        print(f"  {name:<20} {i['accuracy']:>10.4f} {i['macro_f1']:>10.4f} "
              f"{i['kappa']:>8.4f} {i['auc_roc']:>8.4f} "
              f"{r['disk_kb']['int8']:>9.1f} {r['speedup']:>9.2f}x")

    print_separator()

    # Guardar resultados
    out = RESULTS_DIR / "demo_metrics.json"
    RESULTS_DIR.mkdir(exist_ok=True)
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Resultados guardados en: {out}")
    print_separator()


if __name__ == "__main__":
    main()
