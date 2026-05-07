"""
Fase 5 - Validacion en desktop
Evalua todos los modelos (baseline, variantes podadas, cuantizado) con:
  - Latencia ONNX RT: 50 warm-up + 500 inferencias -> media, std, P95 (1 muestra)
  - RAM del proceso: antes/despues de cargar el modelo
  - Tamano en disco (.pth y .onnx)
  - Metricas de clasificacion sobre test set completo

Salida: results/metrics_eval.json
"""

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import psutil
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,
)
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models import SleepApneaCNN
from src.pruning import PrunedSleepApneaCNN

PROCESSED_DIR = Path("data/processed")
CKPT_DIR      = Path("checkpoints")
RESULTS_DIR   = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

WARMUP = 50
RUNS   = 500
N_CALIB = 200


# ---------------------------------------------------------------------------
# Calibrador reutilizable
# ---------------------------------------------------------------------------

class EEGCalibReader(CalibrationDataReader):
    def __init__(self, x: np.ndarray, n: int) -> None:
        idx = np.random.choice(len(x), min(n, len(x)), replace=False)
        self._data = [{"eeg": x[i:i+1].astype(np.float32)} for i in idx]
        self._pos  = 0

    def get_next(self):
        if self._pos >= len(self._data):
            return None
        r = self._data[self._pos]; self._pos += 1; return r

    def rewind(self):
        self._pos = 0


# ---------------------------------------------------------------------------
# Helpers ONNX
# ---------------------------------------------------------------------------

def export_fp32(model: torch.nn.Module, path: Path) -> None:
    if path.exists():
        return
    dummy = torch.randn(1, 1, 3000)
    with torch.no_grad():
        torch.onnx.export(
            model, dummy, str(path),
            input_names=["eeg"], output_names=["logits"],
            dynamic_axes={"eeg": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=17, do_constant_folding=True, dynamo=False,
        )


def export_int8(fp32_path: Path, int8_path: Path, x_val: np.ndarray) -> None:
    if int8_path.exists():
        return
    calib = EEGCalibReader(x_val, N_CALIB)
    quantize_static(
        model_input=str(fp32_path), model_output=str(int8_path),
        calibration_data_reader=calib,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
        per_channel=False,
    )


def make_session(path: Path) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=opts,
                                providers=["CPUExecutionProvider"])


def bench_latency(sess: ort.InferenceSession) -> dict:
    dummy = np.random.randn(1, 1, 3000).astype(np.float32)
    for _ in range(WARMUP):
        sess.run(None, {"eeg": dummy})
    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        sess.run(None, {"eeg": dummy})
        times.append((time.perf_counter() - t0) * 1000)
    arr = np.array(times)
    return {
        "mean_ms": round(float(arr.mean()), 3),
        "std_ms":  round(float(arr.std()),  3),
        "p95_ms":  round(float(np.percentile(arr, 95)), 3),
    }


def eval_session(sess: ort.InferenceSession,
                 x: np.ndarray, y: np.ndarray,
                 batch: int = 64) -> dict:
    all_probs, all_preds = [], []
    for i in range(0, len(x), batch):
        xb  = x[i:i+batch].astype(np.float32)
        out = sess.run(None, {"eeg": xb})[0]
        exp = np.exp(out - out.max(axis=1, keepdims=True))
        probs = exp / exp.sum(axis=1, keepdims=True)
        all_probs.append(probs[:, 1])
        all_preds.append(out.argmax(axis=1))
    probs = np.concatenate(all_probs)
    preds = np.concatenate(all_preds)
    return {
        "accuracy": round(float(accuracy_score(y, preds)), 4),
        "f1":       round(float(f1_score(y, preds, zero_division=0)), 4),
        "auc_roc":  round(float(roc_auc_score(y, probs)), 4),
    }


def ram_mb() -> float:
    return psutil.Process().memory_info().rss / 1024 / 1024


# ---------------------------------------------------------------------------
# Evaluacion de un modelo completo
# ---------------------------------------------------------------------------

def evaluate_model(
    name: str,
    pth_path: Path,
    model_obj: torch.nn.Module,
    onnx_fp32: Path,
    onnx_int8: Path,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    quantize: bool = True,
) -> dict:
    print(f"\n  [{name}]")
    gc.collect()

    n_params = sum(p.numel() for p in model_obj.parameters())

    # Exportar
    export_fp32(model_obj, onnx_fp32)
    if quantize:
        export_int8(onnx_fp32, onnx_int8, x_val)

    # Tamanos en disco
    pth_kb  = round(pth_path.stat().st_size / 1024, 1) if pth_path.exists() else None
    fp32_kb = round(onnx_fp32.stat().st_size / 1024, 1)
    int8_kb = round(onnx_int8.stat().st_size / 1024, 1) if quantize else None

    # RAM al cargar FP32
    gc.collect(); ram_before = ram_mb()
    sess_fp32 = make_session(onnx_fp32)
    ram_fp32  = round(ram_mb() - ram_before, 1)

    # Latencia FP32
    lat_fp32 = bench_latency(sess_fp32)

    # Metricas FP32
    m_val_fp32  = eval_session(sess_fp32, x_val,  y_val)
    m_test_fp32 = eval_session(sess_fp32, x_test, y_test)

    result = {
        "n_params": n_params,
        "disk_kb":  {"pth": pth_kb, "fp32": fp32_kb, "int8": int8_kb},
        "ram_load_mb": {"fp32": ram_fp32},
        "latency":  {"fp32": lat_fp32},
        "metrics":  {"fp32": {"val": m_val_fp32, "test": m_test_fp32}},
    }

    if quantize:
        gc.collect(); ram_before = ram_mb()
        sess_int8 = make_session(onnx_int8)
        ram_int8  = round(ram_mb() - ram_before, 1)
        lat_int8  = bench_latency(sess_int8)
        m_val_int8  = eval_session(sess_int8, x_val,  y_val)
        m_test_int8 = eval_session(sess_int8, x_test, y_test)

        result["ram_load_mb"]["int8"] = ram_int8
        result["latency"]["int8"]     = lat_int8
        result["metrics"]["int8"]     = {"val": m_val_int8, "test": m_test_int8}

    print(f"    n_params={n_params:,}  fp32_kb={fp32_kb}  "
          f"lat_fp32={lat_fp32['mean_ms']:.3f}ms  "
          f"test_auc={m_test_fp32['auc_roc']:.4f}")
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    np.random.seed(42)
    print("=" * 60)
    print("Fase 5 - Validacion completa de modelos")
    print("=" * 60)

    x_val  = np.load(PROCESSED_DIR / "X_val.npy")
    y_val  = np.load(PROCESSED_DIR / "y_apnea_val.npy")
    x_test = np.load(PROCESSED_DIR / "X_test.npy")
    y_test = np.load(PROCESSED_DIR / "y_apnea_test.npy")

    all_results = {}

    # ---- Baseline ----
    baseline = SleepApneaCNN(num_classes=2)
    baseline.load_state_dict(torch.load(CKPT_DIR / "sleep_baseline.pth",
                                        map_location="cpu"))
    baseline.eval()
    all_results["baseline"] = evaluate_model(
        "baseline",
        pth_path=CKPT_DIR / "sleep_baseline.pth",
        model_obj=baseline,
        onnx_fp32=CKPT_DIR / "baseline_fp32.onnx",
        onnx_int8=CKPT_DIR / "baseline_int8.onnx",
        x_val=x_val, y_val=y_val, x_test=x_test, y_test=y_test,
    )

    # ---- Variantes podadas ----
    PRUNED_VARIANTS = ["p050_kd", "p050_nokd", "p070_kd", "p070_nokd"]
    for variant in PRUNED_VARIANTS:
        arch_path = CKPT_DIR / variant / "arch_config.json"
        with open(arch_path) as f:
            arch = json.load(f)
        model = PrunedSleepApneaCNN(arch)
        model.load_state_dict(torch.load(CKPT_DIR / variant / "model.pth",
                                         map_location="cpu"))
        model.eval()
        all_results[variant] = evaluate_model(
            variant,
            pth_path=CKPT_DIR / variant / "model.pth",
            model_obj=model,
            onnx_fp32=CKPT_DIR / variant / "model_fp32.onnx",
            onnx_int8=CKPT_DIR / variant / "model_int8.onnx",
            x_val=x_val, y_val=y_val, x_test=x_test, y_test=y_test,
        )

    # ---- Resumen ----
    print(f"\n{'='*70}")
    print("RESUMEN TEST SET")
    print(f"{'='*70}")
    hdr = f"{'Model':<16} {'Params':>8} {'FP32 KB':>8} {'INT8 KB':>8} " \
          f"{'Lat FP32':>10} {'Lat INT8':>10} {'AUC FP32':>10} {'AUC INT8':>10}"
    print(hdr)
    print("-" * len(hdr))
    for name, r in all_results.items():
        lat_fp = r["latency"]["fp32"]["mean_ms"]
        lat_i8 = r["latency"].get("int8", {}).get("mean_ms", float("nan"))
        auc_fp = r["metrics"]["fp32"]["test"]["auc_roc"]
        auc_i8 = r["metrics"].get("int8", {}).get("test", {}).get("auc_roc", float("nan"))
        print(f"{name:<16} {r['n_params']:>8,} "
              f"{r['disk_kb']['fp32']:>8.1f} "
              f"{str(r['disk_kb']['int8'] or '-'):>8}  "
              f"{lat_fp:>9.3f}ms {lat_i8:>9.3f}ms "
              f"{auc_fp:>10.4f} {auc_i8:>10.4f}")

    out = RESULTS_DIR / "metrics_eval.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nGuardado en {out}")
    print("Fase 5 completada.")


if __name__ == "__main__":
    main()
