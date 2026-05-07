"""
Fase 4 - Cuantizacion INT8
Pipeline: PyTorch (FP32) -> ONNX (FP32) -> ONNX INT8 (PTQ con calibracion)
Nota: TFLite requiere TensorFlow que no soporta Windows >= 2.11;
      se usa onnxruntime INT8 como equivalente funcional.

Modelo de entrada : checkpoints/p050_kd/model.pth  (PrunedSleepApneaCNN)
Salidas:
  checkpoints/model_fp32.onnx
  checkpoints/model_int8.onnx
  results/metrics_quantization.json
"""

import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,
)
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models import SleepApneaCNN
from src.pruning import PrunedSleepApneaCNN

PROCESSED_DIR = Path("data/processed")
CKPT_DIR      = Path("checkpoints")
RESULTS_DIR   = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

VARIANT       = "p050_kd"
N_CALIB       = 200      # muestras de calibracion del val set
WARMUP_RUNS   = 10
BENCH_RUNS    = 100


# ---------------------------------------------------------------------------
# Carga del modelo PyTorch
# ---------------------------------------------------------------------------

def load_pruned_model(variant: str) -> torch.nn.Module:
    arch_path = CKPT_DIR / variant / "arch_config.json"
    with open(arch_path) as f:
        arch_config = json.load(f)
    model = PrunedSleepApneaCNN(arch_config)
    state = torch.load(CKPT_DIR / variant / "model.pth", map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Exportacion a ONNX
# ---------------------------------------------------------------------------

def export_onnx(model: torch.nn.Module, path: Path) -> None:
    dummy = torch.randn(1, 1, 3000)
    # Use dynamo=False to avoid Unicode emoji output that breaks Windows cp1252
    torch.onnx.export(
        model, dummy, str(path),
        input_names=["eeg"],
        output_names=["logits"],
        dynamic_axes={"eeg": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"  ONNX FP32 exportado: {path}  ({path.stat().st_size / 1024:.1f} KB)")


# ---------------------------------------------------------------------------
# Calibrador para onnxruntime PTQ
# ---------------------------------------------------------------------------

class EEGCalibrationReader(CalibrationDataReader):
    def __init__(self, x_val: np.ndarray, n: int) -> None:
        idx = np.random.choice(len(x_val), min(n, len(x_val)), replace=False)
        self._data = [{"eeg": x_val[i:i+1].astype(np.float32)} for i in idx]
        self._pos  = 0

    def get_next(self):
        if self._pos >= len(self._data):
            return None
        out = self._data[self._pos]
        self._pos += 1
        return out

    def rewind(self):
        self._pos = 0


# ---------------------------------------------------------------------------
# Cuantizacion INT8 con onnxruntime PTQ
# ---------------------------------------------------------------------------

def quantize_int8(fp32_path: Path, int8_path: Path,
                  x_val: np.ndarray) -> None:
    calib = EEGCalibrationReader(x_val, N_CALIB)
    quantize_static(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        calibration_data_reader=calib,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=False,
    )
    print(f"  ONNX INT8  exportado: {int8_path}  ({int8_path.stat().st_size / 1024:.1f} KB)")


# ---------------------------------------------------------------------------
# Inferencia con ONNX Runtime
# ---------------------------------------------------------------------------

def run_onnx(session: ort.InferenceSession,
             x: np.ndarray, batch: int = 64) -> tuple:
    all_probs, all_preds = [], []
    for i in range(0, len(x), batch):
        xb = x[i:i+batch].astype(np.float32)
        out = session.run(None, {"eeg": xb})[0]   # (B, 2)
        probs = np.exp(out) / np.exp(out).sum(axis=1, keepdims=True)
        all_probs.append(probs[:, 1])
        all_preds.append(out.argmax(axis=1))
    return np.concatenate(all_probs), np.concatenate(all_preds)


def eval_onnx(session: ort.InferenceSession,
              x: np.ndarray, y: np.ndarray) -> dict:
    probs, preds = run_onnx(session, x)
    return {
        "accuracy": round(float(accuracy_score(y, preds)), 4),
        "f1":       round(float(f1_score(y, preds, zero_division=0)), 4),
        "auc_roc":  round(float(roc_auc_score(y, probs)), 4),
    }


def benchmark_latency(session: ort.InferenceSession,
                      warmup: int = WARMUP_RUNS,
                      runs: int = BENCH_RUNS) -> float:
    dummy = np.random.randn(1, 1, 3000).astype(np.float32)
    for _ in range(warmup):
        session.run(None, {"eeg": dummy})
    t0 = time.perf_counter()
    for _ in range(runs):
        session.run(None, {"eeg": dummy})
    return (time.perf_counter() - t0) / runs * 1000   # ms per sample


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    np.random.seed(42)
    print("=" * 60)
    print("Fase 4 - Cuantizacion INT8 (ONNX Runtime PTQ)")
    print("=" * 60)

    # Datos val y test
    x_val  = np.load(PROCESSED_DIR / "X_val.npy")
    y_val  = np.load(PROCESSED_DIR / "y_apnea_val.npy")
    x_test = np.load(PROCESSED_DIR / "X_test.npy")
    y_test = np.load(PROCESSED_DIR / "y_apnea_test.npy")
    print(f"Val: {len(x_val)} | Test: {len(x_test)}")

    # Cargar modelo podado
    model = load_pruned_model(VARIANT)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Modelo: {VARIANT}  ({n_params:,} parametros)")

    # Rutas de salida
    fp32_path = CKPT_DIR / "model_fp32.onnx"
    int8_path = CKPT_DIR / "model_int8.onnx"

    # Exportar FP32
    print("\n[1] Exportando a ONNX FP32...")
    export_onnx(model, fp32_path)

    # Cuantizar a INT8
    print("\n[2] Cuantizando a INT8 con PTQ...")
    quantize_int8(fp32_path, int8_path, x_val)

    # Sesiones ONNX Runtime
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    sess_fp32 = ort.InferenceSession(str(fp32_path), sess_options=opts,
                                     providers=["CPUExecutionProvider"])
    sess_int8 = ort.InferenceSession(str(int8_path), sess_options=opts,
                                     providers=["CPUExecutionProvider"])

    # Metricas de accuracy
    print("\n[3] Evaluando accuracy...")
    m_fp32_val  = eval_onnx(sess_fp32, x_val,  y_val)
    m_int8_val  = eval_onnx(sess_int8, x_val,  y_val)
    m_fp32_test = eval_onnx(sess_fp32, x_test, y_test)
    m_int8_test = eval_onnx(sess_int8, x_test, y_test)

    auc_drop_val  = round(m_fp32_val["auc_roc"]  - m_int8_val["auc_roc"],  4)
    auc_drop_test = round(m_fp32_test["auc_roc"] - m_int8_test["auc_roc"], 4)

    print(f"  FP32 val  AUC={m_fp32_val['auc_roc']:.4f}  test AUC={m_fp32_test['auc_roc']:.4f}")
    print(f"  INT8 val  AUC={m_int8_val['auc_roc']:.4f}  test AUC={m_int8_test['auc_roc']:.4f}")
    print(f"  Caida AUC val={auc_drop_val:.4f}  test={auc_drop_test:.4f}")
    if abs(auc_drop_val) < 0.01:
        print("  -> Caida < 1% en val. QAT no requerido.")
    else:
        print("  -> Caida >= 1% en val. Considerar QAT.")

    # Latencia
    print("\n[4] Midiendo latencia (inferencia de 1 muestra, CPU)...")
    lat_fp32 = benchmark_latency(sess_fp32)
    lat_int8 = benchmark_latency(sess_int8)
    speedup  = round(lat_fp32 / lat_int8, 2)
    print(f"  FP32: {lat_fp32:.3f} ms  |  INT8: {lat_int8:.3f} ms  "
          f"|  Speedup: {speedup:.2f}x")

    # Tamanos de archivo
    size_fp32 = fp32_path.stat().st_size
    size_int8 = int8_path.stat().st_size
    size_ratio = round(size_fp32 / size_int8, 2)
    print(f"\n[5] Tamano en disco:")
    print(f"  FP32: {size_fp32 / 1024:.1f} KB  |  INT8: {size_int8 / 1024:.1f} KB  "
          f"|  Compresion: {size_ratio:.2f}x")

    # Guardar metricas
    metrics_out = {
        "model":    VARIANT,
        "n_params": n_params,
        "files": {
            "fp32_kb": round(size_fp32 / 1024, 1),
            "int8_kb": round(size_int8 / 1024, 1),
            "size_ratio": size_ratio,
        },
        "latency_ms": {
            "fp32":    round(lat_fp32, 3),
            "int8":    round(lat_int8, 3),
            "speedup": speedup,
        },
        "val": {
            "fp32": m_fp32_val,
            "int8": m_int8_val,
            "auc_drop": auc_drop_val,
        },
        "test": {
            "fp32": m_fp32_test,
            "int8": m_int8_test,
            "auc_drop": auc_drop_test,
        },
    }

    out_path = RESULTS_DIR / "metrics_quantization.json"
    with open(out_path, "w") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"\nMetricas guardadas en {out_path}")
    print("\nFase 4 completada.")


if __name__ == "__main__":
    main()
