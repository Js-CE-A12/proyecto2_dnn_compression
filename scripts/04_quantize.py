"""
Fase 4 — Cuantización INT8
- Carga modelo podado + KD (p050_kd/model.pth)
- PyTorch → ONNX → TFLite INT8
- Calibración con 100-200 muestras del val set
- Verifica caída de accuracy < 1% (si no, aplica QAT)
Salida: checkpoints/model_int8.tflite
"""
# TODO: Implementar
