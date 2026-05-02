"""
Fase 1 — Preparación del dataset
- Selección de sujetos y canal EEG
- Extracción de épocas (descartar 1ra y última hora en Sleep-EDF)
- Manejo del desbalance con weighted cross-entropy
- Partición train/val/test POR SUJETO (14/3/3 o LOSO)
Salidas: X_train.npy, y_train.npy, X_val.npy, y_val.npy,
         X_test.npy, y_test.npy, split_subjects.json
"""
# TODO: Implementar
