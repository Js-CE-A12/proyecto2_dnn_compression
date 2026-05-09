# Proyecto 2 — Compresión de Modelos DNN para Detección de Apnea del Sueño

Universidad Interamericana PR – Bayamón | Curso de Ciencia de Datos

## Equipo
- Integrante 1:
- Integrante 2:
- Integrante 3:

## Descripción

Pipeline completo de compresión de redes neuronales profundas (**Pruning → Knowledge Distillation → Cuantización INT8**) para detección binaria de apnea del sueño con EEG monocanal.

- **Entrenamiento**: UCDDB (25 sujetos, canal C3-A2, 100 Hz) — validación cruzada 5-fold por sujeto
- **Evaluación cross-dataset**: ISRUC-Sleep (22 sujetos) — nunca visto durante entrenamiento
- **Hallazgo principal**: los modelos comprimidos generalizan mejor a ISRUC que el baseline completo (AUC E5 = 0.6308 vs baseline 0.6130)

## Instalación

```bash
pip install -r requirements.txt
```

## Orden de ejecución

```bash
# 1. Preprocesar UCDDB (requiere EDF en data/raw/)
python scripts/01_prepare_data.py

# 2. Preprocesar ISRUC (requiere carpetas subj_* en data/raw/isruc/)
python scripts/03_isruc_preprocess.py

# 3. Validación cruzada 5-fold baseline (UCDDB)
python scripts/02b_crossval.py

# 4. Pruning + Knowledge Distillation (train UCDDB, test ISRUC)
python scripts/03_pruning_kd.py

# 5. Cuantización INT8
python scripts/04_quantize.py

# 6. Evaluación completa (cross-dataset ISRUC)
python scripts/05_evaluate.py

# 7. Ablación E0–E5
python scripts/06_ablation.py

# 8. Gráficas Pareto
python scripts/07_pareto_plot.py

# 9. Generar informe
python scripts/generate_report_docx.py
```

## Estructura de carpetas

```
proyecto2_dnn_compression/
├── data/
│   ├── raw/              — EDF originales (UCDDB) y carpetas ISRUC
│   └── processed/        — NPY/NPZ generados por 01 y 03_isruc_preprocess
├── checkpoints/          — Modelos entrenados (.pth, .onnx)
│   ├── sleep_baseline.pth
│   ├── sleep_baseline_full.pth
│   ├── baseline_fp32.onnx / baseline_int8.onnx
│   ├── p050_kd/          — model.pth, model_fp32.onnx, model_int8.onnx, arch_config.json
│   ├── p050_nokd/
│   ├── p070_kd/
│   └── p070_nokd/
├── results/              — JSON de métricas y figuras
│   ├── crossval_metrics.json
│   ├── metrics_eval.json
│   ├── ablation_metrics.json
│   └── figures/          — pareto.png, ablation_table.png
├── src/                  — Módulos Python (models, pruning, distillation)
├── scripts/              — Scripts de pipeline (01–07 + generate_report_docx)
├── informe_final.docx    — Reporte generado automáticamente
└── requirements.txt
```

## Reproducibilidad
- Semilla fija: `seed = 42` en todos los scripts
- Partición por sujeto (no por época) para evitar data leakage
- Evaluación cross-dataset: ISRUC nunca incluido en entrenamiento
