# Guía de Ejecución — Pipeline de Compresión DNN

Ejecutar todos los comandos desde la carpeta raíz del proyecto:
```
cd proyecto2_dnn_compression
```

---

## Requisitos previos

```bash
pip install -r requirements.txt
```

**Datos necesarios antes de empezar:**
- `data/raw/` — archivos EDF de UCDDB (28 sujetos, descargar de PhysioNet)
- `data/raw/isruc/subj_N/` — carpetas de cada sujeto ISRUC con archivos `.rec` y `.xlsx`

---

## Paso 1 — Preprocesar UCDDB

**Script:** `scripts/01_prepare_data.py`  
**Entrada:** `data/raw/` (EDF originales de UCDDB)  
**Salida:** `data/processed/X_train.npy`, `X_val.npy`, `X_test.npy`, `y_apnea_*.npy`, `class_weights_apnea.json`

```bash
python scripts/01_prepare_data.py
```

Segmenta las señales EEG (canal C3-A2) en épocas de 30 segundos a 100 Hz, aplica z-score por época y genera las particiones train/val/test por sujeto.

---

## Paso 2 — Preprocesar ISRUC

**Script:** `scripts/03_isruc_preprocess.py`  
**Entrada:** `data/raw/isruc/subj_N/` (archivos `.rec` y `.xlsx` por sujeto)  
**Salida:** `data/processed/isruc_subjects/subj_N.npz`

```bash
python scripts/03_isruc_preprocess.py
```

Preprocesa los 22 sujetos ISRUC con tasa de apnea ≥ 2%. Re-muestrea de 200 Hz a 100 Hz, lee etiquetas de apnea desde los archivos `.xlsx` (eventos OA, CA, OH, MA, MH). Estos datos se usan **solo como test**, nunca en entrenamiento.

---

## Paso 3 — Validación cruzada baseline (UCDDB)

**Script:** `scripts/02b_crossval.py`  
**Entrada:** `data/processed/` (UCDDB)  
**Salida:** `checkpoints/crossval_best.pth`, `checkpoints/sleep_baseline.pth`, `results/crossval_metrics.json`

```bash
python scripts/02b_crossval.py
```

Entrena la CNN baseline con validación cruzada 5-fold por sujeto sobre UCDDB (25 sujetos). Cada fold rota 5 sujetos como test y 3 como val. Al terminar guarda el mejor modelo como `sleep_baseline.pth`.  
**Tiempo estimado:** 30–90 min según CPU.

---

## Paso 4 — Pruning + Knowledge Distillation

**Script:** `scripts/03_pruning_kd.py`  
**Entrada:** `checkpoints/sleep_baseline.pth`, `data/processed/`  
**Salida:** `checkpoints/p050_kd/`, `checkpoints/p050_nokd/`, `checkpoints/p070_kd/`, `checkpoints/p070_nokd/`

```bash
python scripts/03_pruning_kd.py
```

Genera 4 variantes podadas mediante L1-norm estructurada (50% y 70% de filtros eliminados), con y sin Knowledge Distillation (T=4, α=0.5). Fine-tuning con Adam lr=1e-4, 50 épocas, early stopping paciencia=20.  
El test de cada variante se evalúa sobre ISRUC (cross-dataset).  
**Tiempo estimado:** 20–60 min.

Archivos generados por variante:
| Archivo | Descripción |
|---|---|
| `model.pth` | Mejor checkpoint (mayor AUC val) |
| `model_full.pth` | Checkpoint final (última época) |
| `arch_config.json` | Arquitectura comprimida y ratio de compresión |
| `pruning_masks.pkl` | Máscaras de poda usadas |
| `metrics.json` | Métricas de entrenamiento e historial por época |

---

## Paso 5 — Cuantización INT8

**Script:** `scripts/04_quantize.py`  
**Entrada:** `checkpoints/` (modelos `.pth`)  
**Salida:** `checkpoints/*/model_fp32.onnx`, `checkpoints/*/model_int8.onnx`

```bash
python scripts/04_quantize.py
```

Exporta cada modelo a ONNX FP32 (opset 17) y aplica cuantización estática QDQ INT8 con 200 muestras de calibración del val set. Usa ONNX Runtime (no TFLite, incompatible con Windows 11 ≥ 2.11).

---

## Paso 6 — Evaluación completa (cross-dataset ISRUC)

**Script:** `scripts/05_evaluate.py`  
**Entrada:** `checkpoints/`, `data/processed/isruc_subjects/`  
**Salida:** `results/metrics_eval.json`

```bash
python scripts/05_evaluate.py
```

Evalúa todos los modelos (baseline + 4 variantes, FP32 e INT8) sobre el test set completo de ISRUC. Mide:
- **Latencia ONNX RT**: 50 warm-up + 500 inferencias → media, std, P95
- **RAM**: uso en MB al cargar el modelo
- **Disco**: tamaño en KB de `.pth` y `.onnx`
- **Métricas**: accuracy, F1, AUC-ROC en val (UCDDB) y test (ISRUC)

---

## Paso 7 — Estudio de ablación E0–E5

**Script:** `scripts/06_ablation.py`  
**Entrada:** `results/metrics_eval.json`  
**Salida:** `results/ablation_metrics.json`

```bash
python scripts/06_ablation.py
```

Construye la tabla comparativa E0–E5 aislando la contribución de cada técnica:

| ID | Configuración |
|---|---|
| E0 | Baseline FP32 |
| E1 | Pruning 50% sin KD (FP32) |
| E2 | Solo cuantización INT8 (baseline) |
| E3 | Pruning 50% sin KD + INT8 |
| E4 | Pruning 50% + KD (FP32) |
| E5 | Pruning 50% + KD + INT8 ← **óptimo** |

---

## Paso 8 — Curva de Pareto

**Script:** `scripts/07_pareto_plot.py`  
**Entrada:** `results/ablation_metrics.json`  
**Salida:** `results/figures/pareto.png`, `results/figures/ablation_table.png`

```bash
python scripts/07_pareto_plot.py
```

Genera la gráfica AUC-ROC vs. latencia con la frontera de Pareto. E5 se ubica sobre la frontera dominando a todos los demás modelos en la combinación AUC + velocidad.

---

## Paso 9 — Generar informe

**Script:** `scripts/generate_report_docx.py`  
**Entrada:** `results/*.json`, `results/figures/*.png`  
**Salida:** `informe_final.docx`

```bash
python scripts/generate_report_docx.py
```

Genera el informe completo en formato Word con todos los resultados, tablas y figuras embebidas.

---

## Resumen rápido (todos los pasos)

```bash
python scripts/01_prepare_data.py
python scripts/03_isruc_preprocess.py
python scripts/02b_crossval.py
python scripts/03_pruning_kd.py
python scripts/04_quantize.py
python scripts/05_evaluate.py
python scripts/06_ablation.py
python scripts/07_pareto_plot.py
python scripts/generate_report_docx.py
```
