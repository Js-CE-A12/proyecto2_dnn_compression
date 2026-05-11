# Proyecto 2 — Compresión de Modelos DNN para Detección de Apnea del Sueño

Universidad Interamericana PR – Bayamón | Curso de Ciencia de Datos

## Equipo
- Eli Jaaziel Ayala Ortiz — Y00622025
- Edwin Román Maldonado — R00587066

## Descripción

Pipeline completo de compresión de redes neuronales profundas (**Pruning → Knowledge Distillation → Cuantización INT8**) para detección binaria de apnea del sueño con EEG monocanal.

- **Modelo base**: CNN 1D (272,802 parámetros) entrenada en UCDDB
- **Entrenamiento**: UCDDB (25 sujetos, canal C3-A2, 100 Hz) — validación cruzada 5-fold por sujeto
- **Evaluación cross-dataset**: ISRUC-Sleep (22 sujetos) — nunca visto durante entrenamiento
- **Hallazgo principal**: los modelos comprimidos generalizan mejor a ISRUC que el baseline completo (AUC E5 = 0.6308 vs baseline 0.6130, con 8.5× reducción en disco y 3.90× aceleración)

---

## Estructura del Proyecto

```
.
├── configs/                    # Hiperparámetros y configuración
│   ├── config_baseline.json
│   └── config_compression.json
├── data/
│   ├── raw/                    # Datos crudos EDF (no incluidos — ver descarga)
│   └── processed/              # Datos preprocesados .npy (generados por scripts)
├── checkpoints/                # Modelos guardados (generados por scripts)
│   ├── p050_kd/                # Pruning 50% + KD
│   ├── p050_nokd/              # Pruning 50% sin KD
│   ├── p070_kd/                # Pruning 70% + KD
│   └── p070_nokd/              # Pruning 70% sin KD
├── results/
│   └── figures/                # Gráficas generadas (pareto.png, ablation_table.png)
├── scripts/                    # Scripts de reproducción (correr en orden)
│   ├── 01_prepare_data.py      # Fase 1: preprocesar UCDDB
│   ├── 02b_crossval.py         # Fase 2: entrenar CNN baseline (5-fold CV)
│   ├── 03_isruc_preprocess.py  # Fase 3a: preprocesar ISRUC
│   ├── 03_pruning_kd.py        # Fase 3b: Pruning + KD
│   ├── 04_quantize.py          # Fase 4: Cuantización INT8
│   ├── 05_evaluate.py          # Fase 5: Evaluación completa
│   ├── 06_ablation.py          # Fase 6: Tabla de ablación E0-E5
│   └── 07_pareto_plot.py       # Fase 7: Curva de Pareto
├── src/                        # Módulos Python reutilizables
│   ├── models.py               # Arquitectura CNN
│   ├── pruning.py              # Poda estructurada L1-norm
│   ├── distillation.py         # Knowledge Distillation loss
│   ├── quantization.py         # Cuantización INT8 ONNX Runtime
│   ├── dataset.py              # Dataset loader
│   ├── metrics.py              # Métricas de evaluación
│   └── utils.py                # Utilidades generales
├── requirements.txt
└── README.md
```

---

## Instalación

```bash
pip install -r requirements.txt
```

> **Nota**: Se requiere Python 3.10+. El entrenamiento se realizó en CPU (no se requiere GPU).

---

## Descarga de Datos

### Dataset UCDDB (entrenamiento)
1. Crear cuenta gratuita en [PhysioNet](https://physionet.org/)
2. Ir a: https://physionet.org/content/ucddb/1.0.0/
3. Descargar el ZIP completo
4. Descomprimir y colocar los archivos `.rec`, `_stage.txt`, `_respevt.txt` en:
   ```
   data/raw/st-vincents-university-hospital-university-college-dublin-sleep-apnea-database-1.0.0/files/
   ```
   Debe haber archivos como: `ucddb002.rec`, `ucddb002_stage.txt`, `ucddb002_respevt.txt`, etc.

### Dataset ISRUC-Sleep (evaluación cross-dataset)
1. Ir a: https://sleeptight.isr.uc.pt/ISRUC_Sleep/
2. Descargar los sujetos del subgrupo 1 (sujetos 1–30)
3. Colocar en:
   ```
   data/raw/isruc/
   ```
   Estructura esperada: `data/raw/isruc/subj1/`, `data/raw/isruc/subj2/`, etc.

---

## Orden de Ejecución

Ejecutar los scripts en el siguiente orden desde la raíz del proyecto:

```bash
# Fase 1 — Preprocesar UCDDB
python scripts/01_prepare_data.py

# Fase 2 — Entrenar CNN baseline con validación cruzada 5-fold
python scripts/02b_crossval.py

# Fase 3a — Preprocesar ISRUC (dataset de evaluación cross-dataset)
python scripts/03_isruc_preprocess.py

# Fase 3b — Pruning + Knowledge Distillation
python scripts/03_pruning_kd.py

# Fase 4 — Cuantización INT8 (ONNX Runtime QDQ)
python scripts/04_quantize.py

# Fase 5 — Evaluación completa (latencia, RAM, AUC-ROC en ISRUC)
python scripts/05_evaluate.py

# Fase 6 — Tabla de ablación E0-E5
python scripts/06_ablation.py

# Fase 7 — Curva de Pareto
python scripts/07_pareto_plot.py
```

---

## Resultados Esperados

Después de correr todos los scripts, los resultados estarán en `results/`:

| Archivo | Descripción |
|---------|-------------|
| `results/figures/pareto.png` | Curva de Pareto AUC vs Latencia |
| `results/figures/ablation_table.png` | Tabla de ablación E0-E5 |
| `checkpoints/p050_kd/metrics.json` | Métricas modelo comprimido óptimo |

**Resultado principal**: E5 (Pruning 50% + KD + INT8) logra AUC = 0.6308 en ISRUC con 8.5× reducción en disco y 3.90× aceleración vs baseline.

---

## Configuración Experimental

| Componente | Especificación |
|-----------|---------------|
| CPU | 12th Gen Intel Core i5-12450H |
| RAM | 7.68 GB |
| OS | Windows 11 |
| Python | 3.13 |
| PyTorch | 2.11.0+cu126 |
| ONNX Runtime | 1.25.1 |
| MNE | 1.11.0 |
| scikit-learn | 1.8.0 |
| Semilla aleatoria | 42 |

---

## Uso de IA

Este proyecto utilizó **Claude Code (Anthropic)** para generación y depuración de scripts Python. Todo el código fue revisado y ejecutado por el equipo.
