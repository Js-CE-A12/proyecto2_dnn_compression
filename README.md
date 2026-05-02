# Proyecto 2 — Compresión de Modelos DNN

Universidad Interamericana PR – Bayamón | Curso de Ciencia de Datos

## Equipo
- Integrante 1:
- Integrante 2:
- Integrante 3:

## Descripción
Pipeline de compresión de redes neuronales profundas (Pruning → KD → Cuantización INT8)
para clasificación de estados de sueño y detección de Apnea usando señales EEG.

## Instalación
```bash
pip install -r requirements.txt
```

## Orden de ejecución
```bash
python scripts/01_prepare_data.py
python scripts/02_train_baseline.py
python scripts/03_pruning_kd.py
python scripts/04_quantize.py
python scripts/05_evaluate.py
python scripts/06_ablation.py
python scripts/07_pareto_plot.py
```

## Reproducibilidad
- Semilla fija: seed = 42 en todos los scripts
- Partición por sujeto (no por época)
