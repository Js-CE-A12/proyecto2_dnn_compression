Generado por: scripts/03_pruning_kd.py
Archivos esperados:
  model.pth          — pesos podados + KD fine-tuned
  model_full.pth     — arquitectura completa + pesos
  pruning_masks.pkl  — filtros sobrevivientes (CRITICO para cuantizacion)
  arch_config.json   — dimensiones de capas tras pruning
  metrics.json       — accuracy, F1, kappa en test set
