"""
pruning.py — Poda estructurada de filtros Conv1d por magnitud L1.

Funciones públicas:
  compute_prune_masks   → dict {conv1/2/3: [índices a conservar]}
  build_pruned_model    → (PrunedSleepApneaCNN, arch_config)

La clase PrunedSleepApneaCNN tiene la misma estructura secuencial que
SleepApneaCNN pero con out_channels reducidos; los índices de capas son
idénticos para poder reutilizar el código de transferencia de pesos.

Índices en features Sequential:
  [0]  Conv1d(1 → c1, k=50)   [1]  BN(c1)
  [5]  Conv1d(c1 → c2, k=8)   [6]  BN(c2)
  [10] Conv1d(c2 → c3, k=8)   [11] BN(c3)
Índices en classifier Sequential:
  [1]  Linear(c3*23 → 64)     [4]  Linear(64 → 2)
"""

from typing import Dict, List, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Arquitectura podada
# ---------------------------------------------------------------------------

class PrunedSleepApneaCNN(nn.Module):
    """
    SleepApneaCNN con número de canales reducido según arch_config.
    Input: (B, 1, 3000) float32  →  logits (B, 2).

    Tamaños temporales (idénticos al baseline, solo cambian los canales):
      MaxPool(8) → 375 | MaxPool(4) → 93 | MaxPool(4) → 23
      Flatten: c3 × 23
    """

    def __init__(self, arch_config: dict, dropout: float = 0.5) -> None:
        super().__init__()
        c1 = arch_config["conv1_out"]
        c2 = arch_config["conv2_out"]
        c3 = arch_config["conv3_out"]
        self.features = nn.Sequential(
            nn.Conv1d(1, c1, kernel_size=50, padding="same"),   # [0]
            nn.BatchNorm1d(c1),                                   # [1]
            nn.ReLU(inplace=True),                                # [2]
            nn.MaxPool1d(kernel_size=8, stride=8),                # [3]
            nn.Dropout(dropout),                                  # [4]
            nn.Conv1d(c1, c2, kernel_size=8, padding="same"),   # [5]
            nn.BatchNorm1d(c2),                                   # [6]
            nn.ReLU(inplace=True),                                # [7]
            nn.MaxPool1d(kernel_size=4, stride=4),                # [8]
            nn.Dropout(dropout),                                  # [9]
            nn.Conv1d(c2, c3, kernel_size=8, padding="same"),   # [10]
            nn.BatchNorm1d(c3),                                   # [11]
            nn.ReLU(inplace=True),                                # [12]
            nn.MaxPool1d(kernel_size=4, stride=4),                # [13]
            nn.Dropout(dropout),                                  # [14]
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                         # [0]
            nn.Linear(c3 * 23, 64),              # [1]
            nn.ReLU(inplace=True),                # [2]
            nn.Dropout(dropout),                  # [3]
            nn.Linear(64, 2),                     # [4]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


# ---------------------------------------------------------------------------
# Cálculo de máscaras
# ---------------------------------------------------------------------------

def compute_prune_masks(model: nn.Module,
                        prune_ratio: float) -> Dict[str, List[int]]:
    """
    Calcula los índices de filtros a CONSERVAR en cada Conv1d, eligiendo los
    (1 − prune_ratio) × 100 % con mayor norma L1.

    Asume estructura SleepApneaCNN: conv1=features[0], conv2=[5], conv3=[10].
    Siempre conserva al menos 1 filtro por capa.
    """
    conv_layers = {
        "conv1": model.features[0],
        "conv2": model.features[5],
        "conv3": model.features[10],
    }
    masks: Dict[str, List[int]] = {}
    for name, conv in conv_layers.items():
        # weight: (out_channels, in_channels, kernel_size)
        l1 = conv.weight.detach().cpu().abs().sum(dim=(1, 2))
        n_keep = max(1, round(l1.shape[0] * (1.0 - prune_ratio)))
        keep = torch.argsort(l1, descending=True)[:n_keep]
        masks[name] = sorted(keep.tolist())
    return masks


# ---------------------------------------------------------------------------
# Transferencia de pesos
# ---------------------------------------------------------------------------

def _copy_bn(dst: nn.BatchNorm1d, src: nn.BatchNorm1d,
             idx: List[int]) -> None:
    """Copia parámetros BN (trainables + running stats) para canales idx."""
    def cpu(t: torch.Tensor) -> torch.Tensor:
        return t.detach().cpu()
    dst.weight.copy_(cpu(src.weight)[idx])
    dst.bias.copy_(cpu(src.bias)[idx])
    dst.running_mean.copy_(cpu(src.running_mean)[idx])
    dst.running_var.copy_(cpu(src.running_var)[idx])


def build_pruned_model(
    teacher: nn.Module,
    masks: Dict[str, List[int]],
) -> Tuple[PrunedSleepApneaCNN, dict]:
    """
    Construye PrunedSleepApneaCNN y transfiere los pesos relevantes del teacher.

    Siempre opera en CPU; llamar a .to(device) sobre el resultado antes de
    entrenar. Funciona aunque el teacher esté en cualquier dispositivo.

    Returns
    -------
    student    : PrunedSleepApneaCNN (en CPU, pesos inicializados)
    arch_config: dict con conv{1,2,3}_out, keep_indices
    """
    k1, k2, k3 = masks["conv1"], masks["conv2"], masks["conv3"]
    arch_config = {
        "conv1_out":    len(k1),
        "conv2_out":    len(k2),
        "conv3_out":    len(k3),
        "keep_indices": {n: v for n, v in masks.items()},
    }
    student = PrunedSleepApneaCNN(arch_config)

    def w(t: torch.Tensor) -> torch.Tensor:
        return t.detach().cpu()

    with torch.no_grad():
        # ---- Block 1: Conv(1→c1) ----
        student.features[0].weight.copy_(w(teacher.features[0].weight)[k1])
        student.features[0].bias.copy_(  w(teacher.features[0].bias)[k1])
        _copy_bn(student.features[1], teacher.features[1], k1)

        # ---- Block 2: Conv(c1→c2) ----
        # weight[k2] → (c2, 32, 8); [:, k1, :] → (c2, c1, 8)
        student.features[5].weight.copy_(
            w(teacher.features[5].weight)[k2][:, k1, :])
        student.features[5].bias.copy_(w(teacher.features[5].bias)[k2])
        _copy_bn(student.features[6], teacher.features[6], k2)

        # ---- Block 3: Conv(c2→c3) ----
        student.features[10].weight.copy_(
            w(teacher.features[10].weight)[k3][:, k2, :])
        student.features[10].bias.copy_(w(teacher.features[10].bias)[k3])
        _copy_bn(student.features[11], teacher.features[11], k3)

        # ---- Flatten → Linear(c3*23 → 64) ----
        # En el baseline flatten produce (128*23=2944,); canal ci ocupa
        # posiciones [ci*23 … ci*23+22]. Seleccionamos columnas para k3.
        flat_idx: List[int] = []
        for ci in k3:
            flat_idx.extend(range(ci * 23, ci * 23 + 23))
        student.classifier[1].weight.copy_(
            w(teacher.classifier[1].weight)[:, flat_idx])
        student.classifier[1].bias.copy_(w(teacher.classifier[1].bias))

        # ---- Linear(64 → 2) — sin cambio ----
        student.classifier[4].weight.copy_(w(teacher.classifier[4].weight))
        student.classifier[4].bias.copy_(  w(teacher.classifier[4].bias))

    return student, arch_config
