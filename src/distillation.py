"""
distillation.py — Knowledge Distillation loss (Hinton et al., 2015).

loss = α · CE(student_logits, labels)
     + (1−α) · T² · KL(teacher_soft ∥ student_soft)

donde:
  teacher_soft = softmax(teacher_logits / T)
  student_soft = softmax(student_logits / T)
  T² compensa la reducción de gradientes al elevar la temperatura.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class KDLoss(nn.Module):
    """
    Loss combinada CE + KL-Divergencia para Knowledge Distillation.

    Parameters
    ----------
    temperature : float
        Temperatura T para suavizar distribuciones (por defecto 4).
    alpha : float
        Peso del término CE; (1−alpha) pesa el término KD (por defecto 0.5).
    """

    def __init__(self, temperature: float = 4.0, alpha: float = 0.5) -> None:
        super().__init__()
        self.T     = temperature
        self.alpha = alpha

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        ce_weight: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        student_logits : (B, C) — logits del modelo podado
        teacher_logits : (B, C) — logits del teacher (sin gradiente)
        labels         : (B,)  — etiquetas duras (long)
        ce_weight      : (C,) opcional — pesos de clase para CE
        """
        # Término 1: Cross-Entropy dura
        ce = F.cross_entropy(student_logits, labels, weight=ce_weight)

        # Término 2: KL(teacher_soft || student_soft) × T²
        log_student = F.log_softmax(student_logits / self.T, dim=1)
        soft_teacher = F.softmax(teacher_logits / self.T, dim=1).detach()
        kd = F.kl_div(log_student, soft_teacher,
                      reduction="batchmean") * (self.T ** 2)

        return self.alpha * ce + (1.0 - self.alpha) * kd
