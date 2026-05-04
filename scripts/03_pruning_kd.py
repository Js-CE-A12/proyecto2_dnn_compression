"""
Fase 3 — Pruning estructurado + Knowledge Distillation (PQDistill pipeline).

Teacher : SleepApneaCNN baseline (pesos congelados, cargado de sleep_baseline.pth)
Student : PrunedSleepApneaCNN construido mediante poda L1 de filtros Conv1d

Variantes entrenadas:
  p050_kd   — 50 % de filtros eliminados, fine-tuning con KD
  p050_nokd — 50 % de filtros eliminados, fine-tuning sin KD
  p070_kd   — 70 % de filtros eliminados, fine-tuning con KD
  p070_nokd — 70 % de filtros eliminados, fine-tuning sin KD

Salidas por variante en checkpoints/<variante>/:
  model.pth, model_full.pth, pruning_masks.pkl, arch_config.json, metrics.json

Hiperparámetros:
  Adam lr=1e-4 | batch=32 | epochs=50 | early-stopping paciencia=20
  KD: T=4, α=0.5
"""

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, cohen_kappa_score, f1_score, roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.distillation import KDLoss
from src.models import SleepApneaCNN
from src.pruning import build_pruned_model, compute_prune_masks

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
PROCESSED_DIR = Path("data/processed")
CKPT_DIR      = Path("checkpoints")

SEED    = 42
LR      = 1e-4
BATCH   = 32
EPOCHS  = 50
PATIENCE = 20
T_KD    = 4.0
ALPHA   = 0.5

torch.manual_seed(SEED)
np.random.seed(SEED)

VARIANTS = [
    (0.50, "p050", True,  "p050_kd"),
    (0.50, "p050", False, "p050_nokd"),
    (0.70, "p070", True,  "p070_kd"),
    (0.70, "p070", False, "p070_nokd"),
]


# ---------------------------------------------------------------------------
# Dataset (mismo que Fase 2)
# ---------------------------------------------------------------------------

class ApneaDataset(Dataset):
    def __init__(self, split: str) -> None:
        self.X = np.load(PROCESSED_DIR / f"X_{split}.npy")
        self.y = np.load(PROCESSED_DIR / f"y_apnea_{split}.npy").astype(np.int64)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return torch.from_numpy(self.X[idx]), int(self.y[idx])


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_prob: np.ndarray) -> dict:
    acc   = accuracy_score(y_true, y_pred)
    f1    = f1_score(y_true, y_pred, zero_division=0)
    auc   = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
    kappa = cohen_kappa_score(y_true, y_pred)
    return {
        "accuracy": round(float(acc),   4),
        "f1":       round(float(f1),    4),
        "auc_roc":  round(float(auc),   4),
        "kappa":    round(float(kappa), 4),
    }


@torch.no_grad()
def eval_model(model: nn.Module, loader: DataLoader,
               device: torch.device) -> dict:
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    for X_batch, y_batch in loader:
        logits = model(X_batch.to(device))
        probs  = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds  = logits.argmax(dim=1).cpu().numpy()
        all_probs.extend(probs.tolist())
        all_preds.extend(preds.tolist())
        all_labels.extend(y_batch.tolist())
    return compute_metrics(np.array(all_labels), np.array(all_preds),
                           np.array(all_probs))


# ---------------------------------------------------------------------------
# Entrenamiento de una variante
# ---------------------------------------------------------------------------

def train_variant(
    student:      nn.Module,
    teacher:      nn.Module,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    test_loader:  DataLoader,
    ce_weights:   torch.Tensor,
    use_kd:       bool,
    device:       torch.device,
    out_dir:      Path,
) -> dict:
    """Entrena student y devuelve métricas de test del mejor checkpoint."""

    student = student.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.Adam(student.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=10,
    )
    kd_loss_fn = KDLoss(temperature=T_KD, alpha=ALPHA) if use_kd else None
    ce_loss_fn = nn.CrossEntropyLoss(weight=ce_weights)

    best_val_auc = -1.0
    no_improve   = 0
    history      = []

    print(f"\n  {'Epoch':>5} {'Loss':>9} {'ValAUC':>8} "
          f"{'ValF1':>7} {'Kappa':>7} {'LR':>9}")
    print("  " + "-" * 54)

    for epoch in range(1, EPOCHS + 1):
        student.train()
        total_loss = 0.0
        t0 = time.time()

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_t     = y_batch.long().to(device)

            s_logits = student(X_batch)
            if use_kd:
                with torch.no_grad():
                    t_logits = teacher(X_batch)
                loss = kd_loss_fn(s_logits, t_logits, y_t, ce_weight=ce_weights)
            else:
                loss = ce_loss_fn(s_logits, y_t)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y_t)

        avg_loss    = total_loss / len(train_loader.dataset)
        val_metrics = eval_model(student, val_loader, device)
        scheduler.step(val_metrics["auc_roc"])

        history.append({
            "epoch":      epoch,
            "train_loss": round(avg_loss, 4),
            **{f"val_{k}": v for k, v in val_metrics.items()},
        })

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"  {epoch:>5d} {avg_loss:>9.4f} {val_metrics['auc_roc']:>8.4f} "
              f"{val_metrics['f1']:>7.4f} {val_metrics['kappa']:>7.4f} "
              f"{lr_now:>9.2e}  ({time.time()-t0:.1f}s)")

        if val_metrics["auc_roc"] > best_val_auc:
            best_val_auc = val_metrics["auc_roc"]
            torch.save(student.state_dict(), out_dir / "model.pth")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  Early stopping en época {epoch}.")
                break

    # Guardar estado final
    torch.save(student.state_dict(), out_dir / "model_full.pth")

    # Evaluar en test con mejor checkpoint
    student.load_state_dict(
        torch.load(out_dir / "model.pth", map_location=device))
    test_metrics = eval_model(student, test_loader, device)

    metrics_out = {
        "best_val_auc_roc": round(best_val_auc, 4),
        "test": test_metrics,
        "history": history,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)

    return test_metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    device = torch.device(
        "mps"  if torch.backends.mps.is_available()  else
        "cuda" if torch.cuda.is_available() else
        "cpu"
    )
    print(f"Device: {device}")
    print(f"T={T_KD} | α={ALPHA} | lr={LR} | epochs={EPOCHS} | patience={PATIENCE}")

    # ---- Datos ----
    train_loader = DataLoader(ApneaDataset("train"), batch_size=BATCH,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(ApneaDataset("val"),   batch_size=BATCH,
                              shuffle=False, num_workers=0)
    test_loader  = DataLoader(ApneaDataset("test"),  batch_size=BATCH,
                              shuffle=False, num_workers=0)

    # ---- Pesos de clase ----
    with open(PROCESSED_DIR / "class_weights_apnea.json") as f:
        cw = json.load(f)
    ce_weights = torch.tensor([cw["0"], cw["1"]], dtype=torch.float32).to(device)

    # ---- Teacher ----
    teacher_path = CKPT_DIR / "sleep_baseline.pth"
    if not teacher_path.exists():
        teacher_path = CKPT_DIR / "sleep_baseline_full.pth"
    teacher = SleepApneaCNN(num_classes=2)
    teacher.load_state_dict(torch.load(teacher_path, map_location="cpu"))
    teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    print(f"Teacher cargado desde: {teacher_path}")

    # ---- Ejecutar variantes ----
    all_results: dict = {}
    cached_masks: dict = {}          # reutilizar masks dentro del mismo ratio

    for prune_ratio, ratio_tag, use_kd, variant in VARIANTS:
        out_dir = CKPT_DIR / variant
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*62}")
        print(f"Variante: {variant}  (prune={prune_ratio*100:.0f}%  "
              f"KD={'sí' if use_kd else 'no'})")
        print(f"{'='*62}")

        # Calcular máscaras solo una vez por ratio
        if ratio_tag not in cached_masks:
            cached_masks[ratio_tag] = compute_prune_masks(teacher, prune_ratio)
        masks = cached_masks[ratio_tag]

        print(f"  Filtros conservados → "
              f"conv1: {len(masks['conv1'])}/{teacher.features[0].weight.shape[0]}  "
              f"conv2: {len(masks['conv2'])}/{teacher.features[5].weight.shape[0]}  "
              f"conv3: {len(masks['conv3'])}/{teacher.features[10].weight.shape[0]}")

        # Construir student con pesos transferidos (siempre fresh para cada variante)
        student, arch_config = build_pruned_model(teacher, masks)
        n_params = sum(p.numel() for p in student.parameters())
        n_teacher = sum(p.numel() for p in teacher.parameters())
        compression = n_teacher / n_params
        print(f"  Parámetros student: {n_params:,}  "
              f"({n_params/n_teacher*100:.1f}% del teacher, "
              f"compresión ×{compression:.1f})")

        # Guardar máscaras y arch_config
        with open(out_dir / "pruning_masks.pkl", "wb") as f:
            pickle.dump(masks, f)
        arch_with_meta = {
            **arch_config,
            "prune_ratio":  prune_ratio,
            "use_kd":       use_kd,
            "n_params":     n_params,
            "n_params_teacher": n_teacher,
            "compression_ratio": round(compression, 2),
            "T":     T_KD   if use_kd else None,
            "alpha": ALPHA  if use_kd else None,
        }
        with open(out_dir / "arch_config.json", "w") as f:
            json.dump(arch_with_meta, f, indent=2)

        # Entrenar
        test_metrics = train_variant(
            student, teacher, train_loader, val_loader, test_loader,
            ce_weights, use_kd, device, out_dir,
        )
        all_results[variant] = {**test_metrics,
                                 "n_params": n_params,
                                 "compression": round(compression, 2)}

    # ---- Resumen ----
    print(f"\n{'='*62}")
    print("RESUMEN — MÉTRICAS EN TEST SET")
    print(f"{'='*62}")
    header = f"{'Variante':<14} {'Params':>8} {'Comp':>6} "  \
             f"{'ACC':>7} {'AUC-ROC':>9} {'F1':>7} {'Kappa':>8}"
    print(header)
    print("-" * len(header))
    for variant, m in all_results.items():
        print(f"{variant:<14} {m['n_params']:>8,} {m['compression']:>5.1f}×  "
              f"{m['accuracy']:>7.4f} {m['auc_roc']:>9.4f} "
              f"{m['f1']:>7.4f} {m['kappa']:>8.4f}")

    # Añadir teacher al resumen
    print(f"\nTeacher (baseline): {n_teacher:,} params")


if __name__ == "__main__":
    main()
