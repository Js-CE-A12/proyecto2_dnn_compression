"""
Fase 2 — Entrenamiento del modelo baseline CNN para detección binaria de apnea.
Optimizer: Adam lr=1e-4 | Batch: 32 | Epochs: 200 | Early stopping paciencia 20
Loss: weighted cross-entropy (pesos de class_weights_apnea.json)
Scheduler: ReduceLROnPlateau (factor=0.5, paciencia=10)
Salidas:
  checkpoints/sleep_baseline.pth      (mejor checkpoint por val AUC-ROC)
  checkpoints/sleep_baseline_full.pth (último estado al final del entrenamiento)
  results/metrics_baseline.json
  configs/config.json
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models import SleepApneaCNN

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
PROCESSED_DIR = Path("data/processed")
CKPT_DIR      = Path("checkpoints")
RESULTS_DIR   = Path("results")
CONFIGS_DIR   = Path("configs")

for _d in (CKPT_DIR, RESULTS_DIR, CONFIGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

SEED      = 42
LR        = 1e-4
BATCH     = 32
EPOCHS    = 200
PATIENCE  = 30

torch.manual_seed(SEED)
np.random.seed(SEED)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def augment(x: np.ndarray) -> np.ndarray:
    """Augmentacion online para señales EEG (1, 3000). Solo en training."""
    x = x.copy()
    # Ruido gaussiano
    if np.random.rand() < 0.5:
        x += np.random.normal(0, 0.05, x.shape).astype(np.float32)
    # Escala de amplitud
    if np.random.rand() < 0.5:
        x *= np.random.uniform(0.8, 1.2)
    # Inversion de polaridad
    if np.random.rand() < 0.3:
        x *= -1
    # Time shift ±100 muestras (~1 segundo)
    if np.random.rand() < 0.5:
        shift = np.random.randint(-100, 101)
        x = np.roll(x, shift, axis=1)
    return x


class ApneaDataset(Dataset):
    def __init__(self, split: str, augment_data: bool = False) -> None:
        self.X = np.load(PROCESSED_DIR / f"X_{split}.npy")           # (N, 1, 3000)
        self.y = np.load(PROCESSED_DIR / f"y_apnea_{split}.npy").astype(np.int64)
        self.augment_data = augment_data

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        x = self.X[idx]
        if self.augment_data:
            x = augment(x)
        return torch.from_numpy(x), int(self.y[idx])


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_prob: np.ndarray) -> dict:
    acc  = accuracy_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    auc  = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
    mcc  = matthews_corrcoef(y_true, y_pred)
    tp   = int(((y_pred == 1) & (y_true == 1)).sum())
    tn   = int(((y_pred == 0) & (y_true == 0)).sum())
    fp   = int(((y_pred == 1) & (y_true == 0)).sum())
    fn   = int(((y_pred == 0) & (y_true == 1)).sum())
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        "accuracy":    round(float(acc),  4),
        "f1":          round(float(f1),   4),
        "auc_roc":     round(float(auc),  4),
        "mcc":         round(float(mcc),  4),
        "sensitivity": round(float(sens), 4),
        "specificity": round(float(spec), 4),
    }


# ---------------------------------------------------------------------------
# Entrenamiento / evaluación por época
# ---------------------------------------------------------------------------

def train_epoch(model: nn.Module, loader: DataLoader,
                criterion: nn.Module, optimizer: torch.optim.Optimizer,
                device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.long().to(device)
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model: nn.Module, loader: DataLoader,
               criterion: nn.Module, device: torch.device) -> tuple:
    model.eval()
    total_loss = 0.0
    all_labels, all_preds, all_probs = [], [], []
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_t = y_batch.long().to(device)
        logits = model(X_batch)
        total_loss += criterion(logits, y_t).item() * len(y_t)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()
        all_probs.extend(probs.tolist())
        all_preds.extend(preds.tolist())
        all_labels.extend(y_batch.tolist())
    avg_loss = total_loss / len(loader.dataset)
    metrics  = compute_metrics(np.array(all_labels), np.array(all_preds),
                                np.array(all_probs))
    return avg_loss, metrics


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

    # ---- Data ----
    train_ds = ApneaDataset("train", augment_data=True)
    val_ds   = ApneaDataset("val")
    test_ds  = ApneaDataset("test")
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False, num_workers=0)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    # ---- Pesos de clase ----
    with open(PROCESSED_DIR / "class_weights_apnea.json") as f:
        cw = json.load(f)
    weights = torch.tensor([cw["0"], cw["1"]], dtype=torch.float32).to(device)

    # ---- Modelo ----
    model = SleepApneaCNN(num_classes=2).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parámetros: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(weight=weights)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=10
    )

    # ---- Loop de entrenamiento ----
    best_val_auc = -1.0
    no_improve   = 0
    history      = []

    print(f"\n{'Epoch':>6} {'TrainLoss':>10} {'ValLoss':>9} "
          f"{'ValAUC':>8} {'ValF1':>7} {'Sens':>7} {'Spec':>7} {'LR':>9}")
    print("-" * 72)

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss            = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics = eval_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        lr_now = optimizer.param_groups[0]["lr"]
        entry  = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss":   round(val_loss,   4),
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(entry)

        print(f"{epoch:>6d} {train_loss:>10.4f} {val_loss:>9.4f} "
              f"{val_metrics['auc_roc']:>8.4f} {val_metrics['f1']:>7.4f} "
              f"{val_metrics['sensitivity']:>7.4f} {val_metrics['specificity']:>7.4f} "
              f"{lr_now:>9.2e}  ({time.time()-t0:.1f}s)")

        if val_metrics["auc_roc"] > best_val_auc:
            best_val_auc = val_metrics["auc_roc"]
            torch.save(model.state_dict(), CKPT_DIR / "sleep_baseline.pth")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"\nEarly stopping en época {epoch} (sin mejora en {PATIENCE} épocas).")
                break

    # ---- Guardar modelo final ----
    torch.save(model.state_dict(), CKPT_DIR / "sleep_baseline_full.pth")

    # ---- Evaluación en test (mejor checkpoint) ----
    model.load_state_dict(torch.load(CKPT_DIR / "sleep_baseline.pth",
                                     map_location=device))
    _, test_metrics = eval_epoch(model, test_loader, criterion, device)
    print(f"\nTest metrics: {test_metrics}")

    # ---- Guardar métricas ----
    metrics_out = {
        "best_val_auc_roc": round(best_val_auc, 4),
        "test": test_metrics,
        "history": history,
    }
    with open(RESULTS_DIR / "metrics_baseline.json", "w") as f:
        json.dump(metrics_out, f, indent=2)

    # ---- Guardar config ----
    config = {
        "seed":                    SEED,
        "model":                   "SleepApneaCNN",
        "task":                    "binary_apnea",
        "input_shape":             [1, 3000],
        "num_classes":             2,
        "optimizer":               "Adam",
        "lr":                      LR,
        "batch_size":              BATCH,
        "epochs":                  EPOCHS,
        "early_stopping_patience": PATIENCE,
        "loss":                    "weighted_cross_entropy",
        "class_weights":           {k: float(v) for k, v in cw.items()},
        "scheduler":               "ReduceLROnPlateau",
        "scheduler_factor":        0.5,
        "scheduler_patience":      10,
        "device":                  str(device),
        "n_params":                n_params,
    }
    with open(CONFIGS_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nGuardado:")
    print(f"  {CKPT_DIR}/sleep_baseline.pth")
    print(f"  {CKPT_DIR}/sleep_baseline_full.pth")
    print(f"  {RESULTS_DIR}/metrics_baseline.json")
    print(f"  {CONFIGS_DIR}/config.json")


if __name__ == "__main__":
    main()
