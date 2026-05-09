"""
Fase 2b - Cross-validacion 5-fold por sujeto sobre UCDDB + ISRUC-Sleep

Divide los 25 sujetos UCDDB en 5 grupos de 5. Para cada fold:
  test:  5 sujetos UCDDB del fold actual
  val:   3 sujetos UCDDB del bloque siguiente (early stopping)
  train: 17 sujetos UCDDB restantes + 22 sujetos ISRUC (datos extra)

Reporta metricas promedio +/- std sobre los 5 folds.

Salidas:
  data/processed/subjects/subj_NNN.npz        cache UCDDB por sujeto
  data/processed/isruc_subjects/subj_NNN.npz  cache ISRUC (03_isruc_preprocess.py)
  results/crossval_metrics.json
  checkpoints/crossval_best.pth               mejor checkpoint (mayor val AUC)
"""

import gc
import json
import os
import re
import shutil
import sys
import tempfile
import time
import warnings
from pathlib import Path

import mne
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore", category=RuntimeWarning)
mne.set_log_level("ERROR")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models import SleepApneaCNN

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RAW_DIR = Path(
    "data/raw/st-vincents-university-hospital-university-college-dublin-"
    "sleep-apnea-database-1.0.0/files"
)
PROCESSED_DIR = Path("data/processed")
SUBJ_DIR      = PROCESSED_DIR / "subjects"
ISRUC_DIR     = PROCESSED_DIR / "isruc_subjects"
CKPT_DIR      = Path("checkpoints")
RESULTS_DIR   = Path("results")

for _d in (SUBJ_DIR, CKPT_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Hiperparametros
# ---------------------------------------------------------------------------
SEED     = 42
LR       = 1e-4
BATCH    = 32
EPOCHS   = 200
PATIENCE = 30

FS_TARGET  = 100
EPOCH_SEC  = 30
EPOCH_SAMP = FS_TARGET * EPOCH_SEC   # 3000 muestras

torch.manual_seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Constantes EEG / UCDDB
# ---------------------------------------------------------------------------
EEG_CHANNEL_CANDIDATES = ["C3A2", "C3-A2", "EEG C3-A2", "EEG1"]
APNEA_TYPES = {"APNEA-O", "APNEA-C", "APNEA-M", "APNEA-U",
               "HYP-O",   "HYP-C",   "HYP-M",   "HYP"}
RK_TO_AASM  = {0: 0, 1: 4, 2: 1, 3: 2, 4: 3, 5: 3}

ALL_SUBJECTS = [2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]

# 5 folds, 5 sujetos de test cada uno
FOLDS = [
    [2,  3,  5,  6,  7],
    [8,  9,  10, 11, 12],
    [13, 14, 15, 17, 18],
    [19, 20, 21, 22, 23],
    [24, 25, 26, 27, 28],
]

# 3 sujetos de validacion para cada fold (bloque siguiente, ciclico)
VAL_FOR_FOLD = [
    [8,  9,  10],
    [13, 14, 15],
    [19, 20, 21],
    [24, 25, 26],
    [2,  3,  5],
]

# ---------------------------------------------------------------------------
# Procesamiento EEG (replica de 01_prepare_data.py)
# ---------------------------------------------------------------------------

def _load_eeg_channel(rec_path: Path):
    tmp = tempfile.NamedTemporaryFile(suffix=".edf", delete=False)
    tmp.close()
    shutil.copy2(rec_path, tmp.name)
    try:
        raw = mne.io.read_raw_edf(tmp.name, preload=True, verbose=False, exclude=[])
    finally:
        os.unlink(tmp.name)

    ch_lower = {c.lower(): c for c in raw.ch_names}
    selected = None
    for cand in EEG_CHANNEL_CANDIDATES:
        if cand.lower() in ch_lower:
            selected = ch_lower[cand.lower()]
            break
    if selected is None:
        eeg_chs = [c for c in raw.ch_names if "eeg" in c.lower()]
        selected = eeg_chs[0] if eeg_chs else raw.ch_names[0]

    raw.pick([selected])
    if abs(raw.info["sfreq"] - FS_TARGET) > 1:
        raw.resample(FS_TARGET, npad="auto")
    return raw.get_data()[0]


def _load_staging(stage_path: Path) -> np.ndarray:
    with open(stage_path, "r") as f:
        lines = [l.strip() for l in f if l.strip()]
    rk = np.array([int(l) for l in lines], dtype=np.int8)
    aasm = np.full_like(rk, -1)
    for rk_val, aa_val in RK_TO_AASM.items():
        aasm[rk == rk_val] = aa_val
    return aasm


def _parse_time(t: str) -> float:
    h, m, s = t.strip().split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def _load_apnea(respevt_path: Path, n_epochs: int, rec_start: float) -> np.ndarray:
    labels = np.zeros(n_epochs, dtype=np.int8)
    with open(respevt_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    pat = re.compile(r"(\d{2}:\d{2}:\d{2})\s+([A-Z\-]+)\s+(?:[A-Z/]+\s+)?(\d+)")
    for m in pat.finditer(content):
        ev_type = m.group(2)
        if ev_type not in APNEA_TYPES:
            continue
        rel = _parse_time(m.group(1)) - rec_start
        if rel < 0:
            rel += 86400
        dur = float(m.group(3))
        end = rel + dur
        i0  = max(0, int(rel // EPOCH_SEC))
        i1  = min(n_epochs - 1, int(end // EPOCH_SEC))
        for ep in range(i0, i1 + 1):
            ov = min(end, (ep + 1) * EPOCH_SEC) - max(rel, ep * EPOCH_SEC)
            if ov >= 1.0:
                labels[ep] = 1
    return labels


def _rec_start_time(rec_path: Path) -> float:
    try:
        tmp = tempfile.NamedTemporaryFile(suffix=".edf", delete=False)
        tmp.close()
        shutil.copy2(rec_path, tmp.name)
        raw = mne.io.read_raw_edf(tmp.name, preload=True, verbose=False, exclude=[])
        os.unlink(tmp.name)
        t = raw.info["meas_date"]
        if t is None:
            return 0.0
        return t.hour * 3600 + t.minute * 60 + t.second
    except Exception:
        return 0.0


def process_subject(subj_id: int):
    """Carga y procesa un sujeto. Retorna (X, y_apnea) o None."""
    sid          = f"ucddb{subj_id:03d}"
    rec_path     = RAW_DIR / f"{sid}.rec"
    stage_path   = RAW_DIR / f"{sid}_stage.txt"
    respevt_path = RAW_DIR / f"{sid}_respevt.txt"

    if not rec_path.exists():
        print(f"    [SKIP] {sid}: .rec no encontrado")
        return None

    signal    = _load_eeg_channel(rec_path)
    y_staging = _load_staging(stage_path)
    n_epochs  = min(len(y_staging), len(signal) // EPOCH_SAMP)

    X = np.zeros((n_epochs, 1, EPOCH_SAMP), dtype=np.float32)
    for i in range(n_epochs):
        X[i, 0, :] = signal[i * EPOCH_SAMP:(i + 1) * EPOCH_SAMP]

    y_staging = y_staging[:n_epochs]
    y_apnea   = _load_apnea(respevt_path, n_epochs, _rec_start_time(rec_path))

    valid    = y_staging != -1
    X        = X[valid]
    y_apnea  = y_apnea[valid]

    mu  = X.mean(axis=2, keepdims=True)
    sig = X.std(axis=2, keepdims=True) + 1e-8
    X   = (X - mu) / sig

    pos = int(y_apnea.sum())
    pct = 100 * pos / max(1, len(y_apnea))
    print(f"    subj {subj_id:03d}: {len(X)} epochs | apnea={pos} ({pct:.1f}%)")
    return X, y_apnea


# ---------------------------------------------------------------------------
# Cache por sujeto
# ---------------------------------------------------------------------------

def load_subject(subj_id: int):
    cache = SUBJ_DIR / f"subj_{subj_id:03d}.npz"
    if cache.exists():
        try:
            d = np.load(str(cache))
            return d["X"], d["y_apnea"]
        except Exception:
            print(f"  [WARN] Cache corrupto para sujeto {subj_id}, reprocesando...")
            cache.unlink(missing_ok=True)
    print(f"  Procesando sujeto {subj_id} desde raw...")
    result = process_subject(subj_id)
    if result is None:
        return None, None
    X, y = result
    np.savez_compressed(str(cache), X=X, y_apnea=y)
    return X, y


def load_isruc_subjects() -> tuple:
    """Carga todos los sujetos ISRUC preprocesados como (X, y) concatenados."""
    Xs, ys = [], []
    if not ISRUC_DIR.exists():
        print("  [INFO] ISRUC no disponible (ejecuta 03_isruc_preprocess.py)")
        return None, None
    for npz in sorted(ISRUC_DIR.glob("subj_*.npz"),
                      key=lambda p: int(p.stem.split("_")[1])):
        try:
            d = np.load(str(npz))
            Xs.append(d["X"]); ys.append(d["y_apnea"])
        except Exception as e:
            print(f"  [WARN] {npz.name}: {e}")
    if not Xs:
        return None, None
    X = np.concatenate(Xs); y = np.concatenate(ys)
    print(f"  ISRUC: {len(Xs)} sujetos | {len(X)} epochs | apnea={y.mean():.1%}")
    return X, y


# ---------------------------------------------------------------------------
# Augmentacion
# ---------------------------------------------------------------------------

def augment(x: np.ndarray) -> np.ndarray:
    x = x.copy()
    if np.random.rand() < 0.5:
        x += np.random.normal(0, 0.05, x.shape).astype(np.float32)
    if np.random.rand() < 0.5:
        x *= np.random.uniform(0.8, 1.2)
    if np.random.rand() < 0.3:
        x *= -1
    if np.random.rand() < 0.5:
        shift = np.random.randint(-100, 101)
        x = np.roll(x, shift, axis=1)
    return x


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ApneaDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, aug: bool = False):
        self.X   = X
        self.y   = y.astype(np.int64)
        self.aug = aug

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.aug:
            x = augment(x)
        return torch.from_numpy(x), int(self.y[idx])


# ---------------------------------------------------------------------------
# Metricas
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_prob: np.ndarray) -> dict:
    auc  = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
    f1   = f1_score(y_true, y_pred, zero_division=0)
    acc  = accuracy_score(y_true, y_pred)
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
# Entrenamiento / evaluacion por epoca
# ---------------------------------------------------------------------------

def train_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    total = 0.0
    for Xb, yb in loader:
        Xb = Xb.to(device)
        yb = yb.long().to(device)
        optimizer.zero_grad()
        loss = criterion(model(Xb), yb)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(yb)
    return total / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device) -> tuple:
    model.eval()
    total = 0.0
    labels_all, preds_all, probs_all = [], [], []
    for Xb, yb in loader:
        Xb = Xb.to(device)
        yt = yb.long().to(device)
        logits = model(Xb)
        total += criterion(logits, yt).item() * len(yt)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds = logits.argmax(dim=1).cpu().numpy()
        probs_all.extend(probs.tolist())
        preds_all.extend(preds.tolist())
        labels_all.extend(yb.tolist())
    avg_loss = total / len(loader.dataset)
    m = compute_metrics(np.array(labels_all), np.array(preds_all),
                        np.array(probs_all))
    return avg_loss, m


# ---------------------------------------------------------------------------
# Un fold
# ---------------------------------------------------------------------------

def gather(subjs: list, all_data: dict):
    Xs, ys = [], []
    for s in subjs:
        X, y = all_data.get(s, (None, None))
        if X is not None:
            Xs.append(X)
            ys.append(y)
    if not Xs:
        return None, None
    return np.concatenate(Xs), np.concatenate(ys)


def run_fold(fold_idx: int, train_subjs: list, val_subjs: list,
             test_subjs: list, all_data: dict, device: torch.device,
             isruc_X=None, isruc_y=None):

    print(f"\n{'='*60}")
    print(f"Fold {fold_idx+1}/5")
    print(f"  test : {test_subjs}")
    print(f"  val  : {val_subjs}")
    print(f"  train: {train_subjs}" +
          (f" + ISRUC({len(isruc_X)}ep)" if isruc_X is not None else ""))
    print(f"{'='*60}")

    X_train, y_train = gather(train_subjs, all_data)
    X_val,   y_val   = gather(val_subjs,   all_data)
    X_test,  y_test  = gather(test_subjs,  all_data)

    # Agregar datos ISRUC al train
    if isruc_X is not None and X_train is not None:
        X_train = np.concatenate([X_train, isruc_X])
        y_train = np.concatenate([y_train, isruc_y])

    for name, Xa in [("train", X_train), ("val", X_val), ("test", X_test)]:
        if Xa is None:
            print(f"  [SKIP] fold {fold_idx+1}: sin datos en {name}")
            return None

    print(f"  train={len(X_train)} | val={len(X_val)} | test={len(X_test)}")

    # Pesos de clase desde train
    try:
        ws = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train)
    except Exception:
        ws = np.array([1.0, 1.0])
    weights = torch.tensor(ws, dtype=torch.float32).to(device)
    print(f"  class weights: 0={ws[0]:.3f} 1={ws[1]:.3f}")

    train_ds = ApneaDataset(X_train, y_train, aug=True)
    val_ds   = ApneaDataset(X_val,   y_val)
    test_ds  = ApneaDataset(X_test,  y_test)

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False, num_workers=0)

    model     = SleepApneaCNN(num_classes=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(weight=weights)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=10)

    best_val_auc = -1.0
    best_state   = None
    no_improve   = 0

    print(f"\n  {'Ep':>4} {'TrLoss':>9} {'VaLoss':>9} {'VaAUC':>8} "
          f"{'VaF1':>7} {'Sens':>7} {'Spec':>7}")
    print("  " + "-" * 58)

    for epoch in range(1, EPOCHS + 1):
        t0       = time.time()
        tr_loss  = train_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_m = eval_epoch(model, val_loader, criterion, device)
        scheduler.step(va_loss)

        va_auc = va_m["auc_roc"]
        elapsed = time.time() - t0

        if epoch <= 5 or epoch % 10 == 0:
            print(f"  {epoch:>4d} {tr_loss:>9.4f} {va_loss:>9.4f} {va_auc:>8.4f} "
                  f"{va_m['f1']:>7.4f} {va_m['sensitivity']:>7.4f} "
                  f"{va_m['specificity']:>7.4f}  ({elapsed:.1f}s)")

        if va_auc > best_val_auc:
            best_val_auc = va_auc
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve   = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  Early stopping en epoca {epoch}")
                break

    # Evaluar con mejor checkpoint
    model.load_state_dict(best_state)
    _, test_m = eval_epoch(model, test_loader, criterion, device)
    _, val_m  = eval_epoch(model, val_loader,  criterion, device)

    print(f"\n  Fold {fold_idx+1} TEST -> "
          f"AUC={test_m['auc_roc']:.4f}  F1={test_m['f1']:.4f}  "
          f"Sens={test_m['sensitivity']:.4f}  Spec={test_m['specificity']:.4f}")

    return {
        "fold":           fold_idx + 1,
        "train_subjects": train_subjs,
        "val_subjects":   val_subjs,
        "test_subjects":  test_subjs,
        "n_train":        len(train_ds),
        "n_val":          len(val_ds),
        "n_test":         len(test_ds),
        "best_val_auc":   round(best_val_auc, 4),
        "val":            val_m,
        "test":           test_m,
        "_state":         best_state,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    device = torch.device(
        "mps"  if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available()          else
        "cpu"
    )
    print(f"Device: {device}")

    # 1. Cargar / cachear todos los sujetos UCDDB
    print("\n" + "=" * 60)
    print("Cargando datos por sujeto (cache en data/processed/subjects/)")
    print("=" * 60)
    all_data: dict = {}
    for sid in ALL_SUBJECTS:
        X, y = load_subject(sid)
        all_data[sid] = (X, y)
        if X is not None:
            pos = int(y.sum())
            print(f"  subj {sid:03d} cargado: {len(X)} epochs | apnea={pos} ({100*pos/max(1,len(y)):.1f}%)")

    # 1b. Cargar sujetos ISRUC (datos extra de train)
    print("\n" + "=" * 60)
    print("Cargando datos ISRUC-Sleep (extra train)")
    print("=" * 60)
    isruc_X, isruc_y = load_isruc_subjects()

    # 2. 5-fold CV
    print("\n" + "=" * 60)
    print("5-fold cross-validacion por sujeto")
    print("=" * 60)

    fold_results     = []
    best_fold_auc    = -1.0
    best_state_global = None

    for fi, test_subjs in enumerate(FOLDS):
        val_subjs   = VAL_FOR_FOLD[fi]
        train_subjs = [s for s in ALL_SUBJECTS
                       if s not in test_subjs and s not in val_subjs]

        res = run_fold(fi, train_subjs, val_subjs, test_subjs, all_data, device,
                      isruc_X=isruc_X, isruc_y=isruc_y)
        if res is None:
            continue

        state = res.pop("_state")
        fold_results.append(res)

        if res["best_val_auc"] > best_fold_auc:
            best_fold_auc    = res["best_val_auc"]
            best_state_global = state

        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # 3. Resumen final
    if not fold_results:
        print("Sin resultados.")
        return

    print("\n" + "=" * 60)
    print("RESUMEN POR FOLD")
    print("=" * 60)
    print(f"  {'Fold':>4} {'AUC':>8} {'F1':>8} {'Sens':>8} {'Spec':>8} {'MCC':>8}")
    print("  " + "-" * 48)
    for r in fold_results:
        t = r["test"]
        print(f"  {r['fold']:>4}  {t['auc_roc']:>7.4f}  {t['f1']:>7.4f}  "
              f"{t['sensitivity']:>7.4f}  {t['specificity']:>7.4f}  {t['mcc']:>7.4f}")

    keys = ("accuracy", "f1", "auc_roc", "mcc", "sensitivity", "specificity")
    means = {k: round(float(np.mean([r["test"][k] for r in fold_results])), 4) for k in keys}
    stds  = {k: round(float(np.std( [r["test"][k] for r in fold_results])), 4) for k in keys}

    print(f"\n  Media +/- std:")
    for k in keys:
        print(f"    {k:<14}: {means[k]:.4f} +/- {stds[k]:.4f}")

    # 4. Guardar mejor checkpoint
    if best_state_global is not None:
        torch.save(best_state_global, CKPT_DIR / "crossval_best.pth")
        print(f"\nMejor checkpoint: checkpoints/crossval_best.pth")

    # 5. Guardar JSON
    out = {
        "n_folds":       len(fold_results),
        "mean_test":     means,
        "std_test":      stds,
        "folds":         fold_results,
    }
    out_path = RESULTS_DIR / "crossval_metrics.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Guardado: {out_path}")
    print("Fase 2b completada.")


if __name__ == "__main__":
    main()
