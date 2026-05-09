"""
Fase 3 — Preprocesamiento del dataset ISRUC-Sleep para apnea binaria.

ISRUC-Sleep: ~30 sujetos, canal C3-A2, 200 Hz → remuestreado a 100 Hz
Épocas de 30 s (3000 muestras). Etiqueta de apnea por época desde xlsx.

Salidas en data/processed/isruc_subjects/:
  subj_NNN.npz  →  {"X": (N,1,3000), "y_apnea": (N,)}

Solo se procesan sujetos con ≥2% de épocas de apnea.
"""

import os
import shutil
import tempfile
import warnings
from pathlib import Path

import mne
import numpy as np
import openpyxl

warnings.filterwarnings("ignore", category=RuntimeWarning)
mne.set_log_level("ERROR")

# ---------------------------------------------------------------------------
ISRUC_DIR = Path("data/raw/ISRUC SLEEP")
OUT_DIR   = Path("data/processed/isruc_subjects")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FS_TARGET  = 100
EPOCH_SEC  = 30
EPOCH_SAMP = FS_TARGET * EPOCH_SEC   # 3000

APNEA_EVENTS = {"OA", "CA", "OH", "MA", "MH"}

# Subjects to skip (apnea rate < 2%)
SKIP_SUBJECTS = {3, 4, 6, 11, 14, 25, 27, 29}

EEG_CANDIDATES = ["C3-A2", "C3A2", "EEG C3-A2"]

# ---------------------------------------------------------------------------

def load_isruc_eeg(rec_path: Path):
    """Read .rec EDF, pick C3-A2, resample to 100 Hz. Returns (signal_1d,)."""
    tmp = tempfile.NamedTemporaryFile(suffix=".edf", delete=False)
    tmp.close()
    shutil.copy2(str(rec_path), tmp.name)
    try:
        raw = mne.io.read_raw_edf(tmp.name, preload=True, verbose=False, exclude=[])
    finally:
        os.unlink(tmp.name)

    ch_lower = {ch.lower(): ch for ch in raw.ch_names}
    selected = None
    for cand in EEG_CANDIDATES:
        if cand.lower() in ch_lower:
            selected = ch_lower[cand.lower()]
            break
    if selected is None:
        eeg_chs = [ch for ch in raw.ch_names if "eeg" in ch.lower()]
        selected = eeg_chs[0] if eeg_chs else raw.ch_names[0]
        print(f"  [WARN] C3-A2 not found; using '{selected}'")

    raw.pick([selected])
    if abs(raw.info["sfreq"] - FS_TARGET) > 1:
        raw.resample(FS_TARGET, npad="auto")

    return raw.get_data()[0]   # (n_samples,)


def load_isruc_apnea_labels(subj_dir: Path, n_epochs: int):
    """Parse *_1.xlsx → binary apnea label per epoch."""
    xlsx_files = list(subj_dir.glob("*_1.xlsx"))
    if not xlsx_files:
        return None

    wb = openpyxl.load_workbook(str(xlsx_files[0]), read_only=True, data_only=True)
    ws = wb.active
    rows = list(ws.iter_rows(values_only=True))
    wb.close()

    evt_col    = 4   # default Events column index
    data_start = 0

    for i, row in enumerate(rows):
        if not row or row[0] is None:
            continue
        row_strs = [str(c).strip().lower() if c else "" for c in row]
        if any("epoch" in s or "hich" in s for s in row_strs[:3]):
            for j, h in enumerate(row_strs):
                if "events" in h:
                    evt_col = j
                    break
            data_start = i + 1
            break
        elif isinstance(row[0], (int, float)) and row[0] == 1:
            data_start = i
            break

    apnea = np.zeros(n_epochs, dtype=np.int8)
    for row in rows[data_start:]:
        if not row or row[0] is None:
            continue
        try:
            ep_idx = int(row[0]) - 1   # 1-based → 0-based
        except (ValueError, TypeError):
            continue
        if ep_idx < 0 or ep_idx >= n_epochs:
            continue
        evt = str(row[evt_col]).strip() if len(row) > evt_col and row[evt_col] else ""
        if evt and evt != "None" and any(e in evt for e in APNEA_EVENTS):
            apnea[ep_idx] = 1

    return apnea


def process_subject(subj_id: int):
    subj_dir = ISRUC_DIR / str(subj_id)
    if not subj_dir.exists():
        return

    rec_files = list(subj_dir.glob("*.rec"))
    if not rec_files:
        print(f"  [SKIP] ISRUC-{subj_id}: no .rec file")
        return

    print(f"  Processing ISRUC-{subj_id}...")
    signal = load_isruc_eeg(rec_files[0])

    n_epochs = len(signal) // EPOCH_SAMP

    X = np.zeros((n_epochs, 1, EPOCH_SAMP), dtype=np.float32)
    for i in range(n_epochs):
        s = i * EPOCH_SAMP
        X[i, 0, :] = signal[s:s + EPOCH_SAMP]

    y_apnea = load_isruc_apnea_labels(subj_dir, n_epochs)
    if y_apnea is None:
        print(f"  [SKIP] ISRUC-{subj_id}: no xlsx labels")
        return

    # Trim to shorter of signal or labels
    n = min(len(X), len(y_apnea))
    X, y_apnea = X[:n], y_apnea[:n]

    # Per-epoch z-score normalisation
    mean = X.mean(axis=2, keepdims=True)
    std  = X.std(axis=2, keepdims=True) + 1e-8
    X    = (X - mean) / std

    pct = 100 * y_apnea.mean()
    print(f"    epochs={n}  apnea={y_apnea.sum()} ({pct:.1f}%)")

    out_path = OUT_DIR / f"subj_{subj_id:03d}.npz"
    np.savez_compressed(str(out_path), X=X, y_apnea=y_apnea)
    return X, y_apnea


def main():
    print("=" * 60)
    print("ISRUC-Sleep Preprocessing")
    print("=" * 60)

    subject_dirs = sorted(
        [d for d in ISRUC_DIR.iterdir() if d.is_dir()],
        key=lambda d: int(d.name) if d.name.isdigit() else 999,
    )

    processed = 0
    for subj_dir in subject_dirs:
        try:
            subj_id = int(subj_dir.name)
        except ValueError:
            continue
        if subj_id in SKIP_SUBJECTS:
            print(f"  [SKIP] ISRUC-{subj_id}: low apnea rate")
            continue
        result = process_subject(subj_id)
        if result is not None:
            processed += 1

    print(f"\nDone. Processed {processed} ISRUC subjects → {OUT_DIR}")


if __name__ == "__main__":
    main()
