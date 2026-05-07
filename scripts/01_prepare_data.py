"""
Fase 1 — Preparación del dataset UCDDB
Dataset: St. Vincent's University Hospital / UCD Sleep Apnea Database
- Lee señales EEG de archivos .rec (EDF) con MNE
- Canal: C3-A2 @ 100 Hz → épocas de 30 s (3000 muestras)
- Staging: R&K → AASM (W=0, N1=1, N2=2, N3=3, REM=4), descarta MT (6)
- Apnea: binario por época (1 si ≥1 evento overlap, 0 si no)
- Partición por sujeto: 18 train / 4 val / 3 test
Salidas en data/processed/:
  X_train.npy, y_staging_train.npy, y_apnea_train.npy
  X_val.npy,   y_staging_val.npy,   y_apnea_val.npy
  X_test.npy,  y_staging_test.npy,  y_apnea_test.npy
  split_subjects.json, class_weights_staging.json, class_weights_apnea.json
"""

import json
import os
import re
import shutil
import tempfile
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import mne
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore", category=RuntimeWarning)
mne.set_log_level("ERROR")

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
RAW_DIR = Path("data/raw/st-vincents-university-hospital-university-college-dublin-sleep-apnea-database-1.0.0/files")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
np.random.seed(SEED)

FS_TARGET   = 100    # Hz tras re-muestreo
EPOCH_SEC   = 30     # segundos por época
EPOCH_SAMP  = FS_TARGET * EPOCH_SEC  # 3000 muestras

# Sujetos disponibles (sin 001, 004, 016)
ALL_SUBJECTS = [2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]

# Partición por sujeto (19 / 3 / 3)
# Sujeto 025 (83% apnea) se mueve a train para no sesgar la validación.
# Val queda con sujetos 22/23/24 (5%, 29%, 24% apnea) → representativo del test.
SPLIT = {
    "train": [2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 25],
    "val":   [22, 23, 24],
    "test":  [26, 27, 28],
}

# Mapeo Rechtschaffen & Kales → AASM
#   R&K: 0=Wake, 1=REM, 2=S1, 3=S2, 4=S3, 5=S4, 6=MT
#   AASM: 0=W, 1=N1, 2=N2, 3=N3, 4=REM
RK_TO_AASM = {0: 0, 1: 4, 2: 1, 3: 2, 4: 3, 5: 3}
DISCARD_LABEL = 6  # MT (Movement Time)

# Nombre del canal EEG en los archivos EDF (UCDDB usa "C3A2" sin guion)
EEG_CHANNEL_CANDIDATES = ["C3A2", "C3-A2", "EEG C3-A2", "EEG1"]

# Tipos de eventos que cuentan como apnea/hipopnea
APNEA_TYPES = {"APNEA-O", "APNEA-C", "APNEA-M", "APNEA-U",
               "HYP-O", "HYP-C", "HYP-M", "HYP"}


# ---------------------------------------------------------------------------
# Funciones de carga
# ---------------------------------------------------------------------------

def load_eeg_channel(rec_path: Path) -> object:
    """
    Lee el archivo EDF (.rec), selecciona el canal C3-A2 y re-muestrea a FS_TARGET.
    Devuelve (signal_1d, fs_original).
    Copia a .edf temporal porque MNE rechaza la extensión .rec.
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".edf", delete=False)
    tmp.close()
    shutil.copy2(rec_path, tmp.name)
    try:
        raw = mne.io.read_raw_edf(tmp.name, preload=True, verbose=False, exclude=[])
    finally:
        os.unlink(tmp.name)

    # Buscar canal C3-A2 con distintos nombres posibles
    ch_names_lower = {ch.lower(): ch for ch in raw.ch_names}
    selected = None
    for cand in EEG_CHANNEL_CANDIDATES:
        if cand.lower() in ch_names_lower:
            selected = ch_names_lower[cand.lower()]
            break

    if selected is None:
        # Fallback: usar el primer canal que empiece con "EEG" o el primero de todos
        eeg_chs = [ch for ch in raw.ch_names if "eeg" in ch.lower()]
        selected = eeg_chs[0] if eeg_chs else raw.ch_names[0]
        print(f"  [WARN] C3-A2 no encontrado en {rec_path.name}; usando '{selected}'")

    raw.pick([selected])

    fs_orig = raw.info["sfreq"]
    if abs(fs_orig - FS_TARGET) > 1:
        raw.resample(FS_TARGET, npad="auto")

    signal = raw.get_data()[0]  # (n_samples,)
    return signal, fs_orig


def load_staging_labels(stage_path: Path) -> np.ndarray:
    """
    Lee _stage.txt (un entero R&K por línea) y devuelve array AASM.
    Épocas con MT (6) quedan como -1 para ser descartadas.
    """
    with open(stage_path, "r") as f:
        lines = [l.strip() for l in f if l.strip()]
    rk_labels = np.array([int(l) for l in lines], dtype=np.int8)
    aasm = np.full_like(rk_labels, fill_value=-1)
    for rk, aa in RK_TO_AASM.items():
        aasm[rk_labels == rk] = aa
    return aasm  # -1 = descartar


def _parse_time(t_str: str) -> float:
    """HH:MM:SS → segundos desde medianoche."""
    h, m, s = t_str.strip().split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def load_apnea_labels(respevt_path: Path, n_epochs: int,
                      rec_start_sec: float) -> np.ndarray:
    """
    Parsea _respevt.txt y crea etiqueta binaria por época de 30 s.
    Un época es 1 si tiene ≥1 evento de apnea/hipopnea que se superpone ≥1 s.

    rec_start_sec: tiempo (en segundos desde medianoche) en que empieza el .rec
    """
    apnea_labels = np.zeros(n_epochs, dtype=np.int8)

    with open(respevt_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    # Cada evento tiene formato:  HH:MM:SS  TYPE  ...  DURATION  ...
    # Ejemplo: "00:29:13  HYP-C             16 ..."
    event_pattern = re.compile(
        r"(\d{2}:\d{2}:\d{2})\s+([A-Z\-]+)\s+(?:[A-Z/]+\s+)?(\d+)"
    )

    for m in event_pattern.finditer(content):
        t_str, ev_type, dur_str = m.group(1), m.group(2), m.group(3)
        if ev_type not in APNEA_TYPES:
            continue

        ev_start_abs = _parse_time(t_str)          # seg desde medianoche
        ev_start_rel = ev_start_abs - rec_start_sec # seg desde inicio del .rec
        # Cruce de medianoche: grabación inicia en la noche (ej. 23:07),
        # eventos ocurren al día siguiente (ej. 00:33) → sumar 24h
        if ev_start_rel < 0:
            ev_start_rel += 86400
        ev_dur = float(dur_str)                     # segundos de duración
        ev_end_rel = ev_start_rel + ev_dur

        # Épocas que se superponen con este evento
        epoch_start_idx = max(0, int(ev_start_rel // EPOCH_SEC))
        epoch_end_idx   = min(n_epochs - 1, int(ev_end_rel // EPOCH_SEC))
        for ep in range(epoch_start_idx, epoch_end_idx + 1):
            ep_start = ep * EPOCH_SEC
            ep_end   = ep_start + EPOCH_SEC
            overlap  = min(ev_end_rel, ep_end) - max(ev_start_rel, ep_start)
            if overlap >= 1.0:
                apnea_labels[ep] = 1

    return apnea_labels


def get_rec_start_time(rec_path: Path) -> float:
    """
    Extrae la hora de inicio del EDF en segundos desde medianoche.
    Si falla, devuelve 0.
    """
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


# ---------------------------------------------------------------------------
# Pipeline por sujeto
# ---------------------------------------------------------------------------

def process_subject(subj_id: int) -> object:
    """
    Carga, segmenta y etiqueta un sujeto.
    Devuelve (X, y_staging, y_apnea) o None si falla.
    X shape: (N_epochs, 1, 3000)
    """
    sid = f"ucddb{subj_id:03d}"
    rec_path      = RAW_DIR / f"{sid}.rec"
    stage_path    = RAW_DIR / f"{sid}_stage.txt"
    respevt_path  = RAW_DIR / f"{sid}_respevt.txt"

    if not rec_path.exists():
        print(f"  [SKIP] {sid}: archivo .rec no encontrado")
        return None

    print(f"  Procesando {sid}...")

    # 1. Señal EEG
    signal, fs_orig = load_eeg_channel(rec_path)
    print(f"    fs_orig={fs_orig:.0f} Hz  |  muestras totales={len(signal)}")

    # 2. Etiquetas de staging
    y_staging = load_staging_labels(stage_path)
    n_epochs_annot = len(y_staging)

    # 3. Extraer épocas de la señal (alinear con anotaciones)
    n_epochs_signal = len(signal) // EPOCH_SAMP
    n_epochs = min(n_epochs_annot, n_epochs_signal)

    X_all = np.zeros((n_epochs, 1, EPOCH_SAMP), dtype=np.float32)
    for i in range(n_epochs):
        start = i * EPOCH_SAMP
        X_all[i, 0, :] = signal[start:start + EPOCH_SAMP]

    y_staging = y_staging[:n_epochs]

    # 4. Etiquetas de apnea
    rec_start = get_rec_start_time(rec_path)
    y_apnea = load_apnea_labels(respevt_path, n_epochs, rec_start)

    # 5. Descartar épocas MT (staging == -1)
    valid_mask = y_staging != -1
    X_all     = X_all[valid_mask]
    y_staging = y_staging[valid_mask]
    y_apnea   = y_apnea[valid_mask]

    # 6. Normalización por época (z-score)
    mean = X_all.mean(axis=2, keepdims=True)
    std  = X_all.std(axis=2, keepdims=True) + 1e-8
    X_all = (X_all - mean) / std

    print(f"    Épocas válidas: {len(X_all)}  "
          f"| Apnea positivos: {y_apnea.sum()} ({100*y_apnea.mean():.1f}%)")
    return X_all, y_staging, y_apnea


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Fase 1 — Preprocesamiento UCDDB")
    print("=" * 60)

    split_data = {split: {"X": [], "y_stag": [], "y_apn": []}
                  for split in ("train", "val", "test")}

    subject_to_split = {}
    for split, subjs in SPLIT.items():
        for s in subjs:
            subject_to_split[s] = split

    for subj_id in ALL_SUBJECTS:
        split = subject_to_split[subj_id]
        print(f"\n[{split.upper()}] Sujeto {subj_id:03d}")
        result = process_subject(subj_id)
        if result is None:
            continue
        X, y_stag, y_apn = result
        split_data[split]["X"].append(X)
        split_data[split]["y_stag"].append(y_stag)
        split_data[split]["y_apn"].append(y_apn)

    # Concatenar y guardar
    print("\n" + "=" * 60)
    print("Guardando arrays...")
    for split in ("train", "val", "test"):
        if not split_data[split]["X"]:
            print(f"  [WARN] {split}: sin datos")
            continue

        X      = np.concatenate(split_data[split]["X"],      axis=0)
        y_stag = np.concatenate(split_data[split]["y_stag"], axis=0)
        y_apn  = np.concatenate(split_data[split]["y_apn"],  axis=0)

        np.save(OUT_DIR / f"X_{split}.npy",           X)
        np.save(OUT_DIR / f"y_staging_{split}.npy",   y_stag)
        np.save(OUT_DIR / f"y_apnea_{split}.npy",     y_apn)

        print(f"  {split}: X={X.shape}  staging_dist={dict(zip(*np.unique(y_stag, return_counts=True)))}  apnea_pos={y_apn.sum()}")

    # Pesos de clases (calculados solo con train)
    y_stag_train = np.concatenate(split_data["train"]["y_stag"])
    classes_stag = np.unique(y_stag_train)
    w_stag = compute_class_weight("balanced", classes=classes_stag, y=y_stag_train)
    class_weights_staging = {int(c): float(round(w, 4))
                              for c, w in zip(classes_stag, w_stag)}

    y_apn_train = np.concatenate(split_data["train"]["y_apn"])
    classes_apn = np.unique(y_apn_train)
    w_apn = compute_class_weight("balanced", classes=classes_apn, y=y_apn_train)
    class_weights_apnea = {int(c): float(round(w, 4))
                           for c, w in zip(classes_apn, w_apn)}

    with open(OUT_DIR / "class_weights_staging.json", "w") as f:
        json.dump(class_weights_staging, f, indent=2)
    with open(OUT_DIR / "class_weights_apnea.json", "w") as f:
        json.dump(class_weights_apnea, f, indent=2)

    # Guardar split
    with open(OUT_DIR / "split_subjects.json", "w") as f:
        json.dump(SPLIT, f, indent=2)

    print("\nPesos staging:", class_weights_staging)
    print("Pesos apnea:  ", class_weights_apnea)
    print("\nFase 1 completada. Archivos en data/processed/")


if __name__ == "__main__":
    main()