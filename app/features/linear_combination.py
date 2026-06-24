"""
linear_combination.py
=====================
Funzioni per il calcolo della combinazione lineare su Probe Request (PR).
Nessuna logica di main: tutto viene chiamato da main.py.
"""

import json
import ast
import math
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pandas as pd


# ---------------------------------------------------------------------------
# Campi esclusi dal calcolo
# ---------------------------------------------------------------------------

EXCLUDED_FIELDS = {
    "start_time",
    "end_time",
    "start_seq",
    "end_seq",
    "label",
    "mac"
}


# ---------------------------------------------------------------------------
# Caricamento file JSON
# ---------------------------------------------------------------------------

def load_json(path: str) -> list:
    """
    Carica il file JSON delle PR e restituisce sempre una lista di dizionari.

    Gestisce due formati:
      - Lista di dict:
        [{"mac": "aa:bb:...", "ie1": ..., "label": 1}, ...]

      - Dict con MAC come chiave:
        {"aa:bb:...": {"ie1": ..., "label": 1}, ...}

        In questo caso il MAC viene inserito dentro ogni record.
    """

    data = json.loads(Path(path).read_text())

    if isinstance(data, dict):
        records = []

        for mac, record in data.items():
            entry = {"mac": mac}
            entry.update(record)
            records.append(entry)

        return records

    if data and isinstance(data[0], str):
        data = [json.loads(item) for item in data]

    return data


def load_weights(path: str) -> dict:
    """
    Carica i pesi da file JSON.
    Le chiavi che iniziano con '_' vengono ignorate.
    """

    raw = json.loads(Path(path).read_text())

    return {
        key: value
        for key, value in raw.items()
        if not key.startswith("_")
    }


# ---------------------------------------------------------------------------
# Conversione dei campi in scalari
# ---------------------------------------------------------------------------

def mac_to_scalar(mac_str: str) -> float:
    """
    Converte un MAC address in uno scalare normalizzato in [0, 1].
    Esempio: 'aa:bb:cc:dd:ee:ff'
    """

    try:
        parts = mac_str.strip().split(":")

        mac_int = 0

        for byte in parts:
            mac_int = (mac_int << 8) | int(byte, 16)

        return mac_int / 0xFFFFFFFFFFFF

    except Exception:
        return 0.0


def field_to_scalar(value) -> float:
    """
    Converte il valore di un campo IE in uno scalare.

    Regole:
      - None / "None" / "" -> 0.0
      - int / float        -> float(value)
      - lista Python       -> somma degli elementi
      - stringa lista      -> somma degli elementi
      - altro              -> 0.0
    """

    if value is None:
        return 0.0

    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, list):
        try:
            return float(sum(value))
        except Exception:
            return 0.0

    if isinstance(value, str):
        s = value.strip()

        if s in ("None", ""):
            return 0.0

        try:
            parsed = ast.literal_eval(s)

            if isinstance(parsed, (list, tuple)):
                return float(sum(parsed))

            if isinstance(parsed, (int, float)):
                return float(parsed)

        except Exception:
            pass

    return 0.0


# ---------------------------------------------------------------------------
# Normalizzazione
# ---------------------------------------------------------------------------

def normalize_json_data(data: list) -> list:
    """
    Converte ogni IE in scalare e poi normalizza ogni colonna IE in [0, 1].

    Restituisce una lista di PR, non un DataFrame.
    """

    scalar_data = []

    for pr in data:
        new_pr = {}

        for key, value in pr.items():
            if key in EXCLUDED_FIELDS:
                new_pr[key] = value
            else:
                new_pr[key] = field_to_scalar(value)

        scalar_data.append(new_pr)

    df_data = pd.DataFrame(scalar_data)

    ie_columns = [
        col for col in df_data.columns
        if col not in EXCLUDED_FIELDS
    ]

    if ie_columns:
        df_data[ie_columns] = df_data[ie_columns].apply(
            pd.to_numeric,
            errors="coerce"
        )

        df_data[ie_columns] = df_data[ie_columns].fillna(0.0)

        scaler = MinMaxScaler()
        df_data[ie_columns] = scaler.fit_transform(df_data[ie_columns])

        df_data[ie_columns] = df_data[ie_columns].fillna(0.0)

    return df_data.to_dict("records")


# ---------------------------------------------------------------------------
# Calcolo score Probe Request
# ---------------------------------------------------------------------------

def calculate_score(data: list, weights: dict) -> list:
    """
    Calcola lo score di ogni PR come combinazione lineare dei pesi.

    Restituisce una lista:
    [
        {"mac": ..., "label": ..., "score": ...},
        ...
    ]
    """

    scores = []

    for pr in data:
        score = 0.0

        for key, weight in weights.items():
            if key in EXCLUDED_FIELDS:
                continue

            value = pr.get(key, 0.0)

            if value is None:
                value = 0.0

            try:
                value = float(value)
            except Exception:
                value = 0.0

            if math.isnan(value):
                value = 0.0

            score += value * weight

        scores.append({
            "mac": pr.get("mac"),
            "label": pr.get("label"),
            "score": score
        })

    return scores


# ---------------------------------------------------------------------------
# Calcolo slot e clustering
# ---------------------------------------------------------------------------

def count_real_clusters(records: list) -> int:
    """
    Conta il numero effettivo di cluster/dispositivi reali usando le label.
    """

    labels = {
        record.get("label")
        for record in records
        if record.get("label") is not None
    }

    return len(labels)

def calculate_slots(records: list, threshold: float) -> dict:
    """
    Divide gli score in slot di ampiezza threshold.

    Per ogni PR incrementa il contatore dello slot in cui ricade.

    Restituisce:
    {
        slot_id: numero_PR_nello_slot,
        ...
    }

    Gli slot vuoti intermedi vengono inseriti con valore 0.
    Gli slot vuoti prima del primo slot occupato e dopo l'ultimo non compaiono.
    """

    if not records:
        return {}

    if threshold <= 0:
        raise ValueError("threshold deve essere maggiore di 0")

    valid_scores = []

    for record in records:
        score = record.get("score")

        if score is None:
            continue

        try:
            score = float(score)
        except Exception:
            continue

        if math.isnan(score):
            continue

        valid_scores.append(score)

    if not valid_scores:
        return {}

    min_score = min(valid_scores)

    slots = {}

    for score in valid_scores:
        slot_id = int((score - min_score) / threshold)

        if slot_id not in slots:
            slots[slot_id] = 0

        slots[slot_id] += 1

    first_slot = min(slots.keys())
    last_slot = max(slots.keys())

    full_slots = {}

    for slot_id in range(first_slot, last_slot + 1):
        full_slots[slot_id] = slots.get(slot_id, 0)

    return full_slots


def find_clusters(slots: dict, min_samples: int) -> int:
    """
    Conta quanti slot hanno almeno min_samples PR.

    Ogni slot con conteggio >= min_samples viene considerato cluster.
    """

    clusters = 0

    for slot_id, count in slots.items():
        if count >= min_samples:
            clusters += 1

    return clusters


# ---------------------------------------------------------------------------
# Plot results
# ---------------------------------------------------------------------------
def plot_result(slots: dict, n_clusters: int, n_real_clusters: int, threshold: float, min_samples: int) -> None:
    if not slots:
        print("Nessuno slot da plottare.")
        return

    non_empty_slots = [
        slot_id
        for slot_id, count in slots.items()
        if count > 0
    ]

    if not non_empty_slots:
        print("Tutti gli slot sono vuoti.")
        return

    first_slot = min(non_empty_slots)
    last_slot = max(non_empty_slots)

    slot_ids = list(range(first_slot, last_slot + 1))
    counts = [slots.get(slot_id, 0) for slot_id in slot_ids]

    plt.figure(figsize=(12, 5.5))

    plt.bar(
        slot_ids,
        counts,
        label="Probe Request per intervallo"
    )

    plt.axhline(
        y=min_samples,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Soglia minima = {min_samples} PR"
    )

    plt.title(
        "Distribuzione degli score delle Probe Request",
        fontsize=18,
        fontweight="bold"
    )

    plt.xlabel("Intervallo di score", fontsize=16)
    plt.ylabel("Numero di Probe Request", fontsize=16)

    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)

    info_text = (
        f"Ampiezza intervallo = {threshold}\n"
        f"Soglia minima = {min_samples} PR\n"
        f"Cluster stimati = {n_clusters}\n"
        f"Dispositivi reali = {n_real_clusters}"
    )

    plt.text(
        0.98,
        0.95,
        info_text,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        fontsize=13,
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            alpha=0.9
        )
    )

    plt.legend(
        loc="upper left",
        fontsize=13
    )

    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()