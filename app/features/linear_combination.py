"""
linear_combination.py
=====================
Funzioni per il calcolo della combinazione lineare su Probe Request (PR).
Nessuna logica di main: tutto viene chiamato da main.py.
"""

import json
import ast
import itertools
from pathlib import Path
from collections import defaultdict


# ---------------------------------------------------------------------------
# Campi esclusi dal calcolo (metadati / target)
# ---------------------------------------------------------------------------
EXCLUDED_FIELDS = {"start_time", "end_time", "start_seq", "end_seq", "label", "mac"}


# ---------------------------------------------------------------------------
# Caricamento file JSON
# ---------------------------------------------------------------------------

def load_json(path: str) -> list:
    """
    Carica il file JSON delle PR e restituisce sempre una lista di dizionari.

    Gestisce due formati:
      - Lista di dict  : [{"mac": "aa:bb:...", "ie1": ..., "label": 1}, ...]
      - Dict con MAC come chiave : {"aa:bb:...": {"ie1": ..., "label": 1}, ...}
        In questo caso il MAC viene iniettato dentro ogni record come campo "mac".
    """
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict):
        # Il MAC e' la chiave esterna: lo inietto dentro ogni record
        records = []
        for mac, record in data.items():
            entry = {"mac": mac}
            entry.update(record)
            records.append(entry)
        return records
    # Lista: gestisce eventuale doppia serializzazione (elementi stringa)
    if data and isinstance(data[0], str):
        data = [json.loads(item) for item in data]
    return data


def load_weights(path: str) -> dict:
    raw = json.loads(Path(path).read_text())
    # Rimuove chiavi commento (iniziano con '_')
    return {k: v for k, v in raw.items() if not k.startswith("_")}


# ---------------------------------------------------------------------------
# Conversione dei tipi di IE in scalare
# ---------------------------------------------------------------------------

def mac_to_scalar(mac_str: str) -> float:
    """
    Converte 'aa:bb:cc:dd:ee:ff' in un intero a 48 bit normalizzato in [0, 1].
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
    Converte il valore di un campo IE in uno scalare:
      - None / "None"       -> 0.0
      - int / float         -> float(value)
      - stringa lista JSON  -> somma delle componenti
      - lista Python        -> somma delle componenti
      - altra stringa       -> 0.0
    """
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, list):
        return float(sum(value))
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
# Calcolo del punteggio S per una singola PR
# ---------------------------------------------------------------------------

def compute_score(record: dict, weights: dict) -> float:
    """
    Calcola S = sum(alpha_i * x_i) per una PR.

    Il MAC viene gestito separatamente con il peso 'mac'.
    I campi in EXCLUDED_FIELDS vengono ignorati.
    """
    score = 0.0

    # Contributo MAC (peso 0 se randomizzato)
    mac_weight = weights.get("mac", 0.0)
    if mac_weight > 0:
        score += mac_weight * mac_to_scalar(record.get("mac", ""))

    # Contributo IE
    for field, alpha in weights.items():
        if field == "mac" or alpha == 0.0 or field in EXCLUDED_FIELDS:
            continue
        value = record.get(field)
        if value is None:
            continue
        score += alpha * field_to_scalar(value)

    return score


# ---------------------------------------------------------------------------
# Calcolo punteggi per un intero dataset
# ---------------------------------------------------------------------------

def compute_all_scores(data: list, weights: dict) -> list:
    """
    Restituisce lista di dict con 'label', 'mac' e 'score' per ogni PR.
    """
    return [
        {
            "label": pr.get("label"),
            "mac":   pr.get("mac"),
            "score": compute_score(pr, weights),
        }
        for pr in data
    ]


# ---------------------------------------------------------------------------
# Confronto tra coppie di PR
# ---------------------------------------------------------------------------

def same_device(score_a: float, score_b: float, threshold: float) -> bool:
    """True se le due PR ricadono nello stesso intorno (|S_a - S_b| <= threshold)."""
    return abs(score_a - score_b) <= threshold


# ---------------------------------------------------------------------------
# Valutazione (metriche su tutte le coppie)
# ---------------------------------------------------------------------------

def evaluate(records: list, threshold: float) -> dict:
    """
    Confronta tutte le coppie di PR e restituisce un dict con:
      tp, fp, tn, fn, accuracy, precision, recall, f1
    """
    tp = fp = tn = fn = 0
    for i, j in itertools.combinations(range(len(records)), 2):
        a, b = records[i], records[j]
        pred_same = same_device(a["score"], b["score"], threshold)
        real_same = (a["label"] == b["label"])
        if pred_same and real_same:
            tp += 1
        elif pred_same and not real_same:
            fp += 1
        elif not pred_same and not real_same:
            tn += 1
        else:
            fn += 1

    total = tp + fp + tn + fn
    precision = tp / (tp + fp)             if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn)             if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy  = (tp + tn) / total          if total > 0 else 0.0

    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ---------------------------------------------------------------------------
# Statistiche per label
# ---------------------------------------------------------------------------

def stats_by_label(records: list) -> dict:
    """
    Restituisce un dict label -> {n, min, max, mean, range} dei punteggi.
    """
    by_label = defaultdict(list)
    for r in records:
        by_label[r["label"]].append(r["score"])

    result = {}
    for label, scores in by_label.items():
        mn, mx = min(scores), max(scores)
        result[label] = {
            "n":     len(scores),
            "min":   mn,
            "max":   mx,
            "mean":  sum(scores) / len(scores),
            "range": mx - mn,
        }
    return result


# ---------------------------------------------------------------------------
# Clustering per intorno
# ---------------------------------------------------------------------------

def find_clusters(records: list, threshold: float) -> list:
    """
    Raggruppa le PR in cluster in base al punteggio: due PR appartengono
    allo stesso cluster se |S_a - S_b| <= threshold.

    Algoritmo union-find semplice: ogni PR viene assegnata al primo cluster
    il cui centro (media dei punteggi) rientra nell'intorno.
    Restituisce una lista di cluster_id (uno per ogni record, stesso ordine).
    """
    cluster_ids = [-1] * len(records)
    cluster_scores = []   # lista di liste di score per ogni cluster

    for i, r in enumerate(records):
        assigned = False
        for cid, scores in enumerate(cluster_scores):
            center = sum(scores) / len(scores)
            if abs(r["score"] - center) <= threshold:
                cluster_ids[i] = cid
                scores.append(r["score"])
                assigned = True
                break
        if not assigned:
            cluster_ids[i] = len(cluster_scores)
            cluster_scores.append([r["score"]])

    return cluster_ids


# ---------------------------------------------------------------------------
# Stampa report
# ---------------------------------------------------------------------------

def print_report(records: list, metrics: dict, threshold: float) -> None:
    cluster_ids = find_clusters(records, threshold)
    n_labels   = len(set(r["label"] for r in records))
    n_clusters = len(set(cluster_ids))

    print(f"\n{'='*62}")
    print(f"  COMBINAZIONE LINEARE - Risultati")
    print(f"{'='*62}")
    print(f"  Numero PR      : {len(records)}")
    print(f"  Threshold      : {threshold}")
    print(f"  Label reali    : {n_labels}")
    print(f"  Cluster trovati: {n_clusters}")
    print()

    stats = stats_by_label(records)
    print(f"  {'Label':<10} {'N':>4}  {'Min':>12}  {'Max':>12}  {'Mean':>12}  {'Range':>12}")
    print(f"  {'-'*10} {'-'*4}  {'-'*12}  {'-'*12}  {'-'*12}  {'-'*12}")
    for label in sorted(stats):
        s = stats[label]
        print(f"  {label:<10} {s['n']:>4}  {s['min']:>12.4f}  {s['max']:>12.4f}  {s['mean']:>12.4f}  {s['range']:>12.4f}")

    m = metrics
    print(f"\n  Coppie totali : {m['tp']+m['fp']+m['tn']+m['fn']}")
    print(f"  TP={m['tp']}  FP={m['fp']}  TN={m['tn']}  FN={m['fn']}")
    print()
    print(f"  Accuracy  : {m['accuracy']:.4f}")
    print(f"  Precision : {m['precision']:.4f}")
    print(f"  Recall    : {m['recall']:.4f}")
    print(f"  F1-Score  : {m['f1']:.4f}")
    print(f"{'='*62}\n")


def print_scores(data: list, records: list) -> None:
    print(f"\n{'PR':>4}  {'MAC':<20}  {'Label':>6}  {'Score':>14}")
    print(f"{'-'*4}  {'-'*20}  {'-'*6}  {'-'*14}")
    for idx, r in enumerate(records):
        print(f"{idx:>4}  {str(r.get('mac', '')):20}  {r['label']:>6}  {r['score']:>14.6f}")