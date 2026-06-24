"""
main.py
=======
Punto di ingresso per la Combinazione Lineare su Probe Request.
Modifica i parametri nelle sezioni PARAMETRI e poi esegui questo file.
"""

from features.linear_combination import (
    load_json,
    load_weights,
    compute_all_scores,
    evaluate,
    print_report,
    print_scores,
)

# ============================================================
#   PARAMETRI
# ============================================================

DATA_PATH    = "app/data/probe_requests/scenario_0_burst_features.json"   # file PR (grezzo o preprocessato)
WEIGHTS_PATH = "app/data/weights/weights.json"                     # file pesi

THRESHOLD    = 5.0    # soglia intorno: |S_a - S_b| <= threshold → stesso dispositivo

SHOW_SCORES  = True   # True → stampa il punteggio di ogni singola PR

# ============================================================
#   MAIN
# ============================================================

if __name__ == "__main__":
    data    = load_json(DATA_PATH)
    weights = load_weights(WEIGHTS_PATH)

    records = compute_all_scores(data, weights)

    if SHOW_SCORES:
        print_scores(data, records)

    metrics = evaluate(records, threshold=THRESHOLD)
    print_report(records, metrics, threshold=THRESHOLD)