from features.linear_combination import (
    calculate_slots,
    find_clusters,
    load_json,
    load_weights,
    normalize_json_data,
    calculate_score,
    plot_result,
    count_real_clusters
)

DATA_PATH    = "app/data/probe_requests/scenario_5_burst_features.json"
WEIGHTS_PATH = "app/data/weights/weights.json"

THRESHOLD = 0.1
MIN_SAMPLES = 3

SHOW_SCORES = True


if __name__ == "__main__":
    data = load_json(DATA_PATH)
    weights = load_weights(WEIGHTS_PATH)

    scaled_data = normalize_json_data(data)

    records = calculate_score(scaled_data, weights)

    slots = calculate_slots(records, THRESHOLD)

    found_clusters = find_clusters(slots, MIN_SAMPLES)
    real_clusters = count_real_clusters(records)

    print(f"Numero Probe Request: {len(data)}")
    print(f"Cluster trovati: {found_clusters}")
    print(f"Cluster reali: {real_clusters}")

    plot_result(
        slots,
        found_clusters,
        real_clusters,
        THRESHOLD,
        MIN_SAMPLES
    )