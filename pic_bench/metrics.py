def count_unique_integers(list_of_lists):
    # Flatten the list of lists and create a set of unique integers
    unique_integers = set(num for sublist in list_of_lists for num in sublist)
    return len(unique_integers)


def calculate_f1_at_k(claim_verify_res_dict, k):
    if k <= 0:
        raise ValueError("k must be greater than 0.")
    if not claim_verify_res_dict:
        return 0, 0, 0  # No claims, F1@K is 0

    tot_claims = len(claim_verify_res_dict)
    s_y1 = sum(
        1
        for elem in claim_verify_res_dict
        if elem["verification_result"] == "supported"
    )
    prec_y = s_y1 / tot_claims

    s_y2 = count_unique_integers(
        [elem["supporting_inds"] for elem in claim_verify_res_dict]
    )
    rk_y = min(s_y2 / k, 1)
    # Calculate F1
    if prec_y + rk_y > 0:
        f1_y = 2.0 * (prec_y * rk_y) / (prec_y + rk_y)
    else:
        f1_y = 0
    return prec_y, rk_y, f1_y


def get_num_unsupported(claim_verify_res_dict):
    num_uns = sum(
        1
        for elem in claim_verify_res_dict
        if elem["verification_result"] == "unsupported"
    )
    return num_uns


def calculate_recall_at_k(claim_verify_res_dict, k):
    s_y = count_unique_integers(
        [elem["supporting_inds"] for elem in claim_verify_res_dict]
    )
    return min(s_y / k, 1)


def calculate_scores(data_chunk):
    results = []
    for d in data_chunk:
        claim_verify_res_dict = d.get("claim_verify_res_dict", [])
        k = len(
            d.get(
                "claims",
                d.get("atomic_claims", []),  # legacy
            )
        )
        if k < 1 or not claim_verify_res_dict:
            d["results"] = {
                "f1": 0,
                "precision": 0,
                "recall": 0,
                "num_unsupported": k,
                "unincluded_inds": [],
            }
        else:
            prec, recall, f1 = calculate_f1_at_k(claim_verify_res_dict, k)
            epsilon = 1e-9  # Small tolerance for floating-point comparison
            assert (
                min(prec, recall) - epsilon <= f1 <= max(prec, recall) + epsilon
            ), f"Invalid F1 value: F1={f1}, Precision={prec}, Recall={recall}"
            num_unsupported = get_num_unsupported(claim_verify_res_dict)
            d["results"] = {
                "f1": f1,
                "precision": prec,
                "recall": recall,
                "num_unsupported": num_unsupported,
            }

        results.append(d)
    return results
