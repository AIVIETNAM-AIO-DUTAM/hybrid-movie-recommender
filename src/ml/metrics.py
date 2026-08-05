def precision_at_k(
    predicted: list[int],
    truth: set[int],
    k: int,
) -> float:
    """
    Precision@K = số phim dự đoán đúng / K.
    """

    if k <= 0:
        raise ValueError(
            "k phải lớn hơn 0."
        )

    predicted_at_k = predicted[:k]

    if not predicted_at_k:
        return 0.0

    matched = set(predicted_at_k) & truth

    return len(matched) / k


def recall_at_k(
    predicted: list[int],
    truth: set[int],
    k: int,
) -> float:
    """
    Recall@K = số phim dự đoán đúng /
    tổng số phim positive trong test.
    """

    if k <= 0:
        raise ValueError(
            "k phải lớn hơn 0."
        )

    if not truth:
        return 0.0

    predicted_at_k = set(
        predicted[:k]
    )

    matched = predicted_at_k & truth

    return len(matched) / len(truth)


def hit_rate_at_k(
    predicted: list[int],
    truth: set[int],
    k: int,
) -> float:
    """
    HitRate@K = 1 nếu có ít nhất một phim đúng,
    ngược lại bằng 0.
    """

    if k <= 0:
        raise ValueError(
            "k phải lớn hơn 0."
        )

    predicted_at_k = set(
        predicted[:k]
    )

    matched = predicted_at_k & truth

    return float(bool(matched))