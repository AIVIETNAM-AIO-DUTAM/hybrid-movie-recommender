"""Evaluation helpers for CF. Owned by ML B (+ QA review)."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def leave_last_out_split(ratings: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-user: latest timestamp → test, rest → train.

    Sort key is (userId, timestamp, movieId) with a stable mergesort so that
    ties on timestamp always resolve to the same movieId regardless of the
    original row order in the DataFrame.

    Users with exactly one rating contribute their single row to TEST only;
    they have no train ratings and therefore become CF cold-start users
    (KeyError inside evaluate()). These users are still counted in the
    HR@10_all denominator — only the narrower HR@10 metric excludes them.
    """
    ordered = ratings.sort_values(
        ["userId", "timestamp", "movieId"], kind="mergesort"
    )
    test = ordered.groupby("userId", as_index=False).tail(1)
    train = ordered.drop(index=test.index)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def hit_rate_at_k(recommended_ids: Iterable[int], truth_id: int, k: int = 10) -> float:
    top = list(recommended_ids)[:k]
    return 1.0 if truth_id in top else 0.0


def ndcg_at_k(recommended_ids: Iterable[int], truth_id: int, k: int = 10) -> float:
    top = list(recommended_ids)[:k]
    if truth_id not in top:
        return 0.0
    rank = top.index(truth_id) + 1
    return 1.0 / np.log2(rank + 1)


def summarize_scores(
    hits: list[float],
    ndcgs: list[float],
    n_total_users: int | None = None,
) -> pd.DataFrame:
    """Summarize HR@10 / NDCG@10.

    `n_total_users` is the size of the eligible sample BEFORE skipping
    cold-start / no-candidate users. When provided, we also report
    `HR@10_all` and `NDCG@10_all` (= hits / n_total_users), which are the
    fairer denominators for an apples-to-apples comparison across sample
    sizes — `HR@10` itself only counts users the model could actually
    score, so it inflates when many cold-start users are present.
    """
    n_evaluated = len(hits)
    hr_evaluated = float(np.mean(hits)) if hits else 0.0
    ndcg_evaluated = float(np.mean(ndcgs)) if ndcgs else 0.0
    row = {
        "users_evaluated": n_evaluated,
        "HR@10": hr_evaluated,
        "NDCG@10": ndcg_evaluated,
    }
    if n_total_users is not None and n_total_users > 0:
        n_hits = sum(1 for h in hits if h > 0)
        row["HR@10_all"] = n_hits / n_total_users
        row["NDCG@10_all"] = sum(ndcgs) / n_total_users
    return pd.DataFrame([row])


def prepare_eval(
    ratings: pd.DataFrame,
) -> tuple:
    """One-time setup for CF evaluation.

    Returns (cf_model, test_df, truth_by_user). Call this once, then run
    `evaluate(...)` many times with different sample/top_k/min_rating without
    rebuilding the CF model (which is the expensive step on MovieLens 25M).

    **CF model is always built on the FULL train split** (all rows except the
    per-user held-out last rating). The `sample_size` parameter on
    `run_evaluation` / `evaluate` only controls how many users are drawn for
    the recommend-and-score loop — it does NOT reduce the training data.
    """
    from recommender_cf import build_cf_model

    train, test = leave_last_out_split(ratings)
    cf = build_cf_model(train)
    truth_by_user = test.set_index("userId")["movieId"].to_dict()
    return cf, test, truth_by_user


def evaluate(
    cf,
    movies: pd.DataFrame,
    truth_by_user: dict,
    eligible_users,
    top_k: int = 10,
    min_rating: float = 4.0,
) -> pd.DataFrame:
    """Sweep-friendly evaluation. Reuses a prebuilt CF model from prepare_eval.

    `eligible_users` can be a sample; pass the full Series to evaluate everyone.

    Reports both per-evaluated-user metrics (HR@10) AND per-all-eligible-users
    metrics (HR@10_all) so cold-start bias is visible. See `summarize_scores`.
    """
    from recommender_cf import recommend_for_user

    n_total = 0
    hits: list[float] = []
    ndcgs: list[float] = []
    n_missing_truth = 0
    n_cold_start = 0
    for user_id in eligible_users:
        n_total += 1
        truth_id = truth_by_user.get(int(user_id))
        if truth_id is None:
            n_missing_truth += 1
            continue
        try:
            recs = recommend_for_user(
                cf, movies, int(user_id), top_k=top_k, min_rating=min_rating
            )
        except (KeyError, ValueError):
            # Unknown user or no liked items / no candidates → counted in the
            # denominator (HR@10_all) but not the numerator. App-level
            # fallback handles these in production.
            n_cold_start += 1
            continue
        rec_ids = recs["movieId"].astype(int).tolist()
        hits.append(hit_rate_at_k(rec_ids, int(truth_id), k=top_k))
        ndcgs.append(ndcg_at_k(rec_ids, int(truth_id), k=top_k))

    summary = summarize_scores(hits, ndcgs, n_total_users=n_total)
    summary.attrs["n_missing_truth"] = n_missing_truth
    summary.attrs["n_cold_start"] = n_cold_start
    return summary


def run_evaluation(
    ratings: pd.DataFrame,
    movies: pd.DataFrame,
    sample_size: int | None = 200,
    top_k: int = 10,
    min_rating: float = 4.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Leave-last-out HR@10 / NDCG@10 for item-based CF (one-shot API).

    Thin wrapper around `prepare_eval` + `evaluate`. For QA sweeps prefer
    calling them directly so CF is built only once.

    Pipeline (per spec §6.3):
      1. time-based split: per user, latest rating -> test, rest -> train
      2. build CF on train
      3. for each sampled user, recommend top_k; check if the held-out
         movieId is in the list

    `sample_size` only samples users in the recommend/eval loop. It does NOT
    shrink `prepare_eval`: leave-last-out + `build_cf_model` still run on the
    FULL train split (memory/time dominate there). Pass None to evaluate every
    eligible user after the full build (slow on MovieLens 25M).
    """
    cf, test, truth_by_user = prepare_eval(ratings)

    # Eligible = users whose held-out movie is in the CF train vocabulary.
    cf_movie_ids = set(cf.movie_ids.tolist())
    eligible_users = test.loc[
        test["movieId"].astype(int).isin(cf_movie_ids),
        "userId",
    ].drop_duplicates()
    if sample_size is not None and len(eligible_users) > sample_size:
        eligible_users = eligible_users.sample(n=sample_size, random_state=seed)

    return evaluate(cf, movies, truth_by_user, eligible_users, top_k, min_rating)
