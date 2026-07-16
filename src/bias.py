import numpy as np
import pandas as pd
from scipy import stats


def zscore(values):
    standard_deviation = values.std(ddof=0)
    if not np.isfinite(standard_deviation) or standard_deviation == 0:
        return pd.Series(np.zeros(len(values)), index=values.index, dtype=float)
    return (values - values.mean()) / standard_deviation


def assign_calibration_cases(z_rating, z_review, tau):
    rating_group = np.where(z_rating > tau, "R+", np.where(z_rating < -tau, "R-", "R0"))
    review_group = np.where(z_review > tau, "T+", np.where(z_review < -tau, "T-", "T0"))
    case = pd.Series(rating_group, index=z_rating.index) + pd.Series(review_group, index=z_review.index)
    broad_group = np.where(
        case.isin(["R+T+", "R-T-"]),
        "Aligned",
        np.where(case.isin(["R+T-", "R-T+"]), "Conflict", "Neutral/Other"),
    )
    return pd.DataFrame(
        {
            "rating_bias_group": rating_group,
            "review_bias_group": review_group,
            "calibration_case": case,
            "aligned_conflict_group": broad_group,
        },
        index=z_rating.index,
    )


def compute_user_bias(
    train_sentiment,
    minimum_interactions=5,
    shrinkage_lambda=5.0,
    tau=0.5,
):
    """Estimate both biases using training interactions only.

    Rating residuals remove the training item mean. Review residuals first
    remove the training expectation conditional on rating, then the training
    item's mean residual tone. Both user means receive the same shrinkage.
    """
    frame = train_sentiment.copy()
    sentiment_column = "sentiment_expected_1_5"
    item_rating_mean = frame.groupby("item_id")["rating"].mean()
    rating_sentiment_mean = frame.groupby("rating")[sentiment_column].mean()
    frame["rating_residual"] = frame["rating"] - frame["item_id"].map(item_rating_mean)
    frame["rating_conditioned_tone"] = frame[sentiment_column] - frame["rating"].map(rating_sentiment_mean)
    item_tone_mean = frame.groupby("item_id")["rating_conditioned_tone"].mean()
    frame["review_tone_residual"] = frame["rating_conditioned_tone"] - frame["item_id"].map(item_tone_mean)

    users = (
        frame.groupby("user_id")
        .agg(
            n_train_user=("rating", "size"),
            mean_train_rating=("rating", "mean"),
            mean_train_sentiment=(sentiment_column, "mean"),
            rating_bias_unshrunk=("rating_residual", "mean"),
            review_bias_unshrunk=("review_tone_residual", "mean"),
        )
        .reset_index()
    )
    shrinkage = users["n_train_user"] / (users["n_train_user"] + shrinkage_lambda)
    users["rating_bias_raw"] = users["rating_bias_unshrunk"] * shrinkage
    users["review_bias_raw"] = users["review_bias_unshrunk"] * shrinkage
    users["eligible_user"] = users["n_train_user"] >= minimum_interactions
    users["z_rating_bias"] = np.nan
    users["z_review_bias"] = np.nan
    eligible = users["eligible_user"]
    users.loc[eligible, "z_rating_bias"] = zscore(users.loc[eligible, "rating_bias_raw"])
    users.loc[eligible, "z_review_bias"] = zscore(users.loc[eligible, "review_bias_raw"])
    users = pd.concat(
        [users, assign_calibration_cases(users["z_rating_bias"], users["z_review_bias"], tau)], axis=1
    )
    users["signed_calibration_gap"] = users["z_review_bias"] - users["z_rating_bias"]
    users["abs_calibration_gap"] = users["signed_calibration_gap"].abs()
    users["minimum_training_interactions"] = minimum_interactions
    users["shrinkage_lambda"] = shrinkage_lambda
    users["case_threshold_tau"] = tau
    references = {
        "item_rating_mean": item_rating_mean,
        "rating_sentiment_mean": rating_sentiment_mean,
        "item_tone_mean": item_tone_mean,
        "global_rating_mean": float(frame["rating"].mean()),
        "global_sentiment_mean": float(frame[sentiment_column].mean()),
    }
    return users, references


def attach_heldout_residuals(
    heldout_sentiment, references
):
    frame = heldout_sentiment.copy()
    item_rating_mean = references["item_rating_mean"]
    rating_sentiment_mean = references["rating_sentiment_mean"]
    item_tone_mean = references["item_tone_mean"]
    frame["heldout_rating_residual"] = (
        frame["rating"]
        - frame["item_id"].map(item_rating_mean).fillna(float(references["global_rating_mean"]))
    )
    frame["heldout_review_residual"] = (
        frame["sentiment_expected_1_5"]
        - frame["rating"].map(rating_sentiment_mean).fillna(float(references["global_sentiment_mean"]))
        - frame["item_id"].map(item_tone_mean).fillna(0.0)
    )
    return frame


def spearman(left, right):
    valid = left.notna() & right.notna()
    if valid.sum() < 3 or left[valid].nunique() < 2 or right[valid].nunique() < 2:
        return np.nan
    return float(stats.spearmanr(left[valid], right[valid]).statistic)


def direction_match(left, right):
    valid = left.notna() & right.notna() & left.ne(0) & right.ne(0)
    if valid.sum() == 0:
        return np.nan
    return float((np.sign(left[valid]) == np.sign(right[valid])).mean())


def add_disagreement_fields(frame, epsilon):
    output = frame.copy()
    output["signed_gap"] = output["pred_reviewonly"] - output["pred_neumf"]
    output["m_pred"] = output["signed_gap"].abs()
    output["error_neumf"] = (output["rating"] - output["pred_neumf"]).abs()
    output["error_reviewonly"] = (output["rating"] - output["pred_reviewonly"]).abs()
    output["delta_error"] = output["error_reviewonly"] - output["error_neumf"]
    output["transfer"] = np.where(
        output["delta_error"] < -epsilon,
        "PT",
        np.where(output["delta_error"] > epsilon, "NT", "Neutral"),
    )
    output["error_increase"] = output["delta_error"].clip(lower=0)
    output["oracle_error"] = output[["error_neumf", "error_reviewonly"]].min(axis=1)
    output["m_pred_quartile"] = pd.NA
    ranks = output["m_pred"].rank(method="first")
    output["m_pred_quartile"] = pd.qcut(ranks, 4, labels=["Q1", "Q2", "Q3", "Q4"]).astype(str)
    return output
