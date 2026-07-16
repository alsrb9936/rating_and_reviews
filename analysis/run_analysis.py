#!/usr/bin/env python3
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bias import add_disagreement_fields, direction_match, spearman
from config import load_config, parse_csv, parse_int_csv


def summarize_seeds(frame, groups, value_columns):
    summary = frame.groupby(groups, dropna=False)[value_columns].agg(["mean", "std"]).reset_index()
    summary.columns = [
        "_".join(str(part) for part in column if part != "") if isinstance(column, tuple) else str(column)
        for column in summary.columns
    ]
    return summary


def load_prediction(config, dataset, seed, split):
    path = config.root / "artifacts" / "predictions" / dataset / f"seed{seed}" / f"{split}.csv"
    return pd.read_csv(path)


def load_derived(config):
    root = config.root / "results" / "derived"
    users = pd.read_csv(root / "user_bias.csv")
    heldout = pd.read_csv(root / "heldout_residuals.csv")
    return users, heldout


def make_interaction_eval(config, datasets, seeds, users):
    epsilon = float(config.raw["bias"]["transfer_epsilon"])
    parts = []
    for dataset in datasets:
        dataset_users = users[users["dataset"] == dataset]
        train_sentiment = pd.read_csv(config.root / "artifacts" / "sentiment" / dataset / "train.csv")
        item_counts = train_sentiment.groupby("item_id").size()
        for seed in seeds:
            prediction = load_prediction(config, dataset, seed, "test")
            prediction["dataset"] = dataset
            prediction["seed"] = seed
            evaluated = add_disagreement_fields(prediction, epsilon)
            evaluated = evaluated.merge(dataset_users, on=["dataset", "user_id"], how="left")
            evaluated["n_train_item"] = evaluated["item_id"].map(item_counts).fillna(0).astype(int)
            parts.append(evaluated)
    return pd.concat(parts, ignore_index=True)


def bias_stability(users, heldout, datasets):
    rows = []
    for dataset in datasets:
        user_frame = users[(users["dataset"] == dataset) & users["eligible_user"]].copy()
        for split in ("val", "test"):
            residuals = heldout[(heldout["dataset"] == dataset) & (heldout["split"] == split)]
            residuals = (
                residuals.groupby("user_id")
                .agg(
                    heldout_rating_residual=("heldout_rating_residual", "mean"),
                    heldout_review_residual=("heldout_review_residual", "mean"),
                )
                .reset_index()
            )
            merged = user_frame.merge(residuals, on="user_id", how="inner")
            rows.append(
                {
                    "dataset": dataset,
                    "heldout_split": split,
                    "eligible_users": merged["user_id"].nunique(),
                    "rating_stability": spearman(merged["rating_bias_raw"], merged["heldout_rating_residual"]),
                    "tone_stability": spearman(merged["review_bias_raw"], merged["heldout_review_residual"]),
                    "rating_tone_cross_correlation": spearman(
                        merged["z_rating_bias"], merged["z_review_bias"]
                    ),
                    "rating_sign_match": direction_match(
                        merged["rating_bias_raw"], merged["heldout_rating_residual"]
                    ),
                    "tone_sign_match": direction_match(
                        merged["review_bias_raw"], merged["heldout_review_residual"]
                    ),
                }
            )
    return pd.DataFrame(rows)


def table1_quartiles(interactions):
    rows = []
    for (dataset, seed, quartile), group in interactions.groupby(
        ["dataset", "seed", "m_pred_quartile"], sort=False
    ):
        rows.append(
            {
                "dataset": dataset,
                "seed": seed,
                "quartile": quartile,
                "delta_mae": group["delta_error"].mean(),
                "pt_rate": (group["transfer"] == "PT").mean(),
                "neutral_rate": (group["transfer"] == "Neutral").mean(),
                "nt_rate": (group["transfer"] == "NT").mean(),
                "n": len(group),
            }
        )
    per_seed = pd.DataFrame(rows)
    summary = summarize_seeds(
        per_seed,
        ["dataset", "quartile"],
        ["delta_mae", "pt_rate", "neutral_rate", "nt_rate", "n"],
    )
    return per_seed, summary


def relation_per_seed(interactions, tau):
    rows = []
    eligible = interactions[interactions["eligible_user"]].copy()
    for (dataset, seed), seed_frame in eligible.groupby(["dataset", "seed"], sort=False):
        for subset in ("all", "Q4"):
            frame = seed_frame if subset == "all" else seed_frame[seed_frame["m_pred_quartile"] == "Q4"]
            user_means = (
                frame.groupby("user_id")
                .agg(
                    signed_gap=("signed_gap", "mean"),
                    m_pred=("m_pred", "mean"),
                    signed_calibration_gap=("signed_calibration_gap", "first"),
                    abs_calibration_gap=("abs_calibration_gap", "first"),
                )
                .reset_index()
            )
            non_neutral = frame[frame["abs_calibration_gap"] > tau]
            rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "subset": subset,
                    "rho_D_G": spearman(user_means["signed_calibration_gap"], user_means["signed_gap"]),
                    "rho_A_M": spearman(user_means["abs_calibration_gap"], user_means["m_pred"]),
                    "direction_match": direction_match(
                        non_neutral["signed_calibration_gap"], non_neutral["signed_gap"]
                    ),
                    "n_users": user_means["user_id"].nunique(),
                    "n_interactions": len(frame),
                }
            )
    return pd.DataFrame(rows)


def regression_by_dataset(interactions):
    rows = []
    eligible = interactions[interactions["eligible_user"]].dropna(
        subset=["signed_gap", "signed_calibration_gap"]
    )
    for dataset, frame in eligible.groupby("dataset", sort=False):
        controls = pd.DataFrame(
            {
                "const": 1.0,
                "log_user_activity": np.log1p(frame["n_train_user"]),
                "log_item_popularity": np.log1p(frame["n_train_item"]),
            },
            index=frame.index,
        )
        controls = pd.concat(
            [
                controls,
                pd.get_dummies(frame["rating"].round().astype(int), prefix="rating", drop_first=True, dtype=float),
                pd.get_dummies(frame["seed"].astype(str), prefix="seed", drop_first=True, dtype=float),
            ],
            axis=1,
        )
        full = pd.concat([controls, frame[["signed_calibration_gap"]]], axis=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            base_fit = sm.OLS(frame["signed_gap"].astype(float), controls.astype(float)).fit(
                cov_type="cluster", cov_kwds={"groups": frame["user_id"]}
            )
            full_fit = sm.OLS(frame["signed_gap"].astype(float), full.astype(float)).fit(
                cov_type="cluster", cov_kwds={"groups": frame["user_id"]}
            )
        coefficient = float(full_fit.params["signed_calibration_gap"])
        standard_error = float(full_fit.bse["signed_calibration_gap"])
        rows.append(
            {
                "dataset": dataset,
                "coefficient_D": coefficient,
                "clustered_standard_error": standard_error,
                "ci_low": coefficient - 1.96 * standard_error,
                "ci_high": coefficient + 1.96 * standard_error,
                "p_value": float(full_fit.pvalues["signed_calibration_gap"]),
                "base_r2": float(base_fit.rsquared),
                "full_r2": float(full_fit.rsquared),
                "delta_r2": float(full_fit.rsquared - base_fit.rsquared),
                "n_interactions": len(frame),
                "n_users": frame["user_id"].nunique(),
            }
        )
    return pd.DataFrame(rows)


def table2(
    stability,
    relations,
    regressions,
):
    validation = stability[stability["heldout_split"] == "val"].copy()
    all_summary = summarize_seeds(
        relations[relations["subset"] == "all"], ["dataset"], ["rho_D_G", "rho_A_M"]
    )
    q4_summary = summarize_seeds(
        relations[relations["subset"] == "Q4"],
        ["dataset"],
        ["rho_D_G", "rho_A_M", "direction_match"],
    )
    return (
        validation.merge(all_summary, on="dataset")
        .merge(q4_summary, on="dataset", suffixes=("_all", "_q4"))
        .merge(regressions, on="dataset")
    )


def table3_crossed_cases(interactions):
    crossed = interactions[
        interactions["eligible_user"]
        & interactions["calibration_case"].isin(["R+T-", "R-T+"])
        & interactions["m_pred_quartile"].eq("Q4")
    ]
    rows = []
    for (dataset, seed, case), group in crossed.groupby(["dataset", "seed", "calibration_case"], sort=False):
        rows.append(
            {
                "dataset": dataset,
                "seed": seed,
                "case": case,
                "mean_signed_gap": group["signed_gap"].mean(),
                "delta_mae": group["delta_error"].mean(),
                "pt_rate": (group["transfer"] == "PT").mean(),
                "nt_rate": (group["transfer"] == "NT").mean(),
                "n_users": group["user_id"].nunique(),
                "n_interactions": len(group),
            }
        )
    per_seed = pd.DataFrame(rows)
    summary = summarize_seeds(
        per_seed,
        ["dataset", "case"],
        ["mean_signed_gap", "delta_mae", "pt_rate", "nt_rate", "n_users", "n_interactions"],
    )
    return per_seed, summary


def main():
    parser = argparse.ArgumentParser(description="Compute calibration and disagreement analysis CSVs.")
    parser.add_argument("--config", default="configs/experiment.json")
    parser.add_argument("--datasets", default="instant,office,digital")
    parser.add_argument("--seeds", default="")
    args = parser.parse_args()

    config = load_config(args.config)
    datasets = parse_csv(args.datasets)
    seeds = parse_int_csv(args.seeds) if args.seeds else config.seeds
    users, heldout = load_derived(config)
    interactions = make_interaction_eval(config, datasets, seeds, users)
    stability = bias_stability(users, heldout, datasets)
    per_seed_table1, summary_table1 = table1_quartiles(interactions)
    relations = relation_per_seed(interactions, float(config.raw["bias"]["case_threshold_tau"]))
    regressions = regression_by_dataset(interactions)
    final_table2 = table2(stability, relations, regressions)
    per_seed_table3, summary_table3 = table3_crossed_cases(interactions)

    output = config.root / "results"
    derived = output / "derived"
    tables = output / "tables"
    derived.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    interactions.to_csv(derived / "interaction_eval.csv", index=False)
    stability.to_csv(tables / "bias_stability.csv", index=False)
    per_seed_table1.to_csv(tables / "table1_quartiles_per_seed.csv", index=False)
    summary_table1.to_csv(tables / "table1_quartiles.csv", index=False)
    relations.to_csv(tables / "table2_relations_per_seed.csv", index=False)
    regressions.to_csv(tables / "table2_regression.csv", index=False)
    final_table2.to_csv(tables / "table2_calibration.csv", index=False)
    per_seed_table3.to_csv(tables / "table3_crossed_cases_per_seed.csv", index=False)
    summary_table3.to_csv(tables / "table3_crossed_cases.csv", index=False)
    print(f"[saved] {derived / 'interaction_eval.csv'} rows={len(interactions)}")
    print(f"[saved] {tables}")


if __name__ == "__main__":
    main()
