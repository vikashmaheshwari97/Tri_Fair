from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from promptolution.predictors import MarkerBasedPredictor
from src.predictors import PREDICTION_MODES, create_predictor

from scripts._common import set_generation_limit
from src.config.dataset_configs import ALL_DATASETS
from src.config.model_configs import ALL_MODELS
from src.helpers.llm_creation import create_llm
from src.helpers.task_creation import create_test_task


OPTIMIZERS = ["Tri-Fair", "NSGAII-PO-Fair"]
SEEDS = [42, 43, 44]


def safe_name(value: str) -> str:
    return value.replace("-", "_").replace("/", "_")


def normalize_label(text: object) -> str:
    value = str(text).casefold().strip()
    value = re.sub(r"^the answer is\s+", "", value)
    value = re.sub(r"^answer:\s*", "", value)
    value = value.strip(" .,:;`'\"")
    value = value.replace("-", "_")
    value = re.sub(r"\s+", "_", value)
    value = re.sub(r"[^a-z0-9_]", "", value)
    return value


def raw_output_from_sequence(sequence: object, x: str) -> str:
    seq = str(sequence)
    prefix = f"{x}\n"
    if seq.startswith(prefix):
        return seq[len(prefix):].strip()
    return seq.strip()


def choose_best_prompt(eval_df: pd.DataFrame, budget: int) -> pd.Series:
    if "budget_checkpoint" in eval_df.columns:
        stage = eval_df[eval_df["budget_checkpoint"] == budget].copy()
        if stage.empty:
            stage = eval_df.copy()
    else:
        stage = eval_df.copy()

    return stage.sort_values(
        ["test_quality", "test_fairness", "test_cost"],
        ascending=[False, True, True],
    ).iloc[0]


def run_one(args: argparse.Namespace) -> None:
    if args.dataset != "bias_in_bios":
        raise ValueError("This diagnostic is intentionally Bias-in-Bios only.")

    if args.optimizer not in OPTIMIZERS:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")

    if args.seed not in SEEDS:
        raise ValueError(f"Unexpected seed: {args.seed}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_config = ALL_DATASETS[args.dataset]
    model_config = ALL_MODELS[args.model]

    if getattr(args, "logging_dir", ""):
        logging_dir = Path(args.logging_dir).expanduser().resolve()
    else:
        root = Path(args.results_root) / args.model / args.dataset
        run_root = root / args.optimizer / f"seed{args.seed}"
        if args.budget_dir:
            run_root = run_root / f"budget{args.budget}"
        logging_file = run_root / "logging_dir.txt"

        if not logging_file.is_file():
            raise FileNotFoundError(logging_file)

        logging_dir = Path(logging_file.read_text().strip()).expanduser().resolve()

    eval_path = logging_dir / "eval.parquet"

    if not eval_path.is_file():
        raise FileNotFoundError(eval_path)

    eval_df = pd.read_parquet(eval_path)
    best = choose_best_prompt(eval_df, args.budget)

    prompt = str(best["prompt"])
    prompt_id = str(best.get("prompt_id", "unknown"))

    print(f"Dataset:   {args.dataset}", flush=True)
    print(f"Model:     {args.model}", flush=True)
    print(f"Optimizer: {args.optimizer}", flush=True)
    print(f"Seed:      {args.seed}", flush=True)
    print(f"Prompt ID: {prompt_id}", flush=True)
    print(f"eval.parquet test_quality: {float(best['test_quality']):.6f}", flush=True)

    test_task = create_test_task(
        dataset_config=dataset_config,
        eval_strategy="full",
        seed=args.seed,
        manifest_dir=args.manifest_dir,
        regenerate_manifest=False,
    )

    llm = create_llm(model_config=model_config, seed=args.seed)
    set_generation_limit(llm, args.max_output_tokens)
    predictor = create_predictor(
        args.prediction_mode,
        llm,
        test_task.classes,
        dataset=args.dataset,
    )

    df = test_task.df.reset_index(drop=True).copy()
    xs = df["input"].astype(str).tolist()
    y_true = df["target"].astype(str).str.casefold().tolist()

    prompts = [prompt for _ in xs]
    preds, sequences = predictor.predict(prompts=prompts, xs=xs)

    valid_labels = set(str(label).casefold() for label in test_task.classes)

    rows = []
    for i, (row, y, pred, seq) in enumerate(
        zip(df.to_dict("records"), y_true, preds, sequences)
    ):
        pred_text = str(pred).casefold().strip()
        pred_norm = normalize_label(pred_text)
        raw_output = raw_output_from_sequence(seq, str(row["input"]))
        raw_norm = normalize_label(raw_output)

        rows.append(
            {
                "dataset": args.dataset,
                "model": args.model,
                "optimizer": args.optimizer,
                "seed": args.seed,
                "budget": args.budget,
                "prompt_id": prompt_id,
                "row_index": i,
                "manifest_id": row.get("_manifest_id"),
                "source_index": row.get("_source_index"),
                "gender": row.get("gender"),
                "profession_id": row.get("profession_id"),
                "input": row.get("input"),
                "true_label": y,
                "predicted_label": pred_text,
                "predicted_label_normalized": pred_norm,
                "raw_model_output": raw_output,
                "raw_output_normalized": raw_norm,
                "is_valid_predicted_label": pred_text in valid_labels,
                "is_valid_after_normalization": pred_norm in valid_labels,
                "is_correct": pred_text == y,
                "is_correct_after_normalization": pred_norm == y,
            }
        )

    run_df = pd.DataFrame(rows)

    labels = list(test_task.classes)
    y_pred = run_df["predicted_label"].astype(str).tolist()
    y_pred_norm = run_df["predicted_label_normalized"].astype(str).tolist()

    macro_f1_raw = f1_score(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    macro_f1_norm = f1_score(
        y_true, y_pred_norm, labels=labels, average="macro", zero_division=0
    )
    acc_raw = accuracy_score(y_true, y_pred)
    acc_norm = accuracy_score(y_true, y_pred_norm)

    invalid_raw = 1.0 - float(run_df["is_valid_predicted_label"].mean())
    invalid_norm = 1.0 - float(run_df["is_valid_after_normalization"].mean())

    summary = pd.DataFrame(
        [
            {
                "dataset": args.dataset,
                "model": args.model,
                "optimizer": args.optimizer,
                "seed": args.seed,
                "prompt_id": prompt_id,
                "eval_parquet_test_quality": float(best["test_quality"]),
                "recomputed_macro_f1_raw": macro_f1_raw,
                "recomputed_macro_f1_normalized": macro_f1_norm,
                "accuracy_raw": acc_raw,
                "accuracy_normalized": acc_norm,
                "invalid_rate_raw": invalid_raw,
                "invalid_rate_after_normalization": invalid_norm,
                "test_cost": float(best["test_cost"]),
                "test_fairness": float(best["test_fairness"]),
                "n_examples": len(run_df),
            }
        ]
    )

    report = classification_report(
        y_true,
        y_pred_norm,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    per_class_rows = []
    for label in labels:
        metrics = report.get(label, {})
        per_class_rows.append(
            {
                "dataset": args.dataset,
                "model": args.model,
                "optimizer": args.optimizer,
                "seed": args.seed,
                "prompt_id": prompt_id,
                "profession": label,
                "precision": metrics.get("precision", 0.0),
                "recall": metrics.get("recall", 0.0),
                "f1": metrics.get("f1-score", 0.0),
                "support": metrics.get("support", 0.0),
            }
        )

    per_class = pd.DataFrame(per_class_rows)

    cm = pd.DataFrame(
        confusion_matrix(y_true, y_pred_norm, labels=labels),
        index=[f"true__{x}" for x in labels],
        columns=[f"pred__{x}" for x in labels],
    )

    tag = f"{safe_name(args.optimizer)}_seed{args.seed}"

    run_df.to_parquet(out_dir / f"predictions_{tag}.parquet", index=False)
    run_df.to_csv(out_dir / f"predictions_{tag}.csv", index=False)
    summary.to_csv(out_dir / f"summary_{tag}.csv", index=False)
    per_class.to_csv(out_dir / f"per_profession_f1_{tag}.csv", index=False)
    cm.to_csv(out_dir / f"confusion_{tag}.csv")

    print(f"recomputed macro-F1 raw:        {macro_f1_raw:.6f}", flush=True)
    print(f"recomputed macro-F1 normalized: {macro_f1_norm:.6f}", flush=True)
    print(f"accuracy raw:                   {acc_raw:.6f}", flush=True)
    print(f"accuracy normalized:            {acc_norm:.6f}", flush=True)
    print(f"invalid raw:                    {invalid_raw:.3%}", flush=True)
    print(f"invalid after normalization:    {invalid_norm:.3%}", flush=True)
    print(f"Saved outputs to:               {out_dir.resolve()}", flush=True)


def combine_outputs(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)

    pred_files = sorted(out_dir.glob("predictions_*_seed*.parquet"))
    summary_files = sorted(out_dir.glob("summary_*_seed*.csv"))
    per_class_files = sorted(out_dir.glob("per_profession_f1_*_seed*.csv"))

    if not pred_files:
        raise FileNotFoundError(f"No prediction files found in {out_dir}")

    predictions = pd.concat(
        [pd.read_parquet(path) for path in pred_files],
        ignore_index=True,
    )
    summary = pd.concat(
        [pd.read_csv(path) for path in summary_files],
        ignore_index=True,
    )
    per_class = pd.concat(
        [pd.read_csv(path) for path in per_class_files],
        ignore_index=True,
    )

    predictions.to_parquet(
        out_dir / "bias_in_bios_best_prompt_predictions.parquet",
        index=False,
    )
    predictions.to_csv(
        out_dir / "bias_in_bios_best_prompt_predictions.csv",
        index=False,
    )
    summary.to_csv(
        out_dir / "bias_in_bios_best_prompt_prediction_summary.csv",
        index=False,
    )
    per_class.to_csv(
        out_dir / "bias_in_bios_best_prompt_per_profession_f1.csv",
        index=False,
    )

    print("\n=== Combined prediction summary ===")
    print(summary.sort_values(["optimizer", "seed"]).to_string(index=False))

    print("\nMean by optimizer:")
    print(
        summary.groupby("optimizer")[
            [
                "eval_parquet_test_quality",
                "recomputed_macro_f1_raw",
                "recomputed_macro_f1_normalized",
                "accuracy_raw",
                "accuracy_normalized",
                "invalid_rate_raw",
                "invalid_rate_after_normalization",
            ]
        ]
        .mean()
        .to_string()
    )

    print("\nWorst professions by mean F1:")
    print(
        per_class.groupby("profession")["f1"]
        .mean()
        .sort_values()
        .head(15)
        .to_string()
    )

    invalid = predictions[~predictions["is_valid_predicted_label"]]
    print("\nMost common invalid raw predicted labels:")
    if len(invalid):
        print(invalid["predicted_label"].value_counts().head(30).to_string())
    else:
        print("No invalid raw predicted labels.")

    invalid_norm = predictions[~predictions["is_valid_after_normalization"]]
    print("\nMost common invalid labels after normalization:")
    if len(invalid_norm):
        print(invalid_norm["raw_model_output"].value_counts().head(30).to_string())
    else:
        print("No invalid labels after normalization.")

    print("\nCombined outputs written to:")
    print(out_dir.resolve())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="bias_in_bios")
    parser.add_argument("--model", default="qwen-3-30b")
    parser.add_argument("--optimizer", choices=OPTIMIZERS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--budget", type=int, default=1_000_000)
    parser.add_argument("--manifest-dir", default="data/splits")
    parser.add_argument("--max-output-tokens", type=int, default=16)
    parser.add_argument(
        "--prediction-mode",
        choices=PREDICTION_MODES,
        default="marker",
        help="Prediction mechanism for recomputing the best prompt.",
    )
    parser.add_argument("--out-dir", default="analysis/output/bias_prediction_capture")
    parser.add_argument("--results-root", default="results/tri_fair")
    parser.add_argument(
        "--budget-dir",
        action="store_true",
        help="Look for logging_dir.txt under .../seed<seed>/budget<budget>.",
    )
    parser.add_argument(
        "--logging-dir",
        default="",
        help="Direct path to a completed run logging directory. Overrides --results-root.",
    )
    parser.add_argument("--combine-only", action="store_true")
    args = parser.parse_args()

    if args.combine_only:
        combine_outputs(args)
        return

    if args.optimizer is None or args.seed is None:
        raise ValueError("--optimizer and --seed are required unless --combine-only is used")

    run_one(args)


if __name__ == "__main__":
    main()
