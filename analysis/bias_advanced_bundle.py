from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bundle Bias advanced run artifacts.")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--model", default="qwen-3-30b")
    parser.add_argument("--dataset", default="bias_in_bios")
    parser.add_argument("--optimizer", default="Tri-Fair")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--budget", type=int, default=250000)
    parser.add_argument("--tier", default="compact")
    parser.add_argument("--results-root", default="")
    parser.add_argument("--prediction-dir", default="")
    parser.add_argument("--bundle-root", default="analysis/output/bias_advanced_bundles")
    parser.add_argument("--make-tar", action="store_true")
    return parser.parse_args()


def run_text(command: list[str]) -> str:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT)
    except Exception as exc:
        return f"Command failed: {' '.join(command)}\n{exc}\n"


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    results_root = args.results_root or f"results/tri_fair_bias_advanced/{args.tier}"
    out_dir = (
        Path(results_root)
        / args.model
        / args.dataset
        / args.optimizer
        / f"seed{args.seed}"
        / f"budget{args.budget}"
    )
    pointer = out_dir / "logging_dir.txt"
    if not pointer.is_file():
        raise FileNotFoundError(pointer)
    logging_dir = Path(pointer.read_text().strip()).expanduser().resolve()
    return out_dir, logging_dir


def safe_copy(src: Path, dst_dir: Path) -> None:
    if src.is_file():
        shutil.copy2(src, dst_dir / src.name)


def write_tables(
    *,
    args: argparse.Namespace,
    out_dir: Path,
    logging_dir: Path,
    bundle_dir: Path,
) -> Path:
    eval_path = logging_dir / "eval.parquet"
    step_path = logging_dir / "step_results.parquet"

    if not eval_path.is_file():
        raise FileNotFoundError(eval_path)
    if not step_path.is_file():
        raise FileNotFoundError(step_path)

    df = pd.read_parquet(eval_path)
    step = pd.read_parquet(step_path)

    best_macro = df.sort_values(
        ["test_quality", "test_fairness", "test_cost"],
        ascending=[False, True, True],
    ).iloc[0]
    best_fair = df.sort_values(
        ["test_fairness", "test_quality", "test_cost"],
        ascending=[True, False, True],
    ).iloc[0]
    best_cost = df.sort_values(
        ["test_cost", "test_quality", "test_fairness"],
        ascending=[True, False, True],
    ).iloc[0]

    top_cols = [
        "chosen_step",
        "test_quality",
        "test_cost",
        "test_fairness",
        "test_fairness_ready",
        "prompt_id",
    ]
    top10 = (
        df.sort_values(
            ["test_quality", "test_fairness", "test_cost"],
            ascending=[False, True, True],
        )[top_cols]
        .head(10)
        .copy()
    )

    token_summary = (
        step.assign(
            step=pd.to_numeric(step["step"], errors="raise").astype(int),
            total_tokens_downstream=pd.to_numeric(
                step["total_tokens_downstream"], errors="raise"
            ).astype(int),
        )
        .groupby("step")["total_tokens_downstream"]
        .max()
        .sort_index()
    )
    reached = token_summary[token_summary >= args.budget]

    summary_rows = []
    for name, row in [
        ("best_macro_f1", best_macro),
        ("fairest", best_fair),
        ("cheapest", best_cost),
    ]:
        summary_rows.append(
            {
                "selection": name,
                "job_id": args.job_id,
                "tier": args.tier,
                "dataset": args.dataset,
                "model": args.model,
                "optimizer": args.optimizer,
                "seed": args.seed,
                "requested_budget": args.budget,
                "chosen_step": row.get("chosen_step"),
                "prompt_id": row.get("prompt_id"),
                "test_macro_f1": row["test_quality"],
                "test_cost": row["test_cost"],
                "test_unfairness": row["test_fairness"],
                "test_fairness_ready": row.get("test_fairness_ready"),
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(bundle_dir / f"bias_advanced_job{args.job_id}_summary.csv", index=False)
    top10.to_csv(bundle_dir / f"bias_advanced_job{args.job_id}_top10.csv", index=False)

    report_path = bundle_dir / f"bias_advanced_job{args.job_id}_report.txt"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("=== Bias advanced run report ===\n")
        f.write(f"job_id: {args.job_id}\n")
        f.write(f"tier: {args.tier}\n")
        f.write(f"model: {args.model}\n")
        f.write(f"dataset: {args.dataset}\n")
        f.write(f"optimizer: {args.optimizer}\n")
        f.write(f"seed: {args.seed}\n")
        f.write(f"requested_budget: {args.budget}\n")
        f.write(f"out_dir: {out_dir}\n")
        f.write(f"logging_dir: {logging_dir}\n")
        f.write(f"eval_path: {eval_path}\n")
        f.write(f"step_path: {step_path}\n")

        f.write("\n=== Eval shape ===\n")
        f.write(f"rows: {len(df)}\n")
        f.write(f"columns: {list(df.columns)}\n")

        f.write("\n=== Summary selections ===\n")
        f.write(summary.to_string(index=False))
        f.write("\n")

        f.write("\n=== Fairness diagnostics for best Macro-F1 ===\n")
        try:
            diag = json.loads(best_macro["test_fairness_diagnostics_json"])
            for key in [
                "metric",
                "rms_tpr_gap",
                "max_abs_tpr_gap",
                "mean_signed_tpr_gap",
                "valid_professions",
                "required_professions",
            ]:
                if key in diag:
                    f.write(f"{key}: {diag[key]}\n")
        except Exception as exc:
            f.write(f"Could not parse diagnostics: {exc}\n")

        f.write("\n=== Top 10 by Macro-F1 ===\n")
        f.write(top10.to_string(index=False))
        f.write("\n")

        f.write("\n=== Token progress ===\n")
        f.write(f"last_step: {int(token_summary.index.max())}\n")
        f.write(f"max_downstream_tokens: {int(token_summary.max())}\n")
        if len(reached):
            f.write(f"first_step_reaching_budget: {int(reached.index[0])}\n")
            f.write(f"actual_tokens_at_budget: {int(reached.iloc[0])}\n")
        else:
            f.write("budget_not_reached: true\n")

        f.write("\n=== Best prompt text start ===\n")
        f.write(str(best_macro["prompt"])[:6000])
        f.write("\n")

    return report_path


def should_skip(path: Path) -> bool:
    blocked_suffixes = {".pkl", ".pt", ".safetensors"}
    if path.suffix in blocked_suffixes:
        return True
    if "checkpoints" in path.parts:
        return True
    return False


def add_tree(tar: tarfile.TarFile, path: Path, arcname: str) -> None:
    if not path.exists():
        return
    if path.is_file():
        if not should_skip(path):
            tar.add(path, arcname=arcname)
        return

    for item in path.rglob("*"):
        if should_skip(item):
            continue
        rel = item.relative_to(path)
        tar.add(item, arcname=str(Path(arcname) / rel))


def main() -> None:
    args = parse_args()
    out_dir, logging_dir = resolve_paths(args)

    bundle_root = Path(args.bundle_root)
    bundle_dir = bundle_root / f"job_{args.job_id}_{args.tier}"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    (bundle_dir / f"slurm_sacct_{args.job_id}.txt").write_text(
        run_text(
            [
                "sacct",
                "-j",
                args.job_id,
                "--format=JobID,JobName%30,State,ExitCode,Elapsed,NodeList%30",
            ]
        ),
        encoding="utf-8",
    )

    safe_copy(Path(f"logs/bias_advanced_{args.job_id}.out"), bundle_dir)
    safe_copy(Path(f"logs/bias_advanced_{args.job_id}.err"), bundle_dir)

    if args.prediction_dir:
        pred_dir = Path(args.prediction_dir)
        if pred_dir.exists():
            pred_copy = bundle_dir / "prediction_capture"
            pred_copy.mkdir(exist_ok=True)
            for item in pred_dir.glob("*"):
                if item.is_file():
                    shutil.copy2(item, pred_copy / item.name)

    report = write_tables(
        args=args,
        out_dir=out_dir,
        logging_dir=logging_dir,
        bundle_dir=bundle_dir,
    )

    print(report)
    print(report.read_text(encoding="utf-8"))

    if args.make_tar:
        tar_path = bundle_root / f"bias_advanced_job{args.job_id}_{args.tier}_bundle.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            add_tree(tar, bundle_dir, str(bundle_dir))
            add_tree(tar, out_dir, str(out_dir))

        checksum = run_text(["sha256sum", str(tar_path)])
        (tar_path.with_suffix(tar_path.suffix + ".sha256")).write_text(
            checksum,
            encoding="utf-8",
        )
        print(f"\nCreated bundle: {tar_path}")
        print(checksum.strip())


if __name__ == "__main__":
    main()
