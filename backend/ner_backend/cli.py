"""Command line entrypoint for the NER backend."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import BASE_MODEL_NAME, DEFAULT_DATA_ROOT, DEFAULT_WORK_DIR, JUDGE_MODEL_NAME, LABELS, get_dataset_paths


def _path(value: str | None) -> Path | None:
    return Path(value) if value else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Vietnamese medical NER backend")
    sub = parser.add_subparsers(dest="command", required=True)

    describe = sub.add_parser("describe-data", help="Summarize dataset JSONL files")
    describe.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    describe.add_argument("--output-dir", type=Path, default=DEFAULT_WORK_DIR / "dataset_report")
    describe.add_argument("--csv", action="store_true", help="Also save CSV tables")

    train = sub.add_parser("train", help="Train one LoRA adapter")
    train.add_argument("--dataset", choices=sorted(LABELS), required=True)
    train.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    train.add_argument("--output-root", type=Path, default=DEFAULT_WORK_DIR / "ner_adapters")
    train.add_argument("--base-model", default=BASE_MODEL_NAME)
    train.add_argument("--resume-from-checkpoint", default="auto")

    copy_ckpt = sub.add_parser("copy-checkpoint", help="Copy an uploaded checkpoint into the output adapter dir")
    copy_ckpt.add_argument("--dataset", choices=sorted(LABELS), required=True)
    copy_ckpt.add_argument("--checkpoint-dir", type=Path, required=True)
    copy_ckpt.add_argument("--output-root", type=Path, default=DEFAULT_WORK_DIR / "ner_adapters")

    infer = sub.add_parser("infer", help="Run batch inference using a LoRA adapter")
    infer.add_argument("--dataset", choices=sorted(LABELS), required=True)
    infer.add_argument("--adapter-path", type=Path, required=True)
    infer.add_argument("--input-file", type=Path)
    infer.add_argument("--output-file", type=Path)
    infer.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    infer.add_argument("--base-model", default=BASE_MODEL_NAME)
    infer.add_argument("--max-new-tokens", type=int, default=256)
    infer.add_argument("--limit", type=int)
    infer.add_argument("--load-in-4bit", action="store_true")

    clean = sub.add_parser("clean-eval", help="Clean raw predictions and compute exact/IoU metrics")
    clean.add_argument("--dataset", choices=sorted(LABELS), required=True)
    clean.add_argument("--gold-file", type=Path)
    clean.add_argument("--pred-file", type=Path, required=True)
    clean.add_argument("--output-file", type=Path)
    clean.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    clean.add_argument("--iou-threshold", type=float, default=0.5)

    seqeval = sub.add_parser("seqeval", help="Run seqeval BIO evaluation")
    seqeval.add_argument("--name", required=True)
    seqeval.add_argument("--gold-file", type=Path, required=True)
    seqeval.add_argument("--pred-file", type=Path, required=True)
    seqeval.add_argument("--output-dir", type=Path, default=DEFAULT_WORK_DIR / "seqeval")

    judge = sub.add_parser("judge", help="Run local Qwen LLM-as-a-judge")
    judge.add_argument("--gold-file", type=Path, required=True)
    judge.add_argument("--pred-file", type=Path, required=True)
    judge.add_argument("--output-file", type=Path, required=True)
    judge.add_argument("--model-name", default=JUDGE_MODEL_NAME)
    judge.add_argument("--max-samples", type=int)
    judge.add_argument("--no-4bit", action="store_true")

    plot = sub.add_parser("plot-judge", help="Plot one or more judge JSON reports")
    plot.add_argument("--reports", type=Path, nargs="+", required=True)
    plot.add_argument("--output-file", type=Path)

    zip_cmd = sub.add_parser("zip", help="Zip an artifact directory")
    zip_cmd.add_argument("--src-dir", type=Path, required=True)
    zip_cmd.add_argument("--zip-path", type=Path, required=True)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "describe-data":
        from .describe_data import describe_datasets

        describe_datasets(args.data_root, output_dir=args.output_dir, csv=args.csv)
        return 0

    if args.command == "train":
        from .training import train_one_adapter

        train_one_adapter(
            dataset_name=args.dataset,
            data_root=args.data_root,
            output_root=args.output_root,
            base_model_name=args.base_model,
            resume_from_checkpoint=args.resume_from_checkpoint,
        )
        return 0

    if args.command == "copy-checkpoint":
        from .training import copy_checkpoint_for_resume

        copied = copy_checkpoint_for_resume(args.checkpoint_dir, args.dataset, args.output_root)
        print(f"Copied checkpoint to: {copied}")
        return 0

    if args.command == "infer":
        from .inference import run_inference_file

        paths = get_dataset_paths(args.data_root)
        input_file = args.input_file or paths[args.dataset]["test"]
        output_file = args.output_file or DEFAULT_WORK_DIR / "predictions" / f"{args.dataset}_test_results_full.jsonl"
        out_path = run_inference_file(
            dataset_name=args.dataset,
            input_file=input_file,
            output_file=output_file,
            adapter_path=args.adapter_path,
            base_model_name=args.base_model,
            max_new_tokens=args.max_new_tokens,
            limit=args.limit,
            load_in_4bit=args.load_in_4bit,
        )
        print(f"Saved predictions to: {out_path}")
        return 0

    if args.command == "clean-eval":
        from .postprocess import clean_and_evaluate, format_metrics_report

        paths = get_dataset_paths(args.data_root)
        gold_file = args.gold_file or paths[args.dataset]["test"]
        output_file = args.output_file or DEFAULT_WORK_DIR / "predictions" / f"{args.dataset}_cleaned.jsonl"
        report = clean_and_evaluate(
            dataset_name=args.dataset,
            gold_file=gold_file,
            pred_file=args.pred_file,
            output_file=output_file,
            iou_threshold=args.iou_threshold,
        )
        print(format_metrics_report(report))
        return 0

    if args.command == "seqeval":
        from .seqeval_eval import run_seqeval

        report = run_seqeval(args.gold_file, args.pred_file, args.output_dir, args.name)
        metrics = report["metrics"]
        print(
            f"[{args.name}] used={report['sentences_used']} "
            f"mismatch={report['mismatched_sentences']} "
            f"P={metrics['precision']:.4f} R={metrics['recall']:.4f} F1={metrics['f1']:.4f}"
        )
        return 0

    if args.command == "judge":
        from .judge import run_pipeline_local_judge

        report = run_pipeline_local_judge(
            gold_path=args.gold_file,
            pred_path=args.pred_file,
            output_path=args.output_file,
            model_name=args.model_name,
            max_samples=args.max_samples,
            load_in_4bit=not args.no_4bit,
        )
        print(report["metrics"])
        return 0

    if args.command == "plot-judge":
        from .plots import plot_judge_reports

        plot_judge_reports(args.reports, output_path=args.output_file)
        return 0

    if args.command == "zip":
        from .io_utils import zip_directory

        zip_path = zip_directory(args.src_dir, args.zip_path)
        print(f"Saved zip to: {zip_path}")
        return 0

    raise ValueError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
