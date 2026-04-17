from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run assignment experiments.")
    parser.add_argument("--preset", choices=["smoke", "quick", "full"], default="quick")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def base_cmd() -> list[str]:
    return [sys.executable, "src/train.py"]


def experiment_matrix(preset: str) -> list[tuple[str, list[str]]]:
    if preset == "smoke":
        common = ["--epochs", "1", "--train-size", "256", "--valid-size", "64", "--test-size", "64", "--batch-size", "32", "--d-model", "64", "--d-ff", "128", "--max-len", "48"]
        return [
            ("smoke_basic", common + ["--dataset", "multi30k", "--src-lang", "en", "--tgt-lang", "de", "--tokenizer", "word", "--output-dir", "outputs/smoke_basic"]),
            ("smoke_gqa_moe", common + ["--dataset", "multi30k", "--src-lang", "en", "--tgt-lang", "de", "--tokenizer", "bpe", "--vocab-size", "1000", "--attn-type", "gqa", "--num-kv-heads", "2", "--use-moe", "--output-dir", "outputs/smoke_gqa_moe"]),
        ]

    if preset == "full":
        basic = ["--epochs", "8", "--train-size", "12000", "--valid-size", "1000", "--test-size", "1000", "--batch-size", "96", "--d-model", "192", "--d-ff", "768", "--max-len", "80"]
        complex_ = ["--epochs", "6", "--train-size", "30000", "--valid-size", "1500", "--test-size", "1000", "--batch-size", "96", "--d-model", "192", "--d-ff", "768", "--max-len", "80", "--tokenizer", "bpe", "--vocab-size", "12000"]
    else:
        basic = ["--epochs", "3", "--train-size", "4000", "--valid-size", "500", "--test-size", "300", "--batch-size", "64", "--d-model", "128", "--d-ff", "512", "--max-len", "72"]
        complex_ = ["--epochs", "3", "--train-size", "8000", "--valid-size", "800", "--test-size", "400", "--batch-size", "64", "--d-model", "128", "--d-ff", "512", "--max-len", "72", "--tokenizer", "bpe", "--vocab-size", "8000"]

    return [
        (
            "basic_multi30k_mha",
            basic
            + [
                "--dataset",
                "multi30k",
                "--src-lang",
                "en",
                "--tgt-lang",
                "de",
                "--tokenizer",
                "word",
                "--attn-type",
                "mha",
                "--output-dir",
                "outputs/basic_multi30k_mha",
            ],
        ),
        (
            "complex_opus100_bpe",
            complex_
            + [
                "--dataset",
                "opus100",
                "--dataset-config",
                "en-fr",
                "--src-lang",
                "en",
                "--tgt-lang",
                "fr",
                "--attn-type",
                "mha",
                "--output-dir",
                "outputs/complex_opus100_bpe",
            ],
        ),
        (
            "mqa_multi30k",
            basic
            + [
                "--dataset",
                "multi30k",
                "--src-lang",
                "en",
                "--tgt-lang",
                "de",
                "--tokenizer",
                "word",
                "--attn-type",
                "mqa",
                "--num-kv-heads",
                "1",
                "--output-dir",
                "outputs/mqa_multi30k",
            ],
        ),
        (
            "gqa_moe_multi30k",
            basic
            + [
                "--dataset",
                "multi30k",
                "--src-lang",
                "en",
                "--tgt-lang",
                "de",
                "--tokenizer",
                "word",
                "--attn-type",
                "gqa",
                "--num-kv-heads",
                "2",
                "--use-moe",
                "--num-experts",
                "4",
                "--output-dir",
                "outputs/gqa_moe_multi30k",
            ],
        ),
    ]


def load_metrics(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_summary(metrics: list[dict]) -> None:
    out = Path("outputs/summary.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 实验结果汇总",
        "",
        "| 实验 | 数据集 | 分词 | 注意力 | MoE | 参数量 | Valid BLEU | Test BLEU | 最优Valid Loss | 运行时间 |",
        "|---|---|---|---|---|---:|---:|---:|---:|---|",
    ]
    for m in metrics:
        lines.append(
            "| {name} | {dataset} | {tok} | {attn}/{kv}kv | {moe} | {params:,} | {vbleu:.2f} | {tbleu:.2f} | {loss:.3f} | {runtime} |".format(
                name=Path(m["output_dir"]).name,
                dataset=f"{m['dataset']} {m['src_lang']}-{m['tgt_lang']}",
                tok=m["tokenizer"],
                attn=m["attn_type"].upper(),
                kv=m["num_kv_heads"],
                moe="yes" if m["use_moe"] else "no",
                params=m["parameters"],
                vbleu=m["valid_bleu"],
                tbleu=m["test_bleu"],
                loss=m["best_valid_loss"],
                runtime=m["runtime"],
            )
        )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    metrics: list[dict] = []
    for name, extra in experiment_matrix(args.preset):
        output_dir = Path(extra[extra.index("--output-dir") + 1])
        metrics_path = output_dir / "metrics.json"
        if args.skip_existing and metrics_path.exists():
            print(f"skip {name}: existing metrics")
            metrics.append(load_metrics(metrics_path))
            continue
        print(f"\n=== Running {name} ===")
        cmd = base_cmd() + extra
        subprocess.run(cmd, check=True)
        metrics.append(load_metrics(metrics_path))
    write_summary(metrics)


if __name__ == "__main__":
    main()
