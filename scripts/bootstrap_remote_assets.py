"""Bootstrap ignored datasets and lightweight stage-2 artifacts on a fresh remote workspace."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.request import urlretrieve
import zipfile


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core_mem.v2.datasets import load_stage2_source_config, summarize_stage2_sources
from scripts.normalize_stage2_public_data import normalize_public_datasets
from scripts.prepare_stage2_data import prepare_stage2_data


PERSONAMEM_32K = {
    "questions": "https://huggingface.co/datasets/bowen-upenn/PersonaMem/resolve/main/questions_32k.csv",
    "contexts": "https://huggingface.co/datasets/bowen-upenn/PersonaMem/resolve/main/shared_contexts_32k.jsonl",
}

LONGMEMEVAL_S = {
    "dataset": "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json",
}

STAGE2_PUBLIC = {
    "sgd_repo_zip": "https://codeload.github.com/google-research-datasets/dstc8-schema-guided-dialogue/zip/refs/heads/master",
    "multiwoz24_zip": "https://raw.githubusercontent.com/smartyfh/MultiWOZ2.4/main/data/MULTIWOZ2.4.zip",
    "personachat_tgz": "https://parl.ai/downloads/convai2/convai2_fix_723.tgz",
    "mquake_cf_3k_v2": "https://raw.githubusercontent.com/princeton-nlp/MQuAKE/main/datasets/MQuAKE-CF-3k-v2.json",
    "recoe_repo": "https://github.com/awslabs/ReCoE.git",
}


def _download(url: str, target: Path) -> dict[str, object]:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0:
        return {"path": str(target), "downloaded": False, "bytes": target.stat().st_size}
    with tempfile.NamedTemporaryFile(delete=False, dir=str(target.parent)) as handle:
        tmp_path = Path(handle.name)
    try:
        urlretrieve(url, tmp_path)
        tmp_path.replace(target)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return {"path": str(target), "downloaded": True, "bytes": target.stat().st_size}


def _extract_sgd_repo(zip_path: Path, target_root: Path) -> dict[str, object]:
    train_dir = target_root / "train"
    if train_dir.exists():
        dialogue_count = len(list(train_dir.glob("dialogues_*.json")))
        return {"path": str(target_root), "extracted": False, "dialogue_files": dialogue_count}
    target_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=str(target_root.parent)) as tmp_dir:
        tmp_root = Path(tmp_dir)
        with zipfile.ZipFile(zip_path) as bundle:
            bundle.extractall(tmp_root)
        extracted_root = tmp_root / "dstc8-schema-guided-dialogue-master"
        if target_root.exists():
            shutil.rmtree(target_root)
        shutil.move(str(extracted_root), str(target_root))
    dialogue_count = len(list((target_root / "train").glob("dialogues_*.json")))
    return {"path": str(target_root), "extracted": True, "dialogue_files": dialogue_count}


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _clone_repo(url: str, target_root: Path) -> dict[str, object]:
    if target_root.exists():
        return {"path": str(target_root), "cloned": False}
    target_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=str(target_root.parent)) as tmp_dir:
        tmp_root = Path(tmp_dir) / "repo"
        subprocess.run(
            ["git", "clone", "--depth", "1", url, str(tmp_root)],
            check=True,
            cwd=REPO_ROOT,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        )
        shutil.move(str(tmp_root), str(target_root))
    return {"path": str(target_root), "cloned": True}


def _build_recovered_source_config(
    default_config_path: Path,
    output_path: Path,
) -> dict[str, object]:
    payload = load_stage2_source_config(default_config_path)
    datasets = payload.get("datasets", {})
    for dataset_key, spec in datasets.items():
        dataset_path = REPO_ROOT / spec["path"]
        spec["enabled"] = dataset_path.exists()
    _write_json(output_path, payload)
    return payload


def _run_smoke_prepare(source_config_path: Path, output_root: Path, max_rows_per_dataset: int | None) -> dict[str, object]:
    return prepare_stage2_data(
        output_root=output_root,
        source_config=source_config_path,
        strict_sources=True,
        max_rows_per_dataset=max_rows_per_dataset,
    )


def bootstrap_assets(
    *,
    skip_benchmarks: bool,
    skip_stage2_public: bool,
    recoe_root: Path | None,
    prepare_stage2: bool,
    output_root: Path,
    max_rows_per_dataset: int | None,
) -> dict[str, object]:
    report: dict[str, object] = {
        "benchmarks": {},
        "stage2_public_raw": {},
        "stage2_public_normalized": {},
        "configs": {},
        "prepared": None,
    }

    if not skip_benchmarks:
        report["benchmarks"] = {
            "personamem_questions_32k": _download(
                PERSONAMEM_32K["questions"],
                REPO_ROOT / "data" / "personamem" / "questions_32k.csv",
            ),
            "personamem_contexts_32k": _download(
                PERSONAMEM_32K["contexts"],
                REPO_ROOT / "data" / "personamem" / "shared_contexts_32k.jsonl",
            ),
            "longmemeval_s_cleaned": _download(
                LONGMEMEVAL_S["dataset"],
                REPO_ROOT / "data" / "longmemeval" / "longmemeval_s_cleaned.json",
            ),
        }

    available_datasets: list[str] = []
    if not skip_stage2_public:
        sgd_zip = REPO_ROOT / "data" / "stage2_public" / "sgd" / "raw" / "dstc8-schema-guided-dialogue-master.zip"
        multiwoz_zip = REPO_ROOT / "data" / "stage2_public" / "multiwoz24" / "raw" / "MultiWOZ2.4" / "data" / "MULTIWOZ2.4.zip"
        personachat_tgz = REPO_ROOT / "data" / "stage2_public" / "personachat" / "raw" / "convai2_fix_723.tgz"
        mquake_json = REPO_ROOT / "data" / "stage2_public" / "mquake" / "raw" / "MQuAKE" / "datasets" / "MQuAKE-CF-3k-v2.json"

        report["stage2_public_raw"] = {
            "sgd_archive": _download(STAGE2_PUBLIC["sgd_repo_zip"], sgd_zip),
            "sgd_extracted": _extract_sgd_repo(
                sgd_zip,
                REPO_ROOT / "data" / "stage2_public" / "sgd" / "raw" / "dstc8-schema-guided-dialogue",
            ),
            "multiwoz24_zip": _download(STAGE2_PUBLIC["multiwoz24_zip"], multiwoz_zip),
            "personachat_tgz": _download(STAGE2_PUBLIC["personachat_tgz"], personachat_tgz),
            "mquake_cf_3k_v2": _download(STAGE2_PUBLIC["mquake_cf_3k_v2"], mquake_json),
        }

        available_datasets.extend(["sgd", "multiwoz24", "personachat", "mquake"])

        target_root = REPO_ROOT / "data" / "stage2_public" / "recoe" / "raw" / "ReCoE"
        if recoe_root is not None:
            if not target_root.exists():
                shutil.copytree(recoe_root, target_root)
            available_datasets.append("recoe")
            report["stage2_public_raw"]["recoe"] = {"path": str(target_root), "provided": True}
        else:
            report["stage2_public_raw"]["recoe"] = _clone_repo(STAGE2_PUBLIC["recoe_repo"], target_root)
            available_datasets.append("recoe")

        report["stage2_public_normalized"] = normalize_public_datasets(
            source_root=REPO_ROOT / "data" / "stage2_public",
            datasets=available_datasets,
            max_rows_per_dataset=None,
        )

    recovered_config_path = REPO_ROOT / "configs" / "stage2_data_sources.recovered.json"
    _build_recovered_source_config(
        REPO_ROOT / "configs" / "stage2_data_sources.json",
        recovered_config_path,
    )
    report["configs"] = {
        "default": str(REPO_ROOT / "configs" / "stage2_data_sources.json"),
        "recovered": str(recovered_config_path),
        "recovered_summary": summarize_stage2_sources(recovered_config_path),
        "default_summary": summarize_stage2_sources(REPO_ROOT / "configs" / "stage2_data_sources.json"),
    }

    if prepare_stage2:
        report["prepared"] = _run_smoke_prepare(
            recovered_config_path,
            output_root=output_root,
            max_rows_per_dataset=max_rows_per_dataset,
        )

    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-benchmarks", action="store_true")
    parser.add_argument("--skip-stage2-public", action="store_true")
    parser.add_argument("--recoe-root")
    parser.add_argument("--no-prepare-stage2", action="store_true")
    parser.add_argument("--output-root", default="outputs_v2")
    parser.add_argument("--max-rows-per-dataset", type=int, default=64)
    parser.add_argument("--report-path", default="outputs_v2/artifacts/remote_bootstrap_report.json")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    payload = bootstrap_assets(
        skip_benchmarks=args.skip_benchmarks,
        skip_stage2_public=args.skip_stage2_public,
        recoe_root=Path(args.recoe_root).resolve() if args.recoe_root else None,
        prepare_stage2=not args.no_prepare_stage2,
        output_root=Path(args.output_root),
        max_rows_per_dataset=args.max_rows_per_dataset,
    )
    _write_json(REPO_ROOT / args.report_path, payload)
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"report_path={REPO_ROOT / args.report_path}")
        print(f"recovered_source_config={payload['configs']['recovered']}")
        if payload["prepared"]:
            print(f"prepared_manifest={payload['prepared']['prepared_manifest']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
