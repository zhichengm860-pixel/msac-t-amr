#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
清理 experiments 目录中不完整或空的实验结果。

判定标准（可扩展）：
- simple_training_*：需包含顶层的 config.json 和 results.json
- evaluation_*：需包含顶层的 evaluation_results.json
- baseline_comparison_*：需包含顶层的 comparison_results.json
- final_report_*：需包含顶层的 experiment_report.md 和 summary.json
- pretrain_* / pretrain_lite_*：需包含顶层的 results.json（仅保留有结果的）
- amrnet_experiment_*：需包含顶层 results.json 或 results 目录内有文件
- experiments/experiments（嵌套目录）：视为异常目录，清理

清理方式默认移动到 experiments/_trash/<timestamp>/ 下进行“回收站”式存放，
如需永久删除可指定 --mode delete。

支持 --dry-run 进行干跑，仅打印待清理项而不执行。
执行后会在 experiments/_cleanup_logs/ 生成清理摘要日志。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List


def has_file(dir_path: Path, filename: str, top_only: bool = True) -> bool:
    """检查目录中是否存在指定文件。
    默认仅检查顶层；若 top_only=False 则递归检查。
    """
    if top_only:
        return (dir_path / filename).exists()
    for root, _, files in os.walk(dir_path):
        if filename in files:
            return True
    return False


def count_contents(dir_path: Path) -> Dict[str, int]:
    files_count = 0
    dirs_count = 0
    for _, dirs, files in os.walk(dir_path):
        files_count += len(files)
        dirs_count += len(dirs)
    return {"files": files_count, "dirs": dirs_count}


def dir_has_any_files(dir_path: Path) -> bool:
    for _, _, files in os.walk(dir_path):
        if files:
            return True
    return False


def evaluate_experiment_dir(exp_dir: Path) -> Dict:
    name = exp_dir.name
    info = {
        "name": name,
        "path": str(exp_dir),
        "type": "unknown",
        "complete": False,
        "reason": "",
        "counts": count_contents(exp_dir),
    }

    # 快速空目录判定
    if info["counts"]["files"] == 0:
        info["type"] = "empty"
        info["complete"] = False
        info["reason"] = "目录为空，无任何文件"
        return info

    # 类型判定与完整性检查
    if name.startswith("simple_training_"):
        info["type"] = "simple_training"
        need = [has_file(exp_dir, "config.json"), has_file(exp_dir, "results.json")]
        info["complete"] = all(need)
        if not info["complete"]:
            missing = []
            if not need[0]:
                missing.append("config.json")
            if not need[1]:
                missing.append("results.json")
            info["reason"] = f"缺少: {', '.join(missing)}"
        return info

    if name.startswith("evaluation_"):
        info["type"] = "evaluation"
        need_eval = has_file(exp_dir, "evaluation_results.json")
        info["complete"] = need_eval
        if not info["complete"]:
            info["reason"] = "缺少: evaluation_results.json"
        return info

    if name.startswith("baseline_comparison_"):
        info["type"] = "baseline_comparison"
        need_cmp = has_file(exp_dir, "comparison_results.json")
        info["complete"] = need_cmp
        if not info["complete"]:
            info["reason"] = "缺少: comparison_results.json"
        return info

    if name.startswith("final_report_"):
        info["type"] = "final_report"
        need_md = has_file(exp_dir, "experiment_report.md")
        need_sum = has_file(exp_dir, "summary.json")
        info["complete"] = need_md and need_sum
        if not info["complete"]:
            missing = []
            if not need_md:
                missing.append("experiment_report.md")
            if not need_sum:
                missing.append("summary.json")
            info["reason"] = f"缺少: {', '.join(missing)}"
        return info

    if name.startswith("pretrain_lite_") or name.startswith("pretrain_"):
        info["type"] = "pretrain"
        need_res = has_file(exp_dir, "results.json")
        info["complete"] = need_res
        if not info["complete"]:
            info["reason"] = "缺少: results.json"
        return info

    if name.startswith("amrnet_experiment_"):
        info["type"] = "amrnet_experiment"
        has_res_json = has_file(exp_dir, "results.json")
        results_dir = exp_dir / "results"
        results_has_files = results_dir.exists() and dir_has_any_files(results_dir)
        info["complete"] = has_res_json or results_has_files
        if not info["complete"]:
            info["reason"] = "缺少: results.json 或空的 results 目录"
        return info

    if name == "experiments":
        info["type"] = "nested_experiments"
        info["complete"] = False
        info["reason"] = "嵌套的 experiments 目录（异常生成）"
        return info

    # 未识别类型：默认保留，除非判定空
    info["complete"] = True  # 保守不删
    return info


def cleanup(base_dir: Path, mode: str = "trash", dry_run: bool = False) -> Dict:
    assert mode in {"trash", "delete"}
    if not base_dir.exists():
        raise FileNotFoundError(f"不存在目录: {base_dir}")

    children = [p for p in base_dir.iterdir() if p.is_dir()]
    evaluated: List[Dict] = [evaluate_experiment_dir(p) for p in children]
    to_delete = [e for e in evaluated if not e["complete"]]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    trash_root = base_dir / "_trash" / ts
    log_root = base_dir / "_cleanup_logs"
    log_root.mkdir(parents=True, exist_ok=True)

    summary = {
        "timestamp": ts,
        "base_dir": str(base_dir),
        "mode": mode,
        "dry_run": dry_run,
        "found": len(evaluated),
        "to_cleanup": len(to_delete),
        "items": to_delete,
    }

    # 打印计划
    print("=== 实验目录清理计划 ===")
    print(f"总目录数: {len(evaluated)}")
    print(f"待清理数: {len(to_delete)}")
    for e in to_delete:
        print(f" - {e['name']} [{e['type']}] -> {e['reason']} (files: {e['counts']['files']}, dirs: {e['counts']['dirs']})")

    if dry_run:
        print("(干跑模式) 不执行实际清理")
    else:
        if mode == "trash":
            trash_root.mkdir(parents=True, exist_ok=True)
        for e in to_delete:
            src = Path(e["path"])  # type: ignore
            if not src.exists():
                print(f"跳过不存在目录: {src}")
                continue
            try:
                if mode == "trash":
                    dst = trash_root / src.name
                    print(f"移动到回收站: {src} -> {dst}")
                    shutil.move(str(src), str(dst))
                else:
                    print(f"永久删除: {src}")
                    shutil.rmtree(src)
            except Exception as ex:
                print(f"清理失败: {src}，错误: {ex}")
    # 写日志
    log_file = log_root / f"cleanup_{ts}.json"
    try:
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"清理摘要日志: {log_file}")
    except Exception as ex:
        print(f"写入清理日志失败: {ex}")

    return summary


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="清理不完整或空的实验结果目录")
    parser.add_argument(
        "--base-dir",
        type=str,
        default=str(Path(__file__).parent / "experiments"),
        help="experiments 根目录路径",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["trash", "delete"],
        default="trash",
        help="清理方式：移动到回收站(trash)或永久删除(delete)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="干跑，仅打印计划不执行",
    )

    args = parser.parse_args(argv)
    base_dir = Path(args.base_dir)
    try:
        cleanup(base_dir=base_dir, mode=args.mode, dry_run=args.dry_run)
    except Exception as ex:
        print(f"执行清理失败: {ex}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())