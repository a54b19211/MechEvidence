"""同一真实材料：完整方法、直接模型、复用局部判断的删项比较。"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
from pathlib import Path

from mechevidence import aggregate, direct_baseline, evaluate
from mechevidence.model import DeepSeek
from .cases import build_cases

PROJECT = Path(__file__).resolve().parents[1]


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))


def compare(case: dict, output: Path, baseline: bool = True, reuse_from: Path | None = None) -> dict:
    folder = output / case["id"]
    folder.mkdir()
    write_json(folder / "case.json", case)
    row = {"id": case["id"], "question": case["question"], "methods": {}}
    prior = reuse_from / case["id"] if reuse_from else None
    if prior and json.loads((prior / "case.json").read_text()) != case:
        raise ValueError(f"{case['id']}: 材料或问题已变化，不能复用旧判断")
    for method_name, method in [("full", evaluate)] + ([("direct", direct_baseline)] if baseline else []):
        call = DeepSeek(PROJECT, folder / method_name)
        try:
            reused = []

            def call_with_saved_local(stage, instructions, payload, schema):
                if stage == "local_review" and prior:
                    saved_path = prior / "full/call-01.json"
                    saved = json.loads(saved_path.read_text())
                    if saved["stage"] != "local_review" or saved["status"] != "completed" or saved["model"] != call.model:
                        raise ValueError("旧调用不是成功的同模型局部审核，不能复用")
                    local = json.loads(saved["content"])
                    ids = [obs["id"] for obs in local["observations"]]
                    if len(ids) != len(set(ids)):
                        raise ValueError("旧局部审核内部编号重复，不能改号掩盖问题")
                    # 仅迁移阶段专用编号；保留每项科学判断和原文引用。
                    for number, observation in enumerate(local["observations"], 1):
                        observation["id"] = f"L{number}"
                    reused.append(str(saved_path))
                    write_json(folder / "full/reused-local.json", {"from": str(saved_path), "local_review": local})
                    return local
                return call(stage, instructions, payload, schema)

            if method_name == "direct" and prior and (prior / "direct.json").exists():
                result = json.loads((prior / "direct.json").read_text())
                reused.append(str(prior / "direct.json"))
            else:
                result = method(case, call_with_saved_local)
            write_json(folder / f"{method_name}.json", result)
            row["methods"][method_name] = {"status": "completed", "verdict": result.get("overall", result)["verdict"]}
            if reused:
                row["methods"][method_name]["reused_from"] = reused
            if method_name == "full":
                structured = case | result["decomposition"]
                for name, options in [("without_joins", {"use_joins": False}),
                                      ("without_context", {"use_context": False})]:
                    raw = result["source_reporting"]["raw_reviews"]
                    changed = aggregate(structured, raw["local_review"], raw["mechanism_review"], **options)
                    write_json(folder / f"{name}.json", changed)
                    row["methods"][name] = {"status": "completed", "verdict": changed["overall"]["verdict"]}
                claims = result["claims"]
                # 这是局部支持比例，不是可信度。仅用来展示局部都对、整链仍缺证明的情况。
                row["local_supported_fraction"] = sum(c["verdict"] == "supported" for c in claims) / len(claims) if claims else None
                row["all_local_supported"] = bool(claims) and all(c["verdict"] == "supported" for c in claims)
                row["gaps"] = result["critical_gaps"]
        except Exception as exc:
            row["methods"][method_name] = {"status": "failed", "error": str(exc)}
            write_json(folder / f"{method_name}-error.json", row["methods"][method_name])
        row["methods"][method_name]["usage"] = {
            key: sum(c.get("usage", {}).get(key, 0) for c in call.calls)
            for key in ("prompt_tokens", "completion_tokens", "total_tokens", "prompt_cache_hit_tokens")}
        row["methods"][method_name]["seconds"] = sum(c["seconds"] for c in call.calls)
    write_json(folder / "summary.json", row)
    return row


def main():
    parser = argparse.ArgumentParser(description="真实材料研究比较；工程预期与实际预测分别保存")
    parser.add_argument("--ids", nargs="*")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--without-baseline", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reuse-local-from", type=Path, help="明确复用同案例的旧局部判断及已完成直接基线；用于修复聚合或阶段接口后的比较")
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("workers 应在 1–8 之间")
    cases, expectations = build_cases(PROJECT / "data/fabric_records.json", PROJECT / "data/fulltexts")
    write_json(PROJECT / "data/cases.json", cases)
    write_json(PROJECT / "data/expectations.json", expectations)
    if args.build_only:
        print(f"准备 {len(cases)} 个案例；期望值单独保存在 data/expectations.json，不发送给模型")
        return
    if args.ids:
        missing = set(args.ids) - {c["id"] for c in cases}
        if missing:
            parser.error(f"未知案例：{sorted(missing)}")
        cases = [c for c in cases if c["id"] in args.ids]
    output = args.output or PROJECT / "runs" / ("comparison-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    output.mkdir(parents=True, exist_ok=False)
    # 保存本次方法源码和案例，之后改进提示词也不会覆盖旧实验的含义。
    write_json(output / "method.json", {p.name: p.read_text() for p in (PROJECT / "mechevidence").glob("*.py")})
    write_json(output / "expectations.json", {c["id"]: expectations[c["id"]] for c in cases})
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(compare, c, output, not args.without_baseline, args.reuse_local_from): c["id"] for c in cases}
        for future in as_completed(futures):
            row = future.result()
            results.append(row)
            write_json(output / "summary.json", sorted(results, key=lambda r: r["id"]))
            print(json.dumps({"id": row["id"], "methods": row["methods"], "finished": len(results), "total": len(cases)}, ensure_ascii=False), flush=True)
    print(f"完整实验：{output}", flush=True)


if __name__ == "__main__":
    main()
