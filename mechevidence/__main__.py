"""终端入口；雷达直接调用 evaluate(case, call_json) 即可。"""
import argparse
import json
from datetime import datetime
from pathlib import Path

from . import direct_baseline, evaluate
from .fabric import fetch_records
from .model import DeepSeek

PROJECT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description="机制证据评价：读 Fabric 材料，评价具体机制，返回证据和缺口")
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch", help="读取一页真实 Fabric 记录，保留覆盖信息和游标")
    fetch.add_argument("--terms", nargs="*", default=[])
    fetch.add_argument("--ids", nargs="*", default=[])
    fetch.add_argument("--source", default="europe_pmc")
    fetch.add_argument("--limit", type=int, default=50)
    fetch.add_argument("--cutoff")
    fetch.add_argument("--cursor")
    fetch.add_argument("--output", type=Path, required=True)
    run = commands.add_parser("evaluate", help="评价一个 Case JSON 文件")
    run.add_argument("case", type=Path)
    run.add_argument("--baseline", action="store_true")
    run.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "fetch":
        if args.output.exists():
            parser.error("输出文件已存在，请换一个文件名，保留先前读取的版本")
        response = fetch_records(args.terms, args.ids, args.limit, args.source, args.cutoff, args.cursor)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(response, ensure_ascii=False, indent=2))
        print(f"读取 {len(response.get('items', []))} 条；下一页游标：{response.get('next_cursor')}；已保存 {args.output}")
        return
    case = json.loads(args.case.read_text())
    output = args.output or PROJECT / "runs" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    if output.exists():
        parser.error("输出目录已存在，请换一个目录，保留之前的实际结果")
    output.mkdir(parents=True)
    (output / "case.json").write_text(json.dumps(case, ensure_ascii=False, indent=2))
    call = DeepSeek(PROJECT, output / "calls")
    method = direct_baseline if args.baseline else evaluate
    try:
        result = method(case, call)
    except Exception as exc:
        (output / "error.json").write_text(json.dumps({"status": "failed", "error": str(exc)}, ensure_ascii=False, indent=2))
        raise SystemExit(f"执行失败：{exc}\n已完成调用保留在 {output}") from None
    (output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result.get("overall", {"verdict": result.get("verdict"), "rationale": result.get("rationale")}), ensure_ascii=False, indent=2))
    print(f"完整结果：{output / 'result.json'}")


if __name__ == "__main__":
    main()
