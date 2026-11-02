"""python -m mechevidence evaluate case.json —— 不经 HTTP 直接评价一个请求文件（会真实调用模型）。"""
import json
import sys

from .assess import EvaluationFailed, evaluate

if len(sys.argv) != 3 or sys.argv[1] != "evaluate":
    sys.exit("用法：python -m mechevidence evaluate case.json")
with open(sys.argv[2], encoding="utf-8") as handle:
    case = json.load(handle)
try:
    print(json.dumps(evaluate(case), ensure_ascii=False, indent=2))
except EvaluationFailed as error:
    print(json.dumps({"error_code": error.code, "message": str(error), "llm_calls": error.llm_calls},
                     ensure_ascii=False, indent=2))
    sys.exit(1)
