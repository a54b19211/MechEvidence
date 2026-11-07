# 0001 Synchronous service, two small model calls, program-assigned ids

Status: accepted (2026-11)

## Context

The previous acceptance failed in the local review stage (`model_output_invalid`): the old schema made the model keep its own ids and character offsets, and one wrong id voided the whole answer.

## Decision

上一轮验收在 local review 阶段报 `model_output_invalid`：旧 schema 要求模型自己维护 C/J/L/M 观察编号、`matL:P003` 片段编号、带绝对字符偏移的 source_notes，错一个整份作废；服务还把模型请求交回雷达代调、再回放回执。现在：

- 服务自己调模型（`llm.py`），一次请求就是一次同步 HTTP 调用；删掉反向协议、回执、版本哈希和 capabilities。
- 两次调用、两份小 schema；编号全部由程序分配，坏的一条只丢一条；思考关闭，`max_tokens` 4096，校验失败带错误重试一次。
- 旧的 local review 与 mechanism review 合并为一次审阅；删掉 source_notes 偏移锚点层、组合（combinations）、两两比较（comparisons）、whole_assessment、direct_baseline 和消融开关、cohorts 独立性评审、Fabric 读取命令和 evaluation/ 脚本。
- 原有的确定性评级规则保留，见 README「怎么评」。

## Consequences

The deterministic grading rules are unchanged (README, "怎么评"). Known limits are listed in the README.
