# MechEvidence

机制证据评价服务，维护人：**汪鑫怡**。

给定一个问题、一个候选机制（一句话）、目标范围（物种 / 系统 / 疾病）和雷达取到的材料，判断这些材料在目标范围内支持机制的哪些部分，指出反证、冲突和缺口。模型只做“这段原文说了什么”的语义判断；是否算证据、是否适用、是否冲突、整体结论，全部由程序按固定规则决定。

## 运行

CPython 3.12，依赖只从锁文件装（见下文「开发环境与测试」）。

```sh
make venv                       # .venv，只按 requirements.lock 安装，每个包都校验 sha256
. .venv/bin/activate
export SERVICE_TOKEN=换成私有令牌
export LLM_API_KEY=...          # 放在私有 .env 或部署环境里，不进仓库
uvicorn mechevidence.service:app --host 127.0.0.1 --port 18561
```

| 环境变量 | 说明 |
|---|---|
| `SERVICE_TOKEN` | 必填。请求头 `Authorization: Bearer <token>`；未设置时服务拒绝所有评价请求 |
| `LLM_BASE_URL` | OpenAI 兼容地址，默认 `https://api.deepseek.com` |
| `LLM_API_KEY` | 模型密钥 |
| `LLM_MODEL` | 默认 `deepseek-v4-pro` |
| `LLM_PROVIDER` | `deepseek`（发送 `thinking: disabled`）或 `openai` |
| `LLM_TIMEOUT_SECONDS` | 默认 180；超时不自动重发 |

不经 HTTP 直接跑一个请求文件（会真实调用模型、产生费用）：`python -m mechevidence evaluate examples/question.json`。

## 接口

- `GET /health` → `{"status": "ok"}`（不需要令牌）
- `POST /v1/evaluate`
- 出错时统一返回 `{"error_code": "...", "message": "..."}`：`unauthorized`(401)、`input_invalid`(422)、`decomposition_failed`(502，附 `llm_calls`)、`service_not_configured`(503)、`not_found`(404)。

### 请求

```json
{
  "question": "在培养的小鼠发育期海马锥体神经元中，哪些证据支持 ERK 信号参与介导 APRIL 的促轴突延长作用？",
  "mechanism": "APRIL 激活 ERK1/ERK2，ERK 信号分支参与介导 APRIL 诱导的轴突延长。",
  "scope": {"species": "mouse", "system": "cultured developing hippocampal pyramidal neurons", "disease": null},
  "materials": [
    {"id": "E1", "title": "Selective regulation of axonal growth ... APRIL.", "text": "（摘要原文）",
     "source_kind": "primary", "document_type": "research_article", "text_level": "abstract",
     "pmid": "24444792", "source_id": "europe_pmc", "record_id": "MED:24444792"}
  ],
  "max_llm_calls": 20
}
```

- `source_kind`：`primary | review | graph`，必须按原始记录给出，本服务不推断。
- `document_type`（可选）：`research_article | conference_abstract | trial_results | trial_registration | study_protocol | review | graph | press_release | company_disclosure | news | social_post`。
- `text_level`：`full | abstract | excerpt`。
- `pmid / doi / url / source_id / record_id / source_group`（可选）只用于把同一篇文献的多份记录合并为一个来源组。
- 已有拆解时可直接带 `claims: [{"text", "kind"}]` 和 `joins: [{"text", "from_claim", "to_claim"}]`，省掉第一次模型调用。

完整例子见 [examples/question.json](examples/question.json)。

### 响应（节选）

```json
{
  "overall": "supported",
  "rationale": "存在覆盖全部义务且背景一致的有效见证。",
  "claims": [
    {"id": "C1", "text": "APRIL 激活培养小鼠海马锥体神经元中的 ERK1/ERK2", "kind": "causal", "verdict": "supported",
     "supporting": ["O1"], "opposing": [], "compatible": [], "applicability_gaps": [], "source_groups": ["pmid:24444792"]},
    {"id": "C2", "text": "APRIL 促进这些神经元的轴突延长", "kind": "causal", "verdict": "supported", "supporting": ["O2"], "...": "..."}
  ],
  "joins": [
    {"id": "J1", "text": "ERK 信号参与介导 APRIL 的促轴突延长作用", "from": "C1", "to": "C2", "verdict": "supported", "supporting": ["O3"], "...": "..."}
  ],
  "observations": [
    {"id": "O3", "targets": ["J1"], "material": "E1", "passage": "E1.9",
     "quote": "Inhibition of MEK1/MEK2 (activators of ERK1/ERK2), PI3-kinase (activator of Akt) or Akt inhibited the axon growth-promoting action of APRIL",
     "relation": "supports", "basis": "intervention", "causal_test": true,
     "context": {"species": "mouse", "system": "cultured hippocampal pyramidal neurons", "disease": null},
     "scope_match": {"species": "matched", "system": "matched", "disease": "not_required"},
     "scope_status": "matched", "causal_evidence": true, "source_group": "pmid:24444792"}
  ],
  "conflicts": [],
  "unresolved_counterevidence": [],
  "gaps": [],
  "independence": {"distinct_source_groups": 1, "note": "来源去重不等于确认队列或实验相互独立。"},
  "material_notes": [
    {"material": "E1", "code": "quote_not_found", "message": "E1.10 中找不到短引：..."}
  ],
  "assessability": {"status": "limited", "limitations": ["摘要报告的实验缺少完整方法和结果细节。"]},
  "llm_calls": [
    {"stage": "decompose", "model": "deepseek-v4-pro", "finish_reason": "stop", "usage": {"total_tokens": 812}, "seconds": 3.1, "error": null, "requests": 1},
    {"stage": "review", "model": "deepseek-v4-pro", "finish_reason": "stop", "usage": {"total_tokens": 2950}, "seconds": 9.4, "error": null, "requests": 2}
  ]
}
```

`llm_calls` 每项是一次逻辑调用；`requests` 是这次调用实际发出的 HTTP 请求数（格式校验失败后的那次重试也算一次，未配置模型时为 0），调用方统计请求量时应按它累计，而不是按条数。

## 怎么评

1. **拆解**（模型调用 ①）：把机制拆成原子命题和因果连接。模型只给文本和连接两端的序号，程序编号 `C1..Cn`、`J1..Jm`。
2. **审阅**（模型调用 ②，每批 ≤3 份材料）：程序把每份材料按句切成片段 `E1.1..E1.k`；模型对每条相关结果给出 `targets / passage / quote / relation / basis / causal_test / context / scope_match`。目标或片段编号无效、字段不合格、短引在该片段里 `find` 不到 —— 只丢这一条，记进 `material_notes`。
3. **评级**（程序，`assess.aggregate`）：
   - 范围：已知物种冲突（含别名表：mice/murine/Mus musculus/小鼠 → mouse 等）直接判 mismatched，覆盖模型标记；目标已限定而观察没报告该维度 → unknown；unknown 不算匹配。
   - 因果命题和因果连接只接受“干预 + 直接检验该因果问题 + 原始研究结果”；综述、图谱、登记/方案都不行。`reported` 命题可由原始研究的非干预观察支持，但登记/方案不能借此绕过。
   - 支持与反驳都在目标范围内时：物种、系统、疾病都报告且一致才算 `conflicts`；否则进 `unresolved_counterevidence`，不算已解决、也不忽略；物种明确不同的反向结果只记为适用性缺口。
   - 命题：mixed / supported / refuted / insufficient；连接：supported / compatible_only / unestablished。
   - overall：每项命题和连接都有适用支持、支持者物种一致、无冲突无未解决反证 → supported；部分有支持 → partial；否则 insufficient。
   - `gaps` 为每个未确立的义务写明缺哪类原始观察；“没在本材料包里看到”从不等于“没人做过”。
4. **材料可评性**：新闻、新闻稿、公司声明、社媒不进入评价（记为 `material_type_not_evaluable`，由雷达保留为线索）；`document_type` 与 `source_kind` 矛盾、PMID/DOI 无效或互相矛盾的材料也不进入评价并注明原因。摘要可以支持命题，但 `assessability` 记为 limited。

## 开发环境与测试

唯一的入口是 `Makefile`。需要 uv 0.13.0（在 PATH 上，例如 `~/.local/bin/uv`；或者 `make UV=<路径>`）。

```sh
make venv            # .venv：只按 requirements.lock 安装，每个包都校验 sha256
make test            # 全部测试（不联网）
make check-install   # 一个用完即删的新环境：按锁文件安装、逐个导入模块、再跑一遍测试
make check-lock      # requirements.lock 是否仍与 requirements.txt 一致（要访问包索引）
```

- 直接依赖写在 `requirements.txt`（运行 + 测试，均为精确版本）；`requirements.lock` 是由它解析出的完整锁（运行 + 测试），每个包都带 sha256，解析目标是 CPython 3.12、x86_64-unknown-linux-gnu（锁文件头一行写着解析命令）。改了依赖后运行 `make lock`，再提交锁文件。
- 已验证的组合：uv 0.13.0、CPython 3.12.15、81 个测试通过。

测试不联网，模型由 `tests/stub.py` 的替身代替。`examples/identity_contract.json` 是文献身份的离线契约用例。

## 这次重写改了什么（2026-11）

上一轮验收在 local review 阶段报 `model_output_invalid`：旧 schema 要求模型自己维护 C/J/L/M 观察编号、`matL:P003` 片段编号、带绝对字符偏移的 source_notes，错一个整份作废；服务还把模型请求交回雷达代调、再回放回执。现在：

- 服务自己调模型（`llm.py`），一次请求就是一次同步 HTTP 调用；删掉反向协议、回执、版本哈希和 capabilities。
- 两次调用、两份小 schema；编号全部由程序分配，坏的一条只丢一条；思考关闭，`max_tokens` 4096，校验失败带错误重试一次。
- 旧的 local review 与 mechanism review 合并为一次审阅；删掉 source_notes 偏移锚点层、组合（combinations）、两两比较（comparisons）、whole_assessment、direct_baseline 和消融开关、cohorts 独立性评审、Fabric 读取命令和 evaluation/ 脚本。
- 原有的确定性评级规则保留，见上一节。

## 已知局限

- 冲突的“可比情境”现在只看物种/系统/疾病字符串是否一致（物种用别名表），不再有模型逐对比较；措辞不同的同一系统会落到 `unresolved_counterevidence`。
- 整体见证只检查物种一致；不同实验系统的结果仍可能被拼在一起，需要人看 `observations`。
- 片段按句切分，长全文会让单次审阅的输入很大；目前没有截断。
- `causal_test`、`scope_match` 等语义标记来自模型，不是专家标签；研究效果仍需独立专家标注与跨研究检验。
