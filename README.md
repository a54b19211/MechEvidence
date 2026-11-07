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
uvicorn mechevidence.service:app --host 127.0.0.1 --port 18563
```

| 环境变量 | 说明 |
|---|---|
| `SERVICE_TOKEN` | 必填。请求头 `Authorization: Bearer <token>`；未设置时服务拒绝所有评价请求 |
| `LLM_BASE_URL` | OpenAI 兼容地址，默认 `https://api.deepseek.com` |
| `LLM_API_KEY` | 模型密钥 |
| `LLM_MODEL` | 默认 `deepseek-v4-pro` |
| `LLM_PROVIDER` | `deepseek`（发送 `thinking: disabled`）、`vllm` 或 `openai` |
| `LLM_ENDPOINT_ID` | 运维给模型端点起的名字（不是 URL、不含密钥），进评价器身份；默认 `unnamed` |
| `MECHEVIDENCE_CODE_VERSION` | 构建/部署时写入的提交号，只用于审计，不进评价器身份 |
| `MECH_AUTHORIZATION_URL`、`MECH_AUTHORIZATION_TOKEN`、`MECH_OUTCOME_URL` | v2 每次模型请求前现查授权、每次请求后交回结果的可信端点、令牌和结果地址（含 `{attempt_id}`）；缺任何一项，v2 就不发模型请求 |
| `MECH_AUTHORIZATION_TIMEOUT_SECONDS` | 默认 5 |
| `LLM_TIMEOUT_SECONDS` | 默认 180；超时不自动重发 |

不经 HTTP 直接跑一个请求文件（会真实调用模型、产生费用）：`python -m mechevidence evaluate examples/question.json`。

## 接口

- `GET /health` → `{"status": "ok", "contracts": ["mechevidence.evaluate.v1", "mechevidence.evaluate.v2"], "evaluator_id": "…"}`（不需要令牌）
- `GET /v2/evaluator` → 当前评价器身份
- `POST /v2/evaluate`：版本绑定契约，见下一节
- `POST /v1/evaluate`：兼容路径；评价本身不变，回执多 `contract_version`（v1）、`evaluator`、`binding_state: "unbound"`、`reusable: false`；契约未写的字段现在报 422 `field_not_in_contract`，不再静默丢弃；不查授权、不带原始回答
- 出错时统一返回 `{"error_code": "...", "message": "..."}`：`unauthorized`(401)、`input_invalid`(422)、`field_not_in_contract`(422)、`decomposition_failed`(502，附 `llm_calls`)、`service_not_configured`(503)、`not_found`(404)；v2 另有下一节的几个码。

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

## v2：版本绑定契约

`POST /v2/evaluate` 评价的是“命题 + 范围 + 一组固定版本的材料 + 评价器”，结果可复现、可缓存、可审计。完整的请求、回执和负例见 `examples/contract-v2/`（雷达的测试直接复用）；设计和取舍见 [docs/decisions/0002-version-bound-contract.md](docs/decisions/0002-version-bound-contract.md)。

- 请求：`contract_version: "mechevidence.evaluate.v2"`；`scope` 每个维度明确是 `{"state": "stated", "value": …}`、`{"state": "unknown"}`（没提供）或 `{"state": "not_required"}`（明确不限定），空字符串一律按 `unknown`；每份材料带 `binding`（与雷达的 `material.binding()` 同形：整串 sha256、字符数、literal 段和 part 段，part 段有来源/记录/版本/原件哈希/处理器版本和段哈希）；文献身份字段放在 `identifiers` 里；`authorization_ref`（必填，不透明字符串）；`expected_evaluator_id`（可选）。任何层出现契约未写的字段都是 422。
- 服务只能核“文字和绑定自洽”（回执写 `binding_checked: "self_consistent"`）；“这串文字确实是那个固定版本的那一段”由雷达发出前向 Fabric 重读核对。
- 回执：`contract_version`、完整 `evaluator`（含 `evaluator_id`）、`reuse_key`、`complete`（预算不够或审阅失败时为 false，不得复用）、`audit_complete`、规范化后的 `scope`、每份材料的 `binding_sha256` 和定位、评价结果，以及每次逻辑调用的 `llm_calls[].outputs`：每个实际发出的 HTTP 请求一条 `{attempt, index, attempt_id, decision_id, delivered, status, error, content, sha256, truncated}`（`index`、`attempt_id`、`decision_id` 是整次评价内的全局序号和授权端点登记的行，失败的 502 回应里也带，迟到或失败的回执据此对上服务端已准备的行），`content` 是供应方返回的原始 HTTP 正文（先存档，再解析），`sha256` 是整份正文字节的哈希；没有回答（超时、连不上）时 `content` 为 null。
- 复用键（第一期，保守）：命题、范围、**有序**的材料列表（含调用方别名、标题、类型、层级、身份字段和合组结果）、批大小、评价器 id。同样的材料换顺序、换别名就是另一个键；`max_llm_calls`、`authorization_ref` 不进键。
- `expected_evaluator_id` 与当前评价器不同 → 409 `evaluator_changed`，附当前身份，不评价、不调模型。
- 授权（`model_attempt_authorization.v1`，协议见 `authorization.py`）：每次真实模型 HTTP 请求（拆解、审阅、格式重试各算一次，`attempt.index` 在整次评价内单调递增，`attempt.id` 每次唯一）之前，服务向预配置的可信端点 POST `{authorization_ref, evaluation_key（即 reuse_key）, evaluator_id, attempt{id, stage, index, request_body_sha256, model, endpoint_id}}`；只有 200、`valid:true`、带 `decision_id`、并回显本次 attempt id、evaluation_key、evaluator_id 且未过期（`expires_at` 带时区）才发。裸 `{valid:true}`、回显不符、过期、非 200、非 JSON、端点不可达、未配置都不发，返回 502 `authorization_blocked`。每个请求结束后、下一次请求之前，服务把结果（送达状态、HTTP 状态、原始正文及哈希、是否截断）POST 到 `MECH_OUTCOME_URL`；服务端必须逐字段确认：`contract_version=model_attempt_outcome.v1`、`recorded:true`、同一个 `attempt_id`、同一个 `decision_id`、同一个 `response_sha256`（没有回答时为 null），任何一项不符都算没记下，不再发任何请求。端点只来自环境，不接受请求里的地址。已登记而没有结果的尝试，对账时按“可能已发出”处理。
- 原始回答超过 256 KiB（不能完整存入回执）→ 502 `output_oversize`，没有结论、不可采用，且不再做格式重试。
- 供应方回了 HTTP 200 但响应信封读不出来（例如 `choices` 里不是对象）→ 记 `response_malformed`，已发出的请求和它的原始正文照常进 `llm_calls`，整体是可处理的 502（`decomposition_failed` 等），不是 500。
- 收到 200 回执后，调用方用 `contract.adopt()`，它永不因畸形回执抛异常：先按 `receipt.SHAPE` 表逐字段查 JSON 类型（bool 不是 int，列表/对象各归各位）、枚举和哈希形式；再查跨字段规则——每条存档的 HTTP 回答自身一致（delivered/status/content/sha256/error）、`requests` 与存档条数相等且序号全局连续、attempt_id 唯一、结论行引用的观察/目标/材料都存在、`complete` 与材料备注一致、范围和材料回显与请求一致；最后核对与请求的身份。结构或跨字段不对 → `failed/receipt_invalid`；不是这个请求的回执 → `failed/receipt_mismatch`；审计不完整或有截断 → `failed/audit_incomplete`。三种都保留原始回执做审计，结论不可采用、不可缓存。
- 422 码：`field_not_in_contract`、`contract_version_required`、`material_binding_missing`、`material_binding_mismatch`、`material_binding_invalid`、`duplicate_material_binding`、`scope_state_invalid`、`authorization_required`、`input_invalid`。

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
make lint            # ruff（pyproject.toml）
make metrics         # 函数长度 80 行 / 圈复杂度 20 的门槛（scripts/code_metrics.py；例外清单 scripts/metrics_exceptions.txt 只许变短）
make check-install   # 一个用完即删的新环境：按锁文件安装、逐个导入模块、再跑一遍测试
make check-lock      # requirements.lock 是否仍与 requirements.txt 一致（要访问包索引）
```

`make lint` 的规则：所有仓库共用一份基础规则 `E4, E7, E9, F, W, I`（devtools `README.md` 的“共用的代码门槛”）。圈复杂度只由 `make metrics` 管，不开 ruff 的 C90。在基础规则之上多开规则的仓库：

| 仓库 | 多开的规则 |
|---|---|
| BioRadar | `E`（含行宽 E501，110） |
| BioCheck | `E`（含行宽 E501，120） |

`make check-lock`、`make lock` 要访问 PyPI（`make venv`、`make check-install` 在 uv 缓存里没有的包也要下载）。zktest 上直连 PyPI 有时超时，这时只给这几条命令临时加上本机已有的 HTTP 代理（mihomo，监听 127.0.0.1:7890），例如 `HTTPS_PROXY=http://127.0.0.1:7890 HTTP_PROXY=http://127.0.0.1:7890 make check-lock`；不要把代理写进全局配置。`check-lock` 的结果有两种失败：解析本身失败（包索引或网络不可用、依赖解不出来）时报 `check-lock did not run`，make 显示 `Error 2`，锁文件没有比较过，不能因此重做锁；只有解析成功而内容不同时才报 `requirements.lock is stale`（make 显示 `Error 1`）。两种情况 make 都以非零状态结束。

- v2 契约的纯函数（evaluator id、reuse key、回执校验、adopt、文献身份）在共享包 `extraction-contract` 1.14.1 的 `extraction_contract.mech_v2` 里，雷达导入同一份；wheel 哈希写在 `requirements-contract.txt` 和锁里，`make` 用 `EXTRACTION_CONTRACT_WHEELS` 找 wheel（默认 `../extraction-contract/dist`）。本仓库只保留请求校验（pydantic 模型）并把校验后的请求交给它。
- 模型调用的 HTTP 一侧用共享包 `zk-llm-transport`（0.1.1，兄弟目录 `llm-transport` 构建的 wheel，哈希写在 `requirements-transport.txt` 和锁里；`make` 用 `LLM_TRANSPORT_WHEELS` 找 wheel，默认 `../llm-transport/dist`，即本仓库和 `llm-transport` 同在 `zk_projects/` 下的位置；放在别处的 worktree 用 `make LLM_TRANSPORT_WHEELS=/home/zk_test/zk_projects/llm-transport/dist …` 显式给出，不走 PyPI）。
- 直接依赖写在 `requirements.txt`（运行 + 测试，均为精确版本）；`requirements.lock` 是由它解析出的完整锁（运行 + 测试），每个包都带 sha256，解析目标是 CPython 3.12、x86_64-unknown-linux-gnu（锁文件头一行写着解析命令）。改了依赖后运行 `make lock`，再提交锁文件。
- 已验证的组合：uv 0.13.0、CPython 3.12.15；通过数量以 `make check` 的输出为准，这里不写数字。

测试不联网，模型由 `tests/stub.py` 的替身代替。`examples/identity_contract.json` 是文献身份的离线契约用例；`examples/contract-v2/` 是 v2 契约夹具，由 `tests/contract_cases.py` 生成，改了契约、提示词或规则后用 `MECH_REGEN_FIXTURES=1 python -m pytest tests/test_contract_fixtures.py` 重生。

## 设计来由

服务为什么是现在这个形状（同步 HTTP、两次调用、程序编号），见 [docs/decisions/0001-synchronous-two-call-service.md](docs/decisions/0001-synchronous-two-call-service.md)。

## 已知局限

- v2 第一期的复用键是有序的：材料顺序、调用方别名不同就不复用；规范化排序和别名重映射是单独的下一版评价器（`ordering_rule` 与 `rules_version` 同时升级）。
- v1 兼容路径在一次请求内部不复查授权，也不回传模型原始回答；这是已知的兼容缺口。
- 逐次结算（`MECH_OUTCOME_URL`）的持久语义在可信端点一侧；服务进程在两次请求之间被杀死时，回执（只在整份响应里回传原文）不会产生，已登记而没有结果的尝试就是“结果未知”，这个崩溃审计缺口仍在，直到 Radar 一侧落存接通。
- 授权协议的服务端已实现并有替身端点的测试，但没有与 Radar 一侧联调。
- 评价器身份在每个请求开始时按当前环境算出；同一请求执行期间若进程环境被改，回执里的身份可能与后几次模型调用不一致（部署应通过重启切换）。
- 模型响应由 httpx 整体读入内存后才判断 256 KiB 上限；`max_tokens` 限制了正常输出，但没有流式截断。
- 服务不访问 Fabric：绑定只核到自洽，是否对应固定版本由调用方重读保证。

- 冲突的“可比情境”现在只看物种/系统/疾病字符串是否一致（物种用别名表），不再有模型逐对比较；措辞不同的同一系统会落到 `unresolved_counterevidence`。
- 整体见证只检查物种一致；不同实验系统的结果仍可能被拼在一起，需要人看 `observations`。
- 片段按句切分，长全文会让单次审阅的输入很大；目前没有截断。
- `causal_test`、`scope_match` 等语义标记来自模型，不是专家标签；研究效果仍需独立专家标注与跨研究检验。
