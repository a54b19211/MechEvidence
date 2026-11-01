# MechEvidence

汪鑫怡维护机制证据评价：针对明确问题，判断给定证据支持哪些机制、在哪些人群和条件下成立，以及存在的反证和缺口。Fabric 提供原文版本，唐蔚雯提供对象与关系依据；本模块返回专业评价，BioCheck 再检查报告是否准确表达，雷达负责取证与保存。

## 启动和检查

使用 Python 3.10+，安装 [requirements.txt](requirements.txt)。通过私有环境提供 `MECHEVIDENCE_SERVICE_TOKEN`：

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m mechevidence.service --host 127.0.0.1 --port 18561
python -m unittest discover -s tests -v
```

HTTP 服务不独立持有商业模型密钥；`/health`、`/v1/capabilities`、`/v1/execute` 使用 Bearer 令牌。`bioradar.professional.v1` 请求携带 operation_id、operation=evaluate、arguments 和已保存 model_receipts。需要模型时由雷达执行、记账、保存回执，再继续同一专业任务。

雷达配置 `BIORADAR_MECH_EVIDENCE_TRANSPORT=http`、`BIORADAR_MECH_EVIDENCE_SERVICE_URL`、`BIORADAR_MECH_EVIDENCE_SERVICE_TOKEN`。本地 unittest 验方法；完整任务和四项集成检查在 [BioRadar README](../BioRadar/README.md)，这里没有独立四项 Makefile。

## 跑一个真实问题

查看 [examples/april.json](examples/april.json)，以“APRIL 通路在相关疾病中的作用得到什么证据支持，哪些仍是推测？”组织问题。独立命令为 `python -m mechevidence evaluate examples/april.json --output runs/april-first`，读取项目私有模型配置并产生实际费用；接入雷达时由路由选择专业评价，沿用雷达统一模型执行。

纯函数 `evaluate(case, call_json)` 返回完整评价；`generation_view(result)` 提供给写作者的上下文，原回执仍保存。材料身份冲突返回 needs_input，而不是猜测同一对象；缺证据、与条件不符和确有反证分开。检查对象、干预、人群、研究设计、时间及引用。机制推断不升级成临床有效性，背景图路径不计作独立研究。

`python -m mechevidence fetch --terms APRIL --source europe_pmc --limit 10 --output data/april-page.json` 可读取一页授权 Fabric 资料；覆盖和游标必须保留，一页不能当作完整文献集合。

## 本轮实际验证与未完成

2026-11-01 单一运行记录在 `/home/zk_test/zk_projects/.branch-archive-20261101/runs.jsonl`。系统 32 项业务分析、233 次新模型请求、已知 token 下界 5,863,856，另有 1 次用量未知。97 条专业 HTTP 往返中 MechEvidence evaluate 为 5 条，含恢复过程，不等于 5 次成功完成的独立专业任务。A/B/C PASS 0/3，不能由接口运行推断专业判断有效。

未完成：B 第二次冒烟在 mech_local_review 得到无效模型输出；先用归档回执复现结构和内容问题，再与雷达处理明确失败及未知结果的不同恢复路径。研究效果还需独立专家标签与跨研究测试，来源相同的变体不能充当独立样本。

按用户最新决定，本地仅保留 main，不推送远端。旧分支在 MechEvidence.bundle；现有私有材料与审计日志保留。
