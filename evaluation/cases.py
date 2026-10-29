"""Real-source cases and separate reviewer expectations, never LLM-generated gold."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from mechevidence.assess import prepare_passages
from mechevidence.fabric import to_material


def _fulltext(directory: Path, pmcid: str, pmid: str, title: str,
              selectors: list[tuple[str, int]]) -> tuple[dict, list[dict]]:
    """Keep original body paragraphs; supplementary XML has its own content version."""
    path = directory / f"{pmcid}.xml"
    raw = path.read_bytes()
    root = ET.fromstring(raw)
    texts, locations = [], []
    for section_id, paragraph_index in selectors:
        section = root.find(f'.//body//sec[@id="{section_id}"]')
        if section is None:
            raise ValueError(f"{pmcid}: missing section {section_id}")
        paragraphs = section.findall("p")
        if paragraph_index >= len(paragraphs):
            raise ValueError(f"{pmcid}: missing paragraph {section_id}/{paragraph_index}")
        text = " ".join("".join(paragraphs[paragraph_index].itertext()).split())
        if not text:
            raise ValueError(f"{pmcid}: empty original paragraph")
        start = sum(len(value) + 2 for value in texts)
        texts.append(text)
        locations.append({"section_id": section_id, "paragraph_index": paragraph_index,
                          "selected_text_start": start, "selected_text_end": start + len(text)})
    digest = hashlib.sha256(raw).hexdigest()
    return ({"id": "M1", "text": "\n\n".join(texts),
             "source_id": "europe_pmc_fulltext_supplement", "record_id": pmcid,
             "version": "xml-sha256:" + digest, "title": title,
             "source_group": "PMID:" + pmid, "source_kind": "primary",
             "text_level": "selected_passages", "url": f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/"},
            locations)


def _select_sentences(material: dict, needles: list[str]) -> tuple[dict, list[dict]]:
    """Remove all other text from the model view while preserving original offsets."""
    result = deepcopy(material)
    passages = prepare_passages({"question": "Source selection", "mechanism": "Source selection",
                                 "materials": [result]})
    selected = []
    for needle in needles:
        matches = [item for item in passages if needle in item["text"]]
        if len(matches) != 1:
            raise ValueError(f"Expected one original sentence containing {needle!r}, got {len(matches)}")
        selected.extend(matches)
    result["allowed_passage_ids"] = list(dict.fromkeys(item["id"] for item in selected))
    result["text_level"] = "selected_passages"
    return result, [{key: item[key] for key in ("id", "start", "end")} for item in selected]


def build_cases(records_path: str | Path, fulltexts_dir: str | Path) -> tuple[list[dict], dict[str, dict]]:
    """Return model inputs separately from provisional engineering expectations.

    Variants sharing a family remain one source group for statistical purposes.
    These expectations expose intended failure modes; they are not expert gold labels.
    """
    records = json.loads(Path(records_path).read_text())
    selected = ("24444792", "38744277", "38123696", "32954360",
                "33089218", "40576564", "39806405")
    index = {str(item.get("facts", {}).get("pmid") or item["source_record_id"].split(":")[-1]): item
             for item in records}
    missing = set(selected) - index.keys()
    if missing:
        raise ValueError(f"Frozen Fabric records missing PMIDs: {sorted(missing)}")
    materials = {pmid: to_material(index[pmid], material_id="M1", source_kind="primary")
                 for pmid in selected}
    directory = Path(fulltexts_dir)
    cases, expectations = [], {}

    def add(family, label, reason, question, mechanism, claims, joins, source, scope,
            allowed=("supported",), **checks):
        case_id = f"Q{len(cases) + 1:03d}"
        cases.append({"id": case_id, "question": question, "mechanism": mechanism,
                      "scope": deepcopy(scope), "claims": deepcopy(claims),
                      "joins": deepcopy(joins), "materials": [deepcopy(source)]})
        expectations[case_id] = {"family": family, "label": label, "reason": reason,
                                 "allowed_verdicts": list(allowed),
                                 "annotation_status": "provisional_reviewer_expectation_not_expert_gold",
                                 **checks}
        return case_id

    mouse_neurons = {"species": "mouse", "system": "cultured developing hippocampal pyramidal neurons",
                     "disease": None}
    april_claims = [{"id": "C1", "text": "APRIL treatment activates ERK1/ERK2 in cultured developing mouse hippocampal pyramidal neurons.", "kind": "causal"},
                    {"id": "C2", "text": "APRIL treatment enhances axonal elongation in these cultured neurons.", "kind": "causal"}]
    april_joins = [{"id": "J1", "text": "ERK signaling contributes to mediating the axon-elongating effect of APRIL in these cultured neurons.",
                    "claim_ids": ["C1", "C2"]}]
    april_question = "在培养的小鼠发育期海马锥体神经元中，哪些证据支持 ERK 信号参与介导 APRIL 的促轴突延长作用？"
    april_mechanism = "APRIL 激活 ERK1/ERK2，ERK 信号分支参与介导 APRIL 诱导的轴突延长。"
    original = add("APRIL", "原始摘要中的局部作用与中介干预", "原摘要同时报告 ERK 激活、轴突延长及 MEK 抑制干预，可支持所声明的 ERK 分支；不额外声称 ERK 驱动 Akt。",
                   april_question, april_mechanism, april_claims, april_joins,
                   materials["24444792"], mouse_neurons, claims_should_support=["C1", "C2"],
                   joins_should_support=["J1"])
    april_ft, april_locations = _fulltext(directory, "PMC4008386", "24444792", materials["24444792"]["title"],
        [("s0085", 0), ("s0085", 1), ("s0090", 0), ("s0095", 1), ("s0095", 2),
         ("s0100", 1), ("s0100", 2), ("s0105", 0), ("s0105", 1)])
    add("APRIL", "同一候选的全文补充对照", "相同问题和候选，仅材料换为原始全文实验段。XML 补充独立标记版本，不能冒充 Fabric 摘要版本。",
        april_question, april_mechanism, april_claims, april_joins, april_ft, mouse_neurons,
        paired_case_id=original, claims_should_support=["C1", "C2"], joins_should_support=["J1"],
        fulltext_supplement={"pmcid": "PMC4008386", "locations": april_locations})
    without_join, kept_locations = _select_sentences(materials["24444792"], [
        "are expressed by hippocampal pyramidal cells of fetal and postnatal mice",
        "Recombinant APRIL enhanced axonal elongation",
        "APRIL promoted phosphorylation and activation of ERK1"])
    add("APRIL", "保持两条局部作用但移除全部中介支持", "只向模型开放物种背景、APRIL 促轴突及激酶变化的原句；阻断实验和机制总结均不开放。两个局部结果不能证明 ERK 介导终点。",
        april_question, april_mechanism, april_claims, april_joins, without_join, mouse_neurons,
        allowed=("partial",), paired_case_id=original, claims_should_support=["C1", "C2"],
        joins_must_not_support=["J1"], permitted_source_spans=kept_locations)
    add("APRIL", "将 GSK-3β 磷酸化误判为激活", "原文明确 Ser9 磷酸化对应失活；持续活化 GSK-3β 阻断 APRIL 促轴突效应，不能将 phosphorylation 统一视作 activation。",
        "APRIL 是否通过激活 GSK-3β 促进这些小鼠神经元的轴突延长？",
        "APRIL 引起 GSK-3β 磷酸化并提高其激酶活性，GSK-3β 激活介导轴突延长。",
        [{"id": "C1", "text": "APRIL increases GSK-3β serine phosphorylation in cultured hippocampal neurons.", "kind": "causal"},
         {"id": "C2", "text": "APRIL increases GSK-3β kinase activity in these neurons.", "kind": "causal"}],
        [{"id": "J1", "text": "Activation of GSK-3β mediates APRIL-induced axonal elongation.", "claim_ids": ["C1", "C2"]}],
        april_ft, mouse_neurons, allowed=("partial", "insufficient"), claims_should_refute=["C2"],
        joins_must_not_support=["J1"])
    add("APRIL", "把并列分支串联为 ERK 到 Akt", "ERK 和 PI3K/Akt/GSK-3β 分支各有干预支持，但提供段落未建立 ERK 位于 Akt 上游。局部激活均支持时仍须发现串联缺口。",
        "APRIL 的 ERK 激活是否在这些培养神经元中通过进一步激活 Akt 产生促轴突作用？",
        "APRIL 先激活 ERK，ERK 再激活 Akt，随后促进轴突延长。",
        [{"id": "C1", "text": "APRIL activates ERK1/ERK2 in these neurons.", "kind": "causal"},
         {"id": "C2", "text": "APRIL activates Akt in these neurons.", "kind": "causal"},
         {"id": "C3", "text": "APRIL enhances axonal elongation in these neurons.", "kind": "causal"}],
        [{"id": "J1", "text": "APRIL-induced ERK activation is upstream of and mediates Akt activation in these neurons.", "claim_ids": ["C1", "C2"]},
         {"id": "J2", "text": "Akt signaling contributes to the APRIL-induced axon elongation.", "claim_ids": ["C2", "C3"]}],
        april_ft, mouse_neurons, allowed=("partial",), claims_should_support=["C1", "C2", "C3"],
        joins_must_not_support=["J1"], gap_kind="untested_order_between_parallel_branches")
    add("APRIL", "把轴突终点换成树突", "同一摘要明确没有促进 dendrite elongation，属于可比终点反证，不能只笼统说材料不足。",
        "APRIL 是否增加培养的小鼠海马锥体神经元的树突延长？", "APRIL 增加这些培养神经元的树突延长。",
        [{"id": "C1", "text": "APRIL increases dendrite elongation in cultured developing mouse hippocampal pyramidal neurons.", "kind": "causal"}],
        [], materials["24444792"], mouse_neurons, allowed=("insufficient",), claims_should_refute=["C1"])
    add("APRIL", "从发育期小鼠细胞推广患者临床终点", "给定培养小鼠材料不能验证阿尔茨海默病患者记忆改善；这是适用性及临床转化缺口，不能当作患者实验的反驳。",
        "这些研究能否证明 APRIL 通过 ERK 改善阿尔茨海默病患者的记忆？", "APRIL 激活患者海马神经元 ERK，并通过该环节改善患者记忆。",
        [{"id": "C1", "text": "APRIL activates ERK in hippocampal neurons of patients with Alzheimer's disease.", "kind": "causal"},
         {"id": "C2", "text": "APRIL treatment improves memory in patients with Alzheimer's disease.", "kind": "causal"}],
        [{"id": "J1", "text": "Hippocampal ERK activation mediates the memory improvement in these patients.", "claim_ids": ["C1", "C2"]}],
        april_ft, {"species": "human", "system": "patients with Alzheimer's disease", "disease": "Alzheimer's disease"},
        allowed=("insufficient",), gap_kind="scope_and_clinical_translation", claims_must_not_refute=["C1", "C2"])

    il7_scope = {"species": None, "system": "solid tumour CD8 tumour-infiltrating lymphocytes", "disease": "solid tumours"}
    il7_claims = [{"id": "C1", "text": "rhIL-7-hyFc expands PD-1-negative tumour-nonresponsive bystander CD8 TILs.", "kind": "causal"},
                  {"id": "C2", "text": "Tumour-antigen-specific TCE treatment can induce tumouricidal activity in these bystander CD8 T cells.", "kind": "causal"}]
    add("IL7-bystander", "原摘要的扩增与重定向作用", "摘要明确报告主要扩增旁观者细胞及 TCE 使其获得杀伤功能；不把这个现象强化为免疫完整组合疗效的必要中介。",
        "IL-7-Fc 扩增哪类 CD8 TIL，TCE 能否使其获得杀伤功能？", "IL-7-Fc 增加 PD-1 阴性旁观者 CD8 TIL，TCE 重定向这些细胞产生杀伤功能。",
        il7_claims, [], materials["38744277"], il7_scope, allowed=("supported", "partial"),
        claims_should_support=["C1", "C2"])
    il7_ft, il7_locations = _fulltext(directory, "PMC11148861", "38744277", materials["38744277"]["title"],
        [("sec2.1", 0), ("sec2.3", 0), ("sec2.3", 1), ("sec2.3", 2), ("sec3.1", 0)])
    mouse_tumour = {"species": "mouse", "system": "tumour-bearing mice treated with rhIL-7-hyFc and FTY720", "disease": "experimental solid tumours"}
    fty, fty_locations = _fulltext(directory, "PMC11148861", "38744277", materials["38744277"]["title"], [("sec2.3", 0)])
    add("IL7-bystander", "FTY720 单药机制的直接阴性结果", "FTY720 减少外周和瘤内 CD8 细胞但未降低 IL-7-Fc 单药疗效。只否定指定干预结果，不泛化到旁观者从不贡献或组合疗法机制。",
        "本文 FTY720 阻断外周淋巴细胞迁移后，IL-7-Fc 单药的抗瘤疗效是否降低？", "FTY720 阻断外周淋巴细胞迁移降低了 IL-7-Fc 单药的抗瘤疗效。",
        [{"id": "C1", "text": "In the reported tumour-bearing mouse experiment, FTY720 reduced the antitumour efficacy of rhIL-7-hyFc monotherapy.", "kind": "causal"}],
        [], fty, mouse_tumour, allowed=("insufficient",), claims_should_refute=["C1"], fulltext_supplement={"locations": fty_locations})
    rag, rag_locations = _fulltext(directory, "PMC11148861", "38744277", materials["38744277"]["title"], [("sec2.3", 2)])
    add("IL7-bystander", "特定 RAG1 缺失移植模型的充分性", "原文联合移植和 TCE 的实验支持指定条件下的抗瘤充分性；不外推免疫完整鼠的必要性或患者疗效。",
        "在 MC38 荷瘤 RAG1 缺失小鼠中，移入 IL-7-Fc 扩增旁观者 CD8 并给予 PD-L1×CD3 TCE 能否抑制肿瘤？",
        "在上述移植模型中，旁观者 CD8 细胞联合 PD-L1×CD3 TCE 能抑制肿瘤生长。",
        [{"id": "C1", "text": "Adoptive transfer of rhIL-7-hyFc-expanded PD-1-negative bystander CD8 T cells together with PD-L1×CD3 TCE suppresses MC38 tumour growth in RAG1-deficient mice.", "kind": "causal"}],
        [], rag, {"species": "mouse", "system": "MC38 tumour-bearing RAG1-deficient mice receiving adoptively transferred bystander CD8 T cells", "disease": "MC38 tumours"},
        claims_should_support=["C1"], fulltext_supplement={"locations": rag_locations})
    add("IL7-bystander", "充分性不能代替组合疗效的必要性", "局部扩增和 TCE 杀伤有支持，但 Limitations 明示这些细胞对组合疗效的贡献大小尚未确定；FTY720 单药结果不能直接当组合中介反证。",
        "现有证据是否证明这些旁观者 CD8 是免疫完整荷瘤鼠中 IL-7-Fc 与 TCE 组合疗效的必要中介？",
        "IL-7-Fc 扩增旁观者 CD8，TCE 赋予其杀伤功能；这些细胞是免疫完整小鼠组合疗效的必要中介。",
        il7_claims, [{"id": "J1", "text": "The expanded bystander CD8 T cells are a necessary mediator of the antitumour benefit of rhIL-7-hyFc plus TCE in immunocompetent tumour-bearing mice.", "claim_ids": ["C1", "C2"]}],
        il7_ft, {"species": "mouse", "system": "immunocompetent solid tumour-bearing mice", "disease": "solid tumours"},
        allowed=("partial",), joins_must_not_support=["J1"], joins_must_not_refute=["J1"],
        fulltext_supplement={"locations": il7_locations}, gap_kind="sufficiency_does_not_establish_necessity")
    add("IL7-bystander", "把旁观者误写为肿瘤反应性细胞", "摘要明确多数新增细胞为 PD-1 阴性、原本肿瘤不响应旁观者，且 TCE 对 tumour-reactive CD8 无主要作用。",
        "IL-7-Fc 新增的主要 CD8 TIL 是否为 PD-1 阳性肿瘤反应性细胞？", "IL-7-Fc 主要扩增 PD-1 阳性肿瘤反应性 CD8 TIL，TCE 主要作用于这群细胞。",
        [{"id": "C1", "text": "The majority of rhIL-7-hyFc-expanded CD8 TILs are PD-1-positive tumour-reactive cells.", "kind": "reported"}],
        [], materials["38744277"], il7_scope, allowed=("insufficient",), claims_should_refute=["C1"])

    car_scope = {"species": None, "system": "CAR T cells in the reported preclinical models", "disease": None}
    add("membrane-IL7", "膜结合可降解 IL7 的抗原依赖增殖", "摘要报告共递送膜结合 IL7 后的抗原依赖增殖和体内肿瘤控制；只评价该工程系统。",
        "所述膜结合、lenalidomide 可降解 IL7 系统对 CAR T 细胞有什么实验作用？", "与 CAR 共递送的膜结合 IL7 增强抗原依赖增殖并改善所测体内模型的肿瘤控制。",
        [{"id": "C1", "text": "Co-delivery of the membrane-bound IL-7 fusion with a CAR enhances antigen-dependent proliferative capacity.", "kind": "causal"},
         {"id": "C2", "text": "The IL-7-augmented CAR T system enhances tumour control in the reported in vivo models.", "kind": "causal"}],
        [], materials["38123696"], car_scope, allowed=("supported", "partial"))
    add("membrane-IL7", "倒置药物控制方向", "融合 IL7 被明确描述为 lenalidomide-degradable；不能据此说药物通过提高融合蛋白丰度促进扩增。摘要未提供的定量动力学也不补写。",
        "该系统中 lenalidomide 是否通过增加融合 IL7 蛋白丰度增强 CAR T 扩增？", "Lenalidomide 提高膜结合融合 IL7 的丰度，并通过该提高促进 CAR T 扩增。",
        [{"id": "C1", "text": "Lenalidomide increases the abundance of the engineered membrane-bound IL-7 fusion protein.", "kind": "causal"}],
        [], materials["38123696"], car_scope, allowed=("partial", "insufficient"), claims_must_not_support=["C1"])
    myeloma_mouse = {"species": "mouse", "system": "wildtype and Vk*MYC mouse models", "disease": "monoclonal gammopathy and multiple myeloma"}
    add("MIR15-RB1", "单等位 MIR15 缺失的干预作用", "摘要直接比较候选基因，报告 Mir15a/Mir16-1 单等位缺失加速野生型单克隆病和 Vk*MYC MM 进展；不把人患者关联当人类基因干预。",
        "小鼠模型是否支持 Mir15a/Mir16-1 单等位缺失驱动浆细胞病进展？", "Mir15a/Mir16-1 单等位缺失加速所测野生型及 Vk*MYC 小鼠的浆细胞病进展。",
        [{"id": "C1", "text": "Monoallelic Mir15a/Mir16-1 deletion accelerates monoclonal gammopathy in wildtype mice and myeloma progression in Vk*MYC mice.", "kind": "causal"}],
        [], materials["32954360"], myeloma_mouse)
    add("MIR15-RB1", "混淆单等位与双等位 RB1", "原摘要明确单等位 Rb1 缺失没有相同充分性；RB1 双等位缺失与疾病进展关联不能替代这个干预对比。",
        "Rb1 单等位缺失是否与 Mir15a/Mir16-1 单等位缺失一样足以加速上述小鼠疾病？", "Rb1 单等位缺失足以加速野生型小鼠单克隆病及 Vk*MYC 小鼠 MM 进展。",
        [{"id": "C1", "text": "Monoallelic Rb1 deletion is sufficient to accelerate monoclonal gammopathy in wildtype mice and MM progression in Vk*MYC mice.", "kind": "causal"}],
        [], materials["32954360"], myeloma_mouse, allowed=("insufficient",), claims_should_refute=["C1"])
    dual_scope = {"species": None, "system": "reported preclinical dual-antigen multiple myeloma models", "disease": "multiple myeloma"}
    add("BCMA-GPRC5D", "双 CAR 结构和细胞相互作用", "摘要报告双阳性疾病中 bicistronic 优于 pooled 及同细胞双 CAR 增强相互作用；是否足以证明全部效应中介须保留摘要设计细节限制。",
        "BCMA/GPRC5D bicistronic CAR 相比 pooled CAR 控制双抗原病灶的机制证据到哪一层？", "同细胞表达两个 CAR 增强与靶细胞的相互作用，这参与解释 bicistronic 在双抗原病灶中的优势。",
        [{"id": "C1", "text": "The bicistronic approach is more efficacious than the pooled approach for dual-antigen-expressing disease at the tested subtherapeutic doses.", "kind": "causal"},
         {"id": "C2", "text": "Expressing the two CARs on a single cell enhances CAR T-cell/target-cell interaction strength.", "kind": "causal"}],
        [{"id": "J1", "text": "Enhanced CAR T-cell/target-cell interaction strength contributes to the bicistronic efficacy advantage in the tested dual-antigen-expressing model.", "claim_ids": ["C1", "C2"]}],
        materials["33089218"], dual_scope, allowed=("supported", "partial"))
    add("BCMA-GPRC5D", "误认 single-stalk 普遍最优", "BCMA 阴性疾病中 bicistronic 和 pooled 最有效；双阳性疾病中 bicistronic 更优。不同构型不能仅因同属双靶就互换。",
        "该比较是否支持 dual-scFv single-stalk 在 BCMA 阴性和双阳性病灶中均优于其他构型？", "dual-scFv single-stalk CAR 在 BCMA 阴性和双阳性病灶中均优于 bicistronic 及 pooled CAR。",
        [{"id": "C1", "text": "The dual-scFv single-stalk design outperforms both bicistronic and pooled designs in BCMA-negative and dual-antigen-expressing disease.", "kind": "reported"}],
        [], materials["33089218"], dual_scope, allowed=("insufficient",), claims_should_refute=["C1"])
    add("BCMA-CD229", "低 BCMA 与异质性模型中的双靶效应", "摘要支持所测 BCMA 低表达或缺失克隆模型中的作用，不据此宣称所有患者均获益。",
        "BCMA/CD229 bicistronic CAR 在 BCMA 低表达或含缺失克隆的实验模型中是否有作用？", "BCMA/CD229 bicistronic CAR 在所测低 BCMA 及含 BCMA 缺失克隆的 MM 模型中保持抗瘤效应。",
        [{"id": "C1", "text": "The BCMA/CD229 bicistronic CAR demonstrates efficacy in the reported low-BCMA and heterogeneous myeloma models containing BCMA-deleted clones.", "kind": "causal"}],
        [], materials["40576564"], dual_scope)
    add("BCMA-CD229", "无自相残杀不等于无脱瘤毒性", "摘要同时报告未见 CAR T 自相残杀及有限清除非激活 T 细胞，两项不能合并为证明无任何脱瘤毒性。",
        "未观察到 CAR T fratricide 是否就证明该 CD229 方案无 on-target off-tumour 作用？", "未见 CAR T 自相残杀，且该方案完全不清除非激活 T 细胞，因此不存在 on-target off-tumour 作用。",
        [{"id": "C1", "text": "No fratricide among CAR T cells was observed in the reported experiments.", "kind": "reported"},
         {"id": "C2", "text": "No elimination of nonactivated T cells was observed with the CD229 CAR system.", "kind": "reported"}],
        [{"id": "J1", "text": "The absence of CAR T-cell fratricide establishes absence of on-target off-tumour toxicity.", "claim_ids": ["C1", "C2"]}],
        materials["40576564"], dual_scope, allowed=("partial",), claims_should_support=["C1"], claims_should_refute=["C2"], joins_must_not_support=["J1"])
    b7_scope = {"species": "human", "system": "in vitro co-cultures of ENG T cells and the reported MM cell lines", "disease": "multiple myeloma"}
    add("B7H3-ENG", "所测共培养杀伤结果", "摘要报告工程 T 细胞杀伤所测 B7-H3 阳性 MM 细胞；支持该实验结果，不自动证明细胞系间差异完全由 B7-H3 决定。",
        "αB7-H3-αCD3 ENG T 细胞是否能杀伤本文测试的 B7-H3 阳性 MM 细胞？", "αB7-H3-αCD3 ENG T 细胞在所测共培养条件下杀伤 B7-H3 阳性 MM 细胞。",
        [{"id": "C1", "text": "The engineered αB7-H3-αCD3 ENG T cells exert cytotoxic activity against the tested B7-H3-positive MM cell lines in vitro.", "kind": "causal"}],
        [], materials["39806405"], b7_scope)
    b7_ft, b7_locations = _fulltext(directory, "PMC11727291", "39806405", materials["39806405"]["title"],
        [("Sec16", 0), ("Sec18", 0), ("Sec20", 0), ("Sec24", 9)])
    add("B7H3-ENG", "跨细胞系差异不能证明唯一抗原因果", "B7-H3 不同细胞系及阴性 SupT-1 并非同基因背景抗原消融/救援；全文补充结合和杀伤证据也不证明只有抗原丰度决定全部差异。",
        "这些实验是否证明细胞系间杀伤差异完全且唯一由 B7-H3 丰度引起？", "ENG T 细胞可杀伤 MM 细胞；不同靶细胞系的杀伤差异完全且唯一由 B7-H3 丰度决定。",
        [{"id": "C1", "text": "The engineered ENG T cells kill the tested B7-H3-positive MM cell lines.", "kind": "causal"},
         {"id": "C2", "text": "The tested target cell lines differ in B7-H3 expression.", "kind": "reported"}],
        [{"id": "J1", "text": "B7-H3 abundance alone causally explains all cytotoxicity differences between the tested cell lines, without contributions from other cell-line properties.", "claim_ids": ["C1", "C2"]}],
        b7_ft, b7_scope, allowed=("partial",), claims_should_support=["C1", "C2"], joins_must_not_support=["J1"],
        fulltext_supplement={"pmcid": "PMC11727291", "locations": b7_locations}, gap_kind="cross_cell_line_confounding")

    # These are cross-study transfer tests, not three further independent studies.
    cross_sources = [deepcopy(materials[pmid]) for pmid in ("38744277", "38123696", "33089218")]
    for number, material in enumerate(cross_sources, 1):
        material["id"] = f"M{number}"
    cross_claims = [
        {"id": "C1", "text": "Soluble rhIL-7-hyFc expands PD-1-negative bystander CD8 tumour-infiltrating lymphocytes in the reported solid-tumour study.", "kind": "causal"},
        {"id": "C2", "text": "Co-delivery of the membrane-bound lenalidomide-degradable IL-7 fusion with a CAR enhances antigen-dependent CAR T-cell proliferative capacity in the reported engineered-cell system.", "kind": "causal"},
        {"id": "C3", "text": "Bicistronic BCMA/GPRC5D CAR T cells are more efficacious than pooled single-target CAR T cells against dual-antigen-expressing myeloma at the tested subtherapeutic doses.", "kind": "causal"},
    ]
    cross_joins = [
        {"id": "J1", "text": "The effect of soluble rhIL-7-hyFc on bystander CD8 TILs transfers to BCMA/GPRC5D CAR T cells and reproduces the functional benefit of cell-bound lenalidomide-degradable IL-7 despite the different cytokine presentation and cell identity.", "claim_ids": ["C1", "C2"]},
        {"id": "J2", "text": "Combining soluble rhIL-7-hyFc with BCMA/GPRC5D bicistronic CAR T cells improves myeloma tumour control in the proposed combination regimen, transferring the separate IL-7/TCE and membrane-IL-7 CAR results to this myeloma CAR system.", "claim_ids": ["C1", "C2", "C3"]},
    ]
    cross_id = add("cross-study-transfer", "三篇局部事实能否支持给药和治疗体系迁移",
        "三篇原摘要分别支持旁观者 TIL 扩增、膜结合 IL7-CAR 增殖和双 CAR 的 MM 效应；没有实验建立所声明的可溶性 IL7 与 BCMA/GPRC5D CAR 联用迁移。应为局部支持加连接待补，不应捏造相反结果。",
        "这三篇研究能否证明可溶性 rhIL-7-hyFc 会像膜结合可降解 IL-7 一样增强 BCMA/GPRC5D CAR T 在多发性骨髓瘤中的疗效？",
        "可溶性 rhIL-7-hyFc 在原实体瘤研究中扩增 PD-1 阴性旁观者 CD8 TIL；原工程系统的膜结合可降解 IL-7 增强抗原依赖 CAR T 增殖；原 MM 模型中 BCMA/GPRC5D bicistronic CAR 优于 pooled CAR。因此，可溶性 rhIL-7-hyFc 对旁观者细胞的作用可迁移为膜结合 IL7 对 CAR 细胞的益处，并增强 BCMA/GPRC5D CAR T 的 MM 肿瘤控制。",
        cross_claims, cross_joins, cross_sources[0], {"species": None, "system": None, "disease": None},
        allowed=("partial",), claims_should_support=["C1", "C2", "C3"], joins_must_not_support=["J1", "J2"],
        joins_must_not_refute=["J1", "J2"], evaluation_set="cross_study_stability", controlled=True,
        expected_distinct_source_groups=3, gap_kind="unvalidated_cross_study_transport")
    cases[-1]["materials"] = deepcopy(cross_sources)
    base = deepcopy(cases[-1])
    for label, change in [("同一跨研究问题仅改变材料顺序", "order"),
                           ("同一跨研究问题加入同源重复摘要", "duplicate")]:
        case = deepcopy(base)
        case["id"] = f"Q{len(cases) + 1:03d}"
        if change == "order":
            case["materials"] = [case["materials"][2], case["materials"][0], case["materials"][1]]
        else:
            duplicate = deepcopy(case["materials"][0])
            duplicate["id"] = "M4"
            case["materials"].append(duplicate)
        cases.append(case)
        expectations[case["id"]] = {**deepcopy(expectations[cross_id]), "label": label,
            "paired_case_id": cross_id, "controlled_change": change,
            "reason": "相对于 Q023 仅改变原文顺序或重复同源摘要，三项局部医学主张和迁移义务均保持相同。重复摘要不增加来源组，更不证明独立队列。",
            "must_preserve_verdict_vs": cross_id, "must_not_increase_source_groups_vs": cross_id}

    return cases, expectations
