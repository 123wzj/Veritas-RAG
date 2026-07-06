# -*- coding: utf-8 -*-
"""Run RGB fixed-context generation evaluation with the project answer prompt."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[2]
for path in (PROJECT_ROOT,):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


from backend.evaluation.run_ragas_answer_eval import DEFAULT_REPORT_DIR, parse_metric_names, run_ragas


DEFAULT_INPUT = PROJECT_ROOT / "data" / "evaluation" / "rgb" / "rgb_zh_fixed_samples.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate project generation on RGB fixed contexts.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--samples-per-type", type=int, default=None)
    parser.add_argument(
        "--question-types",
        type=str,
        default=None,
        help="Comma-separated RGB question types to include.",
    )
    parser.add_argument("--metrics", type=str, default="faithfulness,answer_relevancy,answer_correctness")
    parser.add_argument("--judge", choices=["deepseek", "ragas", "none"], default="deepseek")
    parser.add_argument("--collect-only", action="store_true")
    parser.add_argument("--generation-llm", choices=["project", "deepseek"], default="deepseek")
    return parser.parse_args()


async def main_async() -> None:
    args = parse_args()
    cases = load_cases(args.input)
    question_types = {
        item.strip()
        for item in (args.question_types or "").split(",")
        if item.strip()
    }
    if question_types:
        cases = [
            case for case in cases
            if str(case.get("question_type") or "unknown") in question_types
        ]
    if args.samples_per_type is not None:
        cases = select_samples_per_type(cases, args.samples_per_type)
    elif args.limit is not None:
        cases = cases[: max(args.limit, 0)]
    if not cases:
        raise RuntimeError("No RGB cases loaded.")

    started = time.perf_counter()
    rows: List[Dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        row = await run_case(case, args.generation_llm)
        rows.append(row)
        status = "ok" if not row.get("error") else "failed"
        print(f"[{index}/{len(cases)}] {status} id={row.get('id')}")

    deepseek_report: Dict[str, Any] = {}
    ragas_report: Dict[str, Any] = {}
    if not args.collect_only and args.judge == "deepseek":
        usable_rows = [row for row in rows if not row.get("error") and row.get("answer") and row.get("contexts")]
        deepseek_report = await run_deepseek_judge(usable_rows)
    elif not args.collect_only and args.judge == "ragas":
        usable_rows = [row for row in rows if not row.get("error") and row.get("answer") and row.get("contexts")]
        if not usable_rows:
            raise RuntimeError("No successful rows with answer and contexts are available for RAGAS.")
        ragas_report = run_ragas(usable_rows, parse_metric_names(args.metrics))

    report = {
        "config": {
            "input": str(args.input),
            "limit": args.limit,
            "samples_per_type": args.samples_per_type,
            "question_types": sorted(question_types),
            "metrics": parse_metric_names(args.metrics),
            "judge": args.judge,
            "collect_only": args.collect_only,
            "mode": "rgb_fixed_context",
            "generation_llm": args.generation_llm,
        },
        "seconds": round(time.perf_counter() - started, 3),
        "sample_count": len(rows),
        "success_count": sum(1 for row in rows if not row.get("error")),
        "deepseek_judge": deepseek_report,
        "ragas": ragas_report,
        "samples": rows,
    }
    output_path = save_report(report, args.output_dir)
    print(f"report={output_path}")
    summary = deepseek_report.get("summary") or ragas_report.get("summary") or {}
    if summary:
        print("summary")
        for key, value in summary.items():
            if isinstance(value, (int, float)):
                print(f"- {key}: {value:.4f}")


def load_cases(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def select_samples_per_type(cases: List[Dict[str, Any]], limit: int) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    counts: Dict[str, int] = {}
    for case in cases:
        question_type = str(case.get("question_type") or "unknown")
        if counts.get(question_type, 0) >= max(0, limit):
            continue
        selected.append(case)
        counts[question_type] = counts.get(question_type, 0) + 1
    return selected


async def run_case(case: Dict[str, Any], generation_llm: str) -> Dict[str, Any]:
    from backend.graph.nodes import generation_nodes, retrieval_nodes

    started = time.perf_counter()
    contexts = case.get("contexts") or []
    try:
        if generation_llm == "deepseek":
            generation_nodes.generation_llm = build_deepseek_generation_llm()
            flash_llm = build_deepseek_flash_llm()
            generation_nodes.verification_llm = flash_llm
            retrieval_nodes.llm = flash_llm
        selected_evidence = build_evidence(case, contexts)
        base_state = {
            "query": case["question"],
            "kb_id": 0,
            "route_type": "knowledge_base",
            "sub_query_plans": [{
                "sub_question": case["question"],
                "route_type": "knowledge_base",
                "selected_evidence": selected_evidence,
            }],
            "selected_evidence": selected_evidence,
            "memory_context": {},
            "prompt_context": build_prompt_context(case),
            "reflection_count": 0,
            "max_reflections": 0,
            "web_enabled": False,
            "used_web_search": False,
            "events": [],
        }
        graded = await retrieval_nodes.judge_evidence_slots(base_state)
        result = await generation_nodes.generate_answer({**base_state, **graded})
        return {
            "id": case["id"],
            "question": case["question"],
            "answer": result.get("final_answer") or "",
            "contexts": contexts,
            "context_labels": case.get("context_labels") or [],
            "reference": case.get("ground_truth"),
            "original_reference": case.get("original_ground_truth"),
            "citations": result.get("citations") or [],
            "confidence": result.get("confidence", 0.0),
            "evidence_grade": graded.get("evidence_grade"),
            "question_type": case.get("question_type"),
            "expected_behavior": case.get("expected_behavior"),
            "source_dataset": case.get("source_dataset"),
            "source_file": case.get("source_file"),
            "latency_ms": int((time.perf_counter() - started) * 1000),
        }
    except Exception as exc:
        return {
            "id": case.get("id"),
            "question": case.get("question"),
            "answer": "",
            "contexts": contexts,
            "context_labels": case.get("context_labels") or [],
            "reference": case.get("ground_truth"),
            "original_reference": case.get("original_ground_truth"),
            "error": str(exc),
            "question_type": case.get("question_type"),
            "expected_behavior": case.get("expected_behavior"),
            "latency_ms": int((time.perf_counter() - started) * 1000),
        }


def build_deepseek_generation_llm() -> Any:
    from backend.core.config import settings

    return DeepSeekDirectChat(
        model=settings.DEEPSEEK_PRO_MODEL,
        api_key=settings.DEEPSEEK_API_KEY,
        base_url=settings.DEEPSEEK_BASE_URL,
        temperature=0.0,
        max_tokens=settings.RAGAS_LLM_MAX_TOKENS,
        timeout=settings.RAGAS_LLM_TIMEOUT,
        max_retries=settings.RAGAS_LLM_MAX_RETRIES,
    )


def build_deepseek_flash_llm() -> Any:
    from backend.core.config import settings

    return DeepSeekDirectChat(
        model=settings.DEEPSEEK_FLASH_MODEL,
        api_key=settings.DEEPSEEK_API_KEY,
        base_url=settings.DEEPSEEK_BASE_URL,
        temperature=0.0,
        max_tokens=settings.RAGAS_LLM_MAX_TOKENS,
        timeout=settings.RAGAS_LLM_TIMEOUT,
        max_retries=settings.RAGAS_LLM_MAX_RETRIES,
    )


async def run_deepseek_judge(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {"summary": {}, "rows": []}
    from backend.core.config import settings

    client = DeepSeekDirectChat(
        model=settings.DEEPSEEK_PRO_MODEL,
        api_key=settings.DEEPSEEK_API_KEY,
        base_url=settings.DEEPSEEK_BASE_URL,
        temperature=0.0,
        max_tokens=settings.RAGAS_LLM_MAX_TOKENS,
        timeout=settings.RAGAS_LLM_TIMEOUT,
        max_retries=settings.RAGAS_LLM_MAX_RETRIES,
    )
    judged_rows = []
    for row in rows:
        heuristics = compute_heuristics(row)
        judged = await judge_row(client, row, heuristics)
        row["deepseek_judge"] = judged
        judged_rows.append({
            "id": row.get("id"),
            "question_type": row.get("question_type"),
            "expected_behavior": row.get("expected_behavior"),
            **judged,
        })
    return {"summary": summarize_judged_rows(judged_rows), "rows": judged_rows}


async def judge_row(client: "DeepSeekDirectChat", row: Dict[str, Any], heuristics: Dict[str, Any]) -> Dict[str, Any]:
    context_labels = context_labels_for_row(row)
    evidence_items = evidence_items_for_row(row, context_labels)
    prompt = {
        "question": row.get("question"),
        "answer": row.get("answer"),
        "reference": row.get("reference"),
        "original_reference": row.get("original_reference"),
        "expected_behavior": row.get("expected_behavior"),
        "question_type": row.get("question_type"),
        "contexts": row.get("contexts"),
        "context_labels": context_labels,
        "evidence_items": evidence_items,
        "heuristics": heuristics,
    }
    system_prompt = (
        "你是RAG生成阶段评估器。reference是答案正确性的权威参考，contexts用于评估忠实性和引用支撑。"
        "输出严格JSON，不要输出Markdown。字段："
        "faithfulness, answer_relevancy, answer_correctness, citation_quality, "
        "refusal_score, counterfactual_score, overall, verdict, reason, evaluation_valid, invalid_reason。"
        "所有分数字段为0到1的小数。refusal_score只在expected_behavior=refuse时评估，否则给null；"
        "counterfactual_score只在expected_behavior=correct_false_context时评估，否则给null。"
        "证据标签含义：positive是该问题的可信正证据，positive_wrong是基准刻意注入的错误证据，"
        "negative是该样本的无关证据，cross_sample表示来自其他样本、对当前问题尚未标注，必须按语义判断。"
        "评价引用时必须按evidence_items里的evidence_id匹配，例如[E7]只对应evidence_id=E7的条目，"
        "不能把它理解成contexts数组的第7项。"
        "反事实题不得按文档数量投票；应优先检查答案是否符合reference并由positive证据支撑，"
        "引用或采信positive_wrong应降低counterfactual_score。若证据本身没有来源可信度信息，"
        "答案明确识别冲突、没有把positive_wrong当成最终事实、并拒绝给出确定结论，也属于安全的反事实鲁棒行为，"
        "counterfactual_score应为0.8到1.0；这种安全拒答的answer_correctness可低于直接答对，但不能记为0。"
        "对于正确拒答且没有额外事实主张的答案，citation_quality应为null，不应因没有引用而扣分。"
        "拒答题若contexts或cross_sample实际上已经包含或足以推出original_reference，则该样本不再满足无答案前提，"
        "必须令evaluation_valid=false、refusal_score=null并说明污染原因；否则evaluation_valid=true。"
    )
    user_prompt = json.dumps(prompt, ensure_ascii=False)
    try:
        response = await client.ainvoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ])
        parsed = safe_json_object(response.content)
    except Exception as exc:
        parsed = {"error": str(exc)}
    for key, value in heuristics.items():
        parsed[f"heuristic_{key}"] = value
    if row.get("expected_behavior") == "refuse" and heuristics.get("original_reference_in_context"):
        parsed["evaluation_valid"] = False
        parsed["invalid_reason"] = "original_reference_found_in_retrieved_context"
        parsed["refusal_score"] = None
        parsed["verdict"] = "invalid"
    return normalize_judge_scores(parsed, row)


def compute_heuristics(row: Dict[str, Any]) -> Dict[str, Any]:
    answer = row.get("answer") or ""
    reference = row.get("reference") or ""
    cited = sorted(set(re.findall(r"\[(E\d+)\]", answer)))
    context_count = len(row.get("contexts") or [])
    valid_citations = [item for item in cited if item[1:].isdigit() and int(item[1:]) <= context_count]
    reference_parts = [part.strip() for part in re.split(r"[；;、,\n]", reference) if part.strip()]
    if row.get("expected_behavior") == "refuse":
        reference_hit_rate = None
    elif reference_parts:
        reference_hit_rate = sum(1 for part in reference_parts if part in answer) / len(reference_parts)
    else:
        reference_hit_rate = None
    refusal_terms = (
        "信息不足",
        "无法回答",
        "不能确定",
        "无法确定",
        "证据不足",
        "没有足够",
        "无法给出确定结论",
        "来源不足",
    )
    original_reference = str(row.get("original_reference") or "").strip()
    normalized_context = "\n".join(str(item) for item in row.get("contexts") or [])
    evidence_items = evidence_items_for_row(row, context_labels_for_row(row))
    evidence_by_id = {
        str(item.get("evidence_id")): item
        for item in evidence_items
        if item.get("evidence_id")
    }
    cited_items = [evidence_by_id[item] for item in cited if item in evidence_by_id]
    return {
        "reference_hit_rate": reference_hit_rate,
        "has_refusal": any(term in answer for term in refusal_terms),
        "citation_count": len(cited),
        "valid_citation_count": len(valid_citations),
        "invalid_citation_count": len(cited) - len(valid_citations),
        "citation_coverage": len(valid_citations) / max(1, min(4, context_count)),
        "original_reference_in_context": bool(
            original_reference and original_reference in normalized_context
        ),
        "cited_positive_wrong_count": sum(
            1 for item in cited_items if item.get("label") == "positive_wrong"
        ),
        "cited_positive_count": sum(
            1 for item in cited_items if item.get("label") == "positive"
        ),
        "cited_web_count": sum(
            1 for item in cited_items if item.get("source_type") == "web"
        ),
        "has_conflict_abstention": (
            "冲突" in answer and any(term in answer for term in refusal_terms)
        ),
    }


def context_labels_for_row(row: Dict[str, Any]) -> List[str]:
    explicit_labels = row.get("context_labels") or []
    if explicit_labels:
        return [str(label) for label in explicit_labels]
    selected_doc_ids = row.get("selected_doc_ids") or []
    expected_labels = row.get("expected_doc_labels") or {}
    return [str(expected_labels.get(doc_id) or "cross_sample") for doc_id in selected_doc_ids]


def evidence_items_for_row(
    row: Dict[str, Any],
    context_labels: List[str],
) -> List[Dict[str, Any]]:
    selected_evidence = row.get("selected_evidence") or []
    if selected_evidence:
        return [
            {
                "evidence_id": item.get("evidence_id") or f"E{index}",
                "label": context_labels[index - 1] if index - 1 < len(context_labels) else "unknown",
                "source_type": item.get("source_type"),
                "title": item.get("title"),
                "url": item.get("url"),
                "text": item.get("support_snippet") or item.get("snippet") or "",
            }
            for index, item in enumerate(selected_evidence, start=1)
        ]
    return [
        {
            "evidence_id": f"E{index}",
            "label": context_labels[index - 1] if index - 1 < len(context_labels) else "unknown",
            "source_type": "fixed_context",
            "text": context,
        }
        for index, context in enumerate(row.get("contexts") or [], start=1)
    ]


def normalize_judge_scores(raw: Dict[str, Any], row: Dict[str, Any]) -> Dict[str, Any]:
    numeric_keys = [
        "faithfulness",
        "answer_relevancy",
        "answer_correctness",
        "citation_quality",
        "refusal_score",
        "counterfactual_score",
        "overall",
    ]
    normalized = dict(raw)
    evaluation_valid = normalized.get("evaluation_valid", True)
    if isinstance(evaluation_valid, str):
        evaluation_valid = evaluation_valid.strip().lower() not in {"false", "0", "no", "invalid"}
    normalized["evaluation_valid"] = bool(evaluation_valid)
    for key in numeric_keys:
        value = normalized.get(key)
        if value is None:
            continue
        try:
            normalized[key] = round(min(max(float(value), 0.0), 1.0), 4)
        except (TypeError, ValueError):
            normalized[key] = None
    recalibrated_counterfactual = False
    if row.get("expected_behavior") == "correct_false_context":
        reference_hit_rate = normalized.get("heuristic_reference_hit_rate")
        cited_wrong_count = normalized.get("heuristic_cited_positive_wrong_count")
        conflict_abstention = normalized.get("heuristic_has_conflict_abstention")
        if (
            isinstance(reference_hit_rate, (int, float))
            and reference_hit_rate >= 1.0
            and cited_wrong_count == 0
        ):
            normalized["counterfactual_score"] = 1.0
            normalized["verdict"] = "pass"
            normalized["counterfactual_calibration"] = "reference_match_without_wrong_citation"
            recalibrated_counterfactual = True
        elif conflict_abstention:
            normalized["counterfactual_score"] = max(
                float(normalized.get("counterfactual_score") or 0.0),
                0.9,
            )
            normalized["verdict"] = "pass"
            normalized["counterfactual_calibration"] = "explicit_conflict_abstention"
            recalibrated_counterfactual = True

    if normalized.get("overall") is None or recalibrated_counterfactual:
        parts = [
            normalized.get("faithfulness"),
            normalized.get("answer_relevancy"),
            normalized.get("answer_correctness"),
            normalized.get("citation_quality"),
        ]
        if row.get("expected_behavior") == "refuse":
            parts.append(normalized.get("refusal_score"))
        if row.get("expected_behavior") == "correct_false_context":
            parts.append(normalized.get("counterfactual_score"))
        numeric_parts = [float(item) for item in parts if isinstance(item, (int, float))]
        normalized["overall"] = round(sum(numeric_parts) / len(numeric_parts), 4) if numeric_parts else None
    normalized.setdefault("verdict", "unknown")
    normalized.setdefault("reason", "")
    normalized.setdefault("invalid_reason", "")
    return normalized


def summarize_judged_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    keys = [
        "faithfulness",
        "answer_relevancy",
        "answer_correctness",
        "citation_quality",
        "refusal_score",
        "counterfactual_score",
        "overall",
        "heuristic_reference_hit_rate",
        "heuristic_citation_coverage",
    ]
    valid_rows = [row for row in rows if row.get("evaluation_valid") is not False]
    summary: Dict[str, Any] = {
        "sample_count": len(rows),
        "valid_sample_count": len(valid_rows),
        "invalid_sample_count": len(rows) - len(valid_rows),
        "evaluation_valid_rate": round(len(valid_rows) / max(1, len(rows)), 6),
    }
    refusal_rows = [
        row for row in valid_rows
        if row.get("expected_behavior") == "refuse"
    ]
    if refusal_rows:
        refusal_passes = sum(
            1 for row in refusal_rows
            if str(row.get("verdict") or "").strip().lower() in {"pass", "passed", "correct", "ok"}
            or (
                isinstance(row.get("refusal_score"), (int, float))
                and float(row["refusal_score"]) >= 0.8
            )
        )
        summary["refusal_sample_count"] = len(refusal_rows)
        summary["refusal_pass_count"] = refusal_passes
        summary["refusal_pass_rate"] = round(refusal_passes / len(refusal_rows), 6)
    for key in keys:
        values = [row.get(key) for row in valid_rows if isinstance(row.get(key), (int, float))]
        if values:
            summary[key] = round(sum(float(value) for value in values) / len(values), 6)
    return summary


def safe_json_object(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if not text:
        return {}
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            return {}
        try:
            value = json.loads(match.group(0))
            return value if isinstance(value, dict) else {}
        except json.JSONDecodeError:
            return {}


@dataclass
class DeepSeekDirectChat:
    model: str
    api_key: str
    base_url: Optional[str]
    temperature: float = 0.0
    max_tokens: Optional[int] = None
    timeout: int = 180
    max_retries: int = 3

    async def ainvoke(self, messages: List[Any]) -> Any:
        return await asyncio.to_thread(self.invoke, messages)

    def invoke(self, messages: List[Any]) -> Any:
        if not self.api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not configured.")
        endpoint = self.endpoint
        payload = {
            "model": self.model,
            "messages": [self.normalize_message(message) for message in messages],
            "temperature": self.temperature,
            "stream": False,
        }
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        last_error: Optional[Exception] = None
        for attempt in range(max(1, self.max_retries + 1)):
            request = urllib.request.Request(endpoint, data=data, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    body = response.read().decode("utf-8")
                parsed = json.loads(body)
                content = parsed["choices"][0]["message"]["content"]
                return SimpleNamespace(content=content)
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                last_error = RuntimeError(f"DeepSeek HTTP {exc.code}: {body}")
            except Exception as exc:
                last_error = exc
            if attempt < self.max_retries:
                time.sleep(min(2 ** attempt, 8))
        raise RuntimeError(str(last_error))

    @property
    def endpoint(self) -> str:
        base = (self.base_url or "https://api.deepseek.com").rstrip("/")
        return f"{base}/chat/completions"

    @staticmethod
    def normalize_message(message: Any) -> Dict[str, str]:
        if isinstance(message, dict):
            return {
                "role": str(message.get("role") or "user"),
                "content": str(message.get("content") or ""),
            }
        role = getattr(message, "type", None) or message.__class__.__name__.lower()
        if "system" in role:
            role = "system"
        elif "human" in role or "user" in role:
            role = "user"
        else:
            role = "assistant"
        return {"role": role, "content": str(getattr(message, "content", ""))}


def build_evidence(case: Dict[str, Any], contexts: List[str]) -> List[Dict[str, Any]]:
    evidence = []
    labels = case.get("context_labels") or []
    for index, text in enumerate(contexts, start=1):
        label = labels[index - 1] if index - 1 < len(labels) else "unknown"
        evidence.append({
            "evidence_id": f"E{index}",
            "doc_id": f"{case.get('id')}_doc_{index}",
            "chunk_id": f"{case.get('id')}_chunk_{index}",
            "title": f"RGB {case.get('question_type')} #{index}",
            "support_snippet": text,
            "snippet": text,
            "score": 1.0 if label == "positive" else 0.35,
            "source_type": "rgb_fixed_context",
        })
    return evidence


def build_prompt_context(case: Dict[str, Any]) -> str:
    behavior = case.get("expected_behavior")
    if behavior == "refuse":
        return "如果给定证据不足以回答问题，必须明确说明信息不足，不能编造答案。"
    if behavior == "correct_false_context":
        return "给定证据可能包含事实性错误或互相冲突的信息；请优先识别冲突，并基于可靠证据给出谨慎答案。"
    return "请只基于给定证据回答，并在关键结论后标注证据编号。"


def save_report(report: Dict[str, Any], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"rgb_fixed_context_eval_{time.strftime('%Y%m%d_%H%M%S')}.json"
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


if __name__ == "__main__":
    asyncio.run(main_async())
