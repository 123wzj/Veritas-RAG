# -*- coding: utf-8 -*-
"""Run end-to-end RAGAS evaluation against the project's Agentic RAG graph.

Input is a JSON or JSONL file with at least:
  {"question": "...", "ground_truth": "..."}

Accepted reference field aliases:
  ground_truth, reference, expected_answer, answer
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT, BACKEND_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


DEFAULT_REPORT_DIR = PROJECT_ROOT / "data" / "evaluation" / "reports"
DEFAULT_METRICS = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "answer_correctness",
)
REFERENCE_METRICS = {"context_precision", "context_recall", "answer_correctness"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate generated RAG answers with RAGAS using the real Agentic RAG pipeline."
    )
    parser.add_argument("--input", type=Path, required=True, help="JSONL/JSON evaluation samples.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--kb-id", type=int, required=True)
    parser.add_argument("--user-id", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--web-enabled", action="store_true")
    parser.add_argument(
        "--metrics",
        type=str,
        default=",".join(DEFAULT_METRICS),
        help="Comma-separated RAGAS metrics. Default: faithfulness,answer_relevancy,context_precision,context_recall,answer_correctness",
    )
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help="Only run the RAG pipeline and save generated answers; do not call RAGAS.",
    )
    return parser.parse_args()


async def main_async() -> None:
    args = parse_args()
    cases = load_cases(args.input)
    if args.limit is not None:
        cases = cases[: max(args.limit, 0)]
    if not cases:
        raise RuntimeError("No evaluation cases loaded.")

    started = time.perf_counter()
    generated_rows: List[Dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        row = await run_case(case, args)
        generated_rows.append(row)
        status = "ok" if not row.get("error") else "failed"
        print(f"[{index}/{len(cases)}] {status} id={row.get('id') or index}")

    ragas_report: Dict[str, Any] = {}
    if not args.collect_only:
        usable_rows = [
            row for row in generated_rows
            if not row.get("error") and row.get("answer") and row.get("contexts")
        ]
        if not usable_rows:
            raise RuntimeError("No successful rows with answer and contexts are available for RAGAS.")
        ragas_report = run_ragas(usable_rows, parse_metric_names(args.metrics))

    report = {
        "config": {
            "input": str(args.input),
            "kb_id": args.kb_id,
            "user_id": args.user_id,
            "limit": args.limit,
            "top_k": args.top_k,
            "web_enabled": args.web_enabled,
            "metrics": parse_metric_names(args.metrics),
            "collect_only": args.collect_only,
        },
        "seconds": round(time.perf_counter() - started, 3),
        "sample_count": len(generated_rows),
        "success_count": sum(1 for row in generated_rows if not row.get("error")),
        "ragas": ragas_report,
        "samples": generated_rows,
    }
    output_path = save_report(report, args.output_dir, args.kb_id)
    print(f"report={output_path}")
    if ragas_report.get("summary"):
        print("summary")
        for key, value in ragas_report["summary"].items():
            if isinstance(value, (int, float)):
                print(f"- {key}: {value:.4f}")


def load_cases(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    if path.suffix.lower() == ".jsonl":
        cases = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        raw = json.loads(text)
        cases = raw if isinstance(raw, list) else raw.get("samples", [])

    normalized = []
    for index, case in enumerate(cases, start=1):
        question = case.get("question") or case.get("user_input") or case.get("query")
        if not question:
            raise ValueError(f"Case #{index} is missing question/user_input/query.")
        reference = pick_reference(case)
        normalized.append({
            "id": case.get("id") or f"case_{index}",
            "question": str(question),
            "reference": reference,
            "gold_doc_ids": case.get("gold_doc_ids") or [],
            "metadata": {
                key: value
                for key, value in case.items()
                if key not in {"question", "user_input", "query", "ground_truth", "reference", "expected_answer", "answer"}
            },
        })
    return normalized


def pick_reference(case: Dict[str, Any]) -> Optional[str]:
    value = (
        case.get("ground_truth")
        or case.get("reference")
        or case.get("expected_answer")
        or case.get("ground_truths")
    )
    if isinstance(value, list):
        return "\n".join(str(item) for item in value if item is not None)
    if value is None:
        return None
    return str(value)


async def run_case(case: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    from graph.graph import run_agentic_rag

    started = time.perf_counter()
    try:
        final_state: Optional[Dict[str, Any]] = None
        async for state in run_agentic_rag(
            query=case["question"],
            user_id=args.user_id,
            kb_id=args.kb_id,
            session_id=None,
            web_enabled=args.web_enabled,
            stream_events=False,
            top_k=args.top_k,
        ):
            if isinstance(state, dict):
                final_state = state

        if not final_state:
            raise RuntimeError("RAG graph did not return a final state.")

        contexts = extract_contexts(final_state)
        citations = final_state.get("citations") or []
        answer = final_state.get("final_answer") or ""
        return {
            "id": case["id"],
            "question": case["question"],
            "answer": answer,
            "contexts": contexts,
            "reference": case.get("reference"),
            "gold_doc_ids": case.get("gold_doc_ids") or [],
            "citations": citations,
            "selected_evidence": final_state.get("selected_evidence") or [],
            "confidence": final_state.get("confidence", 0.0),
            "verification": final_state.get("verification"),
            "route_type": final_state.get("route_type"),
            "evidence_grade": final_state.get("evidence_grade"),
            "latency_ms": int((time.perf_counter() - started) * 1000),
            "metadata": case.get("metadata") or {},
        }
    except Exception as exc:
        return {
            "id": case["id"],
            "question": case["question"],
            "answer": "",
            "contexts": [],
            "reference": case.get("reference"),
            "gold_doc_ids": case.get("gold_doc_ids") or [],
            "error": str(exc),
            "latency_ms": int((time.perf_counter() - started) * 1000),
            "metadata": case.get("metadata") or {},
        }


def extract_contexts(state: Dict[str, Any]) -> List[str]:
    contexts: List[str] = []
    for evidence in state.get("selected_evidence") or []:
        text = (
            evidence.get("support_snippet")
            or evidence.get("snippet")
            or evidence.get("parent_content")
            or evidence.get("content")
            or ""
        )
        add_unique_context(contexts, text)

    if not contexts:
        for citation in state.get("citations") or []:
            add_unique_context(contexts, citation.get("snippet") or "")

    for plan in state.get("sub_query_plans") or []:
        for evidence in plan.get("selected_evidence") or []:
            text = evidence.get("support_snippet") or evidence.get("snippet") or ""
            add_unique_context(contexts, text)

    return contexts


def add_unique_context(contexts: List[str], text: str) -> None:
    cleaned = " ".join((text or "").split())
    if cleaned and cleaned not in contexts:
        contexts.append(cleaned)


def parse_metric_names(raw: str) -> List[str]:
    names = []
    for item in raw.split(","):
        name = item.strip().lower().replace("-", "_")
        if name and name not in names:
            names.append(name)
    return names


def run_ragas(rows: List[Dict[str, Any]], metric_names: Sequence[str]) -> Dict[str, Any]:
    has_reference = all(bool(row.get("reference")) for row in rows)
    selected_names = [
        name for name in metric_names
        if has_reference or name not in REFERENCE_METRICS
    ]
    skipped = [
        name for name in metric_names
        if name not in selected_names
    ]

    try:
        result = run_ragas_current(rows, selected_names, has_reference)
        api = "current"
    except Exception as current_exc:
        try:
            result = run_ragas_legacy(rows, selected_names, has_reference)
            api = "legacy"
        except Exception as legacy_exc:
            raise RuntimeError(
                "RAGAS evaluation failed with both current and legacy APIs. "
                f"Current error: {current_exc}. Legacy error: {legacy_exc}"
            ) from legacy_exc

    report = ragas_result_to_report(result)
    report["api"] = api
    report["requested_metrics"] = list(metric_names)
    report["used_metrics"] = selected_names
    report["skipped_metrics"] = skipped
    if skipped:
        report["skip_reason"] = "Reference-based metrics require every row to provide a reference/ground_truth."
    return report


def run_ragas_current(rows: List[Dict[str, Any]], metric_names: Sequence[str], has_reference: bool):
    from ragas import evaluate

    try:
        from ragas import EvaluationDataset
    except ImportError:
        from ragas.dataset_schema import EvaluationDataset
    from ragas.dataset_schema import SingleTurnSample

    metrics = load_current_metrics(metric_names, has_reference)
    samples = []
    for row in rows:
        kwargs = {
            "user_input": row["question"],
            "response": row["answer"],
            "retrieved_contexts": row["contexts"],
        }
        if row.get("reference"):
            kwargs["reference"] = row["reference"]
        samples.append(SingleTurnSample(**kwargs))

    dataset = EvaluationDataset(samples=samples)
    return evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=get_ragas_llm(),
        embeddings=get_ragas_embeddings(),
        show_progress=True,
    )


def run_ragas_legacy(rows: List[Dict[str, Any]], metric_names: Sequence[str], has_reference: bool):
    from datasets import Dataset
    from ragas import evaluate

    metrics = load_legacy_metrics(metric_names, has_reference)
    dataset_rows = []
    for row in rows:
        item = {
            "question": row["question"],
            "answer": row["answer"],
            "contexts": row["contexts"],
        }
        if row.get("reference"):
            item["ground_truth"] = row["reference"]
            item["ground_truths"] = [row["reference"]]
        dataset_rows.append(item)
    dataset = Dataset.from_list(dataset_rows)
    return evaluate(
        dataset,
        metrics=metrics,
        llm=get_ragas_llm(),
        embeddings=get_ragas_embeddings(),
        raise_exceptions=False,
    )


def load_current_metrics(metric_names: Sequence[str], has_reference: bool) -> List[Any]:
    from ragas import metrics as ragas_metrics

    class_candidates = {
        "faithfulness": ["Faithfulness"],
        "answer_relevancy": ["ResponseRelevancy", "AnswerRelevancy"],
        "context_precision": ["LLMContextPrecisionWithReference", "ContextPrecision"],
        "context_recall": ["LLMContextRecall", "ContextRecall"],
        "answer_correctness": ["AnswerCorrectness", "FactualCorrectness"],
    }
    loaded = []
    missing = []
    for name in metric_names:
        if name in REFERENCE_METRICS and not has_reference:
            continue
        metric = instantiate_first_available(ragas_metrics, class_candidates.get(name, []))
        if metric is None:
            missing.append(name)
        else:
            loaded.append(metric)
    if not loaded:
        raise RuntimeError(f"No current RAGAS metrics could be loaded. Missing={missing}")
    if missing:
        print(f"warning: skipped unsupported current RAGAS metrics: {missing}")
    return loaded


def instantiate_first_available(module: Any, names: Iterable[str]) -> Optional[Any]:
    for name in names:
        candidate = getattr(module, name, None)
        if candidate is None:
            continue
        try:
            return candidate()
        except TypeError:
            return candidate
    return None


def load_legacy_metrics(metric_names: Sequence[str], has_reference: bool) -> List[Any]:
    from ragas import metrics as ragas_metrics

    loaded = []
    missing = []
    aliases = {
        "faithfulness": ["faithfulness"],
        "answer_relevancy": ["answer_relevancy", "answer_relevance"],
        "context_precision": ["context_precision"],
        "context_recall": ["context_recall"],
        "answer_correctness": ["answer_correctness"],
    }
    for name in metric_names:
        if name in REFERENCE_METRICS and not has_reference:
            continue
        metric = first_attr(ragas_metrics, aliases.get(name, []))
        if metric is None:
            missing.append(name)
        else:
            loaded.append(metric)
    if not loaded:
        raise RuntimeError(f"No legacy RAGAS metrics could be loaded. Missing={missing}")
    if missing:
        print(f"warning: skipped unsupported legacy RAGAS metrics: {missing}")
    return loaded


def first_attr(module: Any, names: Iterable[str]) -> Optional[Any]:
    for name in names:
        candidate = getattr(module, name, None)
        if candidate is not None:
            return candidate
    return None


def get_ragas_llm() -> Any:
    from graph.llm_factory import get_llm

    llm = get_llm()
    try:
        from ragas.llms import LangchainLLMWrapper

        return LangchainLLMWrapper(llm)
    except Exception:
        return llm


def get_ragas_embeddings() -> Any:
    from embeddings.embeddings import get_embeddings

    embeddings = ProjectEmbeddingsAdapter(get_embeddings())
    try:
        from ragas.embeddings import LangchainEmbeddingsWrapper

        return LangchainEmbeddingsWrapper(embeddings)
    except Exception:
        return embeddings


class ProjectEmbeddingsAdapter:
    """Tiny LangChain-compatible wrapper around the project's embedding adapter."""

    def __init__(self, embeddings: Any):
        self._embeddings = embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._embeddings.embed_query(text)

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        return await asyncio.to_thread(self.embed_documents, texts)

    async def aembed_query(self, text: str) -> List[float]:
        return await asyncio.to_thread(self.embed_query, text)


def ragas_result_to_report(result: Any) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    summary: Dict[str, float] = {}

    if hasattr(result, "to_pandas"):
        df = result.to_pandas()
        rows = json.loads(df.to_json(orient="records", force_ascii=False))
        numeric_cols = [
            column for column in df.columns
            if str(df[column].dtype).startswith(("float", "int"))
        ]
        summary = {
            column: round(float(df[column].mean()), 6)
            for column in numeric_cols
        }
    elif hasattr(result, "to_dict"):
        raw = result.to_dict()
        if isinstance(raw, dict):
            summary = {
                key: round(float(value), 6)
                for key, value in raw.items()
                if isinstance(value, (int, float))
            }
    elif isinstance(result, dict):
        summary = {
            key: round(float(value), 6)
            for key, value in result.items()
            if isinstance(value, (int, float))
        }

    return {"summary": summary, "rows": rows}


def save_report(report: Dict[str, Any], output_dir: Path, kb_id: int) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"ragas_answer_eval_kb{kb_id}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


if __name__ == "__main__":
    asyncio.run(main_async())
