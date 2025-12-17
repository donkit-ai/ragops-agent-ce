from __future__ import annotations

import re
import unicodedata
import warnings

# Suppress all warnings immediately, before any other imports
warnings.filterwarnings("ignore")
# Suppress warnings from importlib bootstrap (SWIG-related)
warnings.filterwarnings("ignore", category=DeprecationWarning, module="importlib._bootstrap")
# Suppress all DeprecationWarnings globally
warnings.simplefilter("ignore", DeprecationWarning)

import ast
import asyncio
import csv
import io
import json
import os
import time
from pathlib import Path
from typing import Any

import httpx
from fastmcp import FastMCP
from loguru import logger
from pydantic import BaseModel, Field

# Configure loguru to write to file
_log_level = os.getenv("RAGOPS_LOG_LEVEL", "CRITICAL").upper()
if _log_level == "DEBUG":
    LOG_DIR = Path(
        os.getenv(
            "RAGOPS_LOG_DIR",
            Path(__file__).resolve().parent / "logs",
        )
    )
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    LOG_FILE = LOG_DIR / "evaluation.log"
    logger.add(LOG_FILE, rotation="10 MB", retention="7 days", level="DEBUG")
    logger.info(f"Logging to {LOG_FILE}")


class BatchEvaluationArgs(BaseModel):
    input_path: str = Field(
        description=(
            "Path to input file (CSV or JSON) with fields: "
            "question, answer, relevant_passage/document"
        )
    )
    project_id: str = Field(description="Project ID for organizing results")
    output_csv_path: str | None = Field(
        default=None,
        description=(
            "Path to save results CSV. Defaults to projects/<project_id>/evaluation/results.csv"
        ),
    )
    rag_service_url: str = Field(
        default="http://localhost:8000",
        description="RAG service base URL (e.g., http://localhost:8000)",
    )
    evaluation_service_url: str | None = Field(
        default=None,
        description="Optional URL for external evaluation service (for generation metrics)",
    )
    max_concurrent: int = Field(default=5, description="Max concurrent requests to RAG service")
    max_questions: int | None = Field(
        default=None, description="Limit number of questions to process (for debugging)"
    )


server = FastMCP(
    "rag-evaluation",
)


# --- Helper Functions Ported from utils.py (adapted for no-pandas) ---


def extract_txt_documents(text: str | list[str] | Any) -> list[str]:
    """
    Extract a list of document names from the given text or list.
    Documents are identified by the '.txt, ' pattern or '.txt' at the end.
    """
    if isinstance(text, list):
        # If it's already a list, process each item
        documents = []
        for item in text:
            documents.extend(extract_txt_documents(item))
        return list(set(documents))  # unique

    if not isinstance(text, str):
        return []

    # Handle JSON string representation of list
    if text.startswith("[") and text.endswith("]"):
        try:
            loaded = json.loads(text)
            if isinstance(loaded, list):
                return extract_txt_documents(loaded)
        except json.JSONDecodeError:
            pass

    # QUICKFIX from original code
    # Simple split by comma if string
    return [doc.strip() for doc in text.split(",") if doc.strip()]


def normalize_doc_id(doc_id: str) -> str:
    """Normalize document ID for robust comparison."""
    if doc_id is None:
        return ""

    # Convert to string, strip whitespace
    doc = str(doc_id).strip()

    # Unicode normalization (critical!)
    doc = unicodedata.normalize("NFKC", doc)

    # Remove invisible zero-width characters
    doc = re.sub(r"[\u200B-\u200F\uFEFF]", "", doc)

    # Replace non-breaking spaces with normal
    doc = doc.replace("\u00a0", " ")

    # Normalize whitespace (collapse multiple spaces)
    doc = re.sub(r"\s+", " ", doc)

    # Cut file extensions (как было)
    for ext in [".json", ".pdf", ".docx", ".doc", ".txt", ".xlsx", ".xls", ".pptx", ".ppt"]:
        if doc.lower().endswith(ext):
            doc = doc[: -len(ext)]
            break

    return doc.strip()


def compute_row_metrics(retrieved_docs: list[str], relevant_docs: list[str]) -> dict[str, float]:
    """
    Compute precision, recall, and accuracy for one query.
    """
    retrieved_set = {normalize_doc_id(doc) for doc in retrieved_docs}
    relevant_set = {normalize_doc_id(doc) for doc in relevant_docs}
    logger.debug(f"Computing metrics: retrieved={retrieved_set}, relevant={relevant_set}")

    # Filter out empty strings
    retrieved_set = {d for d in retrieved_set if d}
    relevant_set = {d for d in relevant_set if d}

    intersection_count = len(retrieved_set.intersection(relevant_set))
    retrieved_count = len(retrieved_set)
    relevant_count = len(relevant_set)

    precision = intersection_count / retrieved_count if retrieved_count > 0 else 0.0
    recall = intersection_count / relevant_count if relevant_count > 0 else 0.0
    accuracy = 1.0 if retrieved_set == relevant_set else 0.0

    return {"precision": precision, "recall": recall, "accuracy": accuracy}


async def query_rag_system_async(
    client: httpx.AsyncClient,
    user_query: str,
    rag_system_endpoint: str,
) -> dict:
    """
    Async query to RAG system.
    """
    headers = {"Content-Type": "application/json"}
    data = {
        "query": user_query,
    }
    # Ensure URL ends with /
    base_url = rag_system_endpoint.rstrip("/")
    url = f"{base_url}/api/query/evaluation"
    try:
        response = await client.post(url, headers=headers, json=data)
        if response.status_code == 200:
            return response.json()
        return {"error": response.text, "status_code": response.status_code}
    except Exception as e:
        return {"error": str(e), "status_code": 0}


async def call_evaluation_service(
    client: httpx.AsyncClient,
    evaluation_url: str,
    results: list[dict],
    output_dir: Path | None = None,
) -> list[dict]:
    """
    Send results to external evaluation service for generation metrics.

    The service expects a CSV file upload with columns:
    - user_input (question)
    - response (generated_answer)
    - reference (target_answer)
    - context (chunks)

    Returns CSV with added columns: faithfulness, answer_correctness, donkit_score
    """
    # Prepare CSV for evaluation service
    # Map our column names to what evaluation service expects
    logger.debug(f"Preparing {len(results)} results for evaluation service")
    eval_rows = []
    for r in results:
        if "error" in r:
            continue
        # target_context from input CSV (if present)
        target_context_raw = r.get("_target_context_raw", [])
        if not isinstance(target_context_raw, list):
            target_context_raw = [target_context_raw] if target_context_raw else []

        eval_rows.append(
            {
                "query": r.get("question", ""),
                "generated_answer": r.get("answer", ""),  # answer field from CSV output
                "target": r.get("_target_answer", ""),  # internal field (reference answer)
                "context": repr([]),
                "target_context": repr(target_context_raw),
            }
        )

    if not eval_rows:
        logger.debug("No valid rows to evaluate")
        return results

    # Convert to CSV string
    output = io.StringIO()
    fieldnames = ["query", "generated_answer", "target", "context", "target_context"]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(eval_rows)
    csv_content = output.getvalue().encode("utf-8")
    logger.debug(f"CSV content size: {len(csv_content)} bytes, {len(eval_rows)} rows")
    logger.debug(f"First row sample: {eval_rows[0] if eval_rows else 'N/A'}")

    # Save evaluation input CSV for debugging
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        eval_input_path = output_dir / "evaluation_input.csv"
        eval_input_path.write_bytes(csv_content)
        logger.debug(f"Saved evaluation input to {eval_input_path}")

    try:
        # Send to evaluation service as multipart form
        files = {"file": ("dataset.csv", csv_content, "text/csv")}
        data = {"format": "csv"}
        logger.debug(f"Sending POST to {evaluation_url}")

        response = await client.post(
            evaluation_url,
            files=files,
            data=data,
            timeout=httpx.Timeout(600.0),
        )
        logger.debug(f"Response status: {response.status_code}")

        if response.status_code != 200:
            logger.error(f"Evaluation service error: {response.status_code} - {response.text}")
            return results

        # Parse response CSV
        response_csv = response.text
        logger.debug(f"Response CSV length: {len(response_csv)} chars")
        logger.debug(f"Response CSV preview: {response_csv[:500] if response_csv else 'empty'}")

        # Save evaluation output CSV for debugging
        if output_dir and response_csv:
            eval_output_path = output_dir / "evaluation_output.csv"
            eval_output_path.write_text(response_csv, encoding="utf-8")
            logger.debug(f"Saved evaluation output to {eval_output_path}")

        reader = csv.DictReader(io.StringIO(response_csv))
        evaluated_rows = list(reader)
        logger.debug(f"Parsed {len(evaluated_rows)} evaluated rows")
        if evaluated_rows:
            logger.debug(f"Evaluated row columns: {list(evaluated_rows[0].keys())}")
            logger.debug(f"First evaluated row: {evaluated_rows[0]}")

        # Merge evaluation metrics back into results
        # Evaluator response key changed over time (user_input vs query). Support both.
        eval_by_question = {
            (row.get("user_input") or row.get("query") or ""): row for row in evaluated_rows
        }
        matched_count = 0

        for r in results:
            if "error" in r:
                continue
            question = r.get("question", "")
            if question in eval_by_question:
                eval_data = eval_by_question[question]
                # Metrics from evaluation service response
                r["_answer_accuracy"] = eval_data.get("answer_accuracy", "")
                r["_simple_criteria"] = eval_data.get("simple_criteria", "")
                r["_rubric_score"] = eval_data.get("rubric_score", "")
                r["_faithfulness"] = eval_data.get("faithfulness", "")
                r["_donkit_score"] = eval_data.get("donkit_score", "")
                matched_count += 1

        logger.debug(f"Matched {matched_count} results with evaluation data")
        return results

    except Exception as e:
        logger.error(f"Evaluation service call failed: {e}")
        return results


async def rerun_evaluation_from_input_csv(
    *,
    evaluation_url: str = "http://34.58.196.231/evaluate",
    evaluation_input_csv_path: str | Path = "./evaluation_input.csv",
    evaluation_output_csv_path: str | Path | None = "./evaluation_output.csv",
) -> str:
    input_path = Path(evaluation_input_csv_path)
    if not input_path.exists():
        return json.dumps(
            {"error": f"Input CSV not found: {input_path}"},
            ensure_ascii=False,
            indent=2,
        )

    output_path = (
        Path(evaluation_output_csv_path)
        if evaluation_output_csv_path is not None
        else input_path.parent / "evaluation_output.csv"
    )

    csv_content = input_path.read_bytes()
    files = {"file": ("dataset.csv", csv_content, "text/csv")}
    data = {"format": "csv"}

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(600.0)) as client:
            response = await client.post(
                evaluation_url,
                files=files,
                data=data,
            )

        if response.status_code != 200:
            return json.dumps(
                {
                    "error": "Evaluation service error",
                    "status_code": response.status_code,
                    "detail": response.text,
                },
                ensure_ascii=False,
                indent=2,
            )

        output_path.write_text(response.text, encoding="utf-8")
        return json.dumps(
            {
                "status": "success",
                "input_file": str(input_path),
                "output_file": str(output_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    except Exception:
        raise


def extract_answer_and_sources(answer_json: dict) -> tuple[str, list[str], list[str]]:
    """
    Extract answer, context IDs, and chunks from RAG response.

    RAG response format:
    {
        "answer": "...",
        "context": "filename.json" or "file1.json, file2.json",
        "chunks": ["chunk1 text", "chunk2 text", ...]
    }
    """
    answer_text = answer_json.get("answer", "")

    # context field can be a single filename or comma-separated list
    context_raw = answer_json.get("context", "")
    if isinstance(context_raw, str):
        # Split by comma if multiple files, otherwise single file
        context_ids = [c.strip() for c in context_raw.split(",") if c.strip()]
    elif isinstance(context_raw, list):
        context_ids = context_raw
    else:
        context_ids = []

    # chunks is a list of strings (chunk content)
    chunks = answer_json.get("chunks", [])
    if not isinstance(chunks, list):
        chunks = [chunks] if chunks else []

    return answer_text, context_ids, chunks


# --- Tools ---


@server.tool(
    name="evaluate_batch",
    description=(
        "Run batch evaluation from a CSV or JSON file. "
        "Input fields: 'question', 'answer' (optional), 'relevant_passage'/'document'. "
        "Calculates Precision, Recall, Accuracy for retrieval."
    ).strip(),
)
async def evaluate_batch(args: BatchEvaluationArgs) -> str:
    """
    Process a batch of questions from CSV or JSON,
    call RAG service, compute metrics, and save results.
    """
    input_path = Path(args.input_path)

    # Default output path based on project_id (same pattern as chunker_server)
    if args.output_csv_path:
        output_path = Path(args.output_csv_path).resolve()
    else:
        output_path = Path(f"projects/{args.project_id}/evaluation/results.csv").resolve()

    if not input_path.exists():
        return json.dumps(
            {"error": f"Input file {input_path} not found."}, ensure_ascii=False, indent=2
        )

    # Read input file (CSV or JSON)
    rows = []
    try:
        if input_path.suffix.lower() == ".json":
            # Read JSON file
            with open(input_path, encoding="utf-8") as f:
                json_data = json.load(f)

            if not isinstance(json_data, list):
                return json.dumps(
                    {"error": "JSON file must contain a list of objects"},
                    ensure_ascii=False,
                    indent=2,
                )

            for item in json_data:
                # Map JSON fields to normalized row
                question = item.get("question") or item.get("user_input") or item.get("query")
                answer = item.get("answer") or item.get("response") or item.get("target")
                relevant = (
                    item.get("document") or item.get("documents") or item.get("relevant_passage")
                )
                target_context = item.get("target_context") or item.get("reference_context")

                if question:
                    rows.append(
                        {
                            "question": question,
                            "answer": answer,
                            "relevant_passage": relevant,
                            "target_context": target_context,
                        }
                    )
        else:
            # Read CSV file
            with open(input_path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames or []

                # map column names if needed
                question_col = next(
                    (c for c in fieldnames if c.lower() in ["question", "user_input", "query"]),
                    "question",
                )
                answer_col = next(
                    (
                        c
                        for c in fieldnames
                        if c.lower() in ["answer", "response", "target", "reference_answer"]
                    ),
                    "answer",
                )
                context_col = next(
                    (
                        c
                        for c in fieldnames
                        if c.lower()
                        in ["relevant_passage", "relevant_passage_ids", "document", "documents"]
                    ),
                    "relevant_passage",
                )
                target_context_col = next(
                    (c for c in fieldnames if c.lower() in ["target_context", "reference_context"]),
                    None,
                )

                if question_col not in fieldnames:
                    return json.dumps(
                        {"error": f"Input CSV must have a question column. Found: {fieldnames}"},
                        ensure_ascii=False,
                        indent=2,
                    )

                for row in reader:
                    normalized_row = {
                        "question": row.get(question_col),
                        "answer": row.get(answer_col),
                        "relevant_passage": row.get(context_col),
                        "target_context": row.get(target_context_col)
                        if target_context_col
                        else None,
                    }
                    rows.append(normalized_row)
    except Exception as e:
        return json.dumps(
            {"error": f"Failed to read input file: {str(e)}"}, ensure_ascii=False, indent=2
        )

    def _has_real_value(value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, list):
            return any(_has_real_value(v) for v in value)
        if not isinstance(value, str):
            return bool(value)

        s = value.strip()
        if not s:
            return False

        placeholders = {"-", "—", "–", "_", "null", "none", "na", "n/a", "nan"}
        return s.lower() not in placeholders

    def has_ground_truth(rows_data: list[dict[str, Any]]) -> bool:
        for r in rows_data:
            if _has_real_value(r.get("relevant_passage")):
                return True
            if _has_real_value(r.get("target_context")):
                return True
        return False

    benchmark_mode = not has_ground_truth(rows)
    if benchmark_mode and args.output_csv_path is None:
        output_path = Path(f"projects/{args.project_id}/evaluation/result.csv").resolve()

    # Limit number of questions if max_questions is set (for debugging)
    logger.debug(f"Total rows read from CSV: {len(rows)}")
    logger.debug(f"max_questions param: {args.max_questions}")
    if args.max_questions is not None and args.max_questions > 0:
        rows = rows[: args.max_questions]
        logger.info(f"Limited to {len(rows)} questions (max_questions={args.max_questions})")
    else:
        logger.debug(f"Processing all {len(rows)} questions (no limit set)")

    # Semaphore for concurrency
    semaphore = asyncio.Semaphore(args.max_concurrent)

    async def process_row(row):
        async with semaphore:
            question = row.get("question")
            if not question:
                return None

            try:
                async with httpx.AsyncClient(timeout=60.0) as client:
                    start_time = time.time()
                    rag_response = await query_rag_system_async(
                        client,
                        question,
                        args.rag_service_url,
                    )
                    rag_response_time = time.time() - start_time

                # Extract data
                if "error" in rag_response:
                    return {
                        "question": question,
                        "error": rag_response["error"],
                        "status_code": rag_response.get("status_code"),
                    }

                generated_answer, retrieved_ids, chunks = extract_answer_and_sources(rag_response)

                relevant_ids: list[str] = []
                metrics = {"precision": 0.0, "recall": 0.0, "accuracy": 0.0}
                if not benchmark_mode:
                    # Parse expected relevant docs from CSV
                    relevant_passage_raw = row.get("relevant_passage", "")
                    relevant_ids = extract_txt_documents(relevant_passage_raw)

                    # Compute Metrics
                    metrics = compute_row_metrics(retrieved_ids, relevant_ids)
                    logger.debug(f"Metrics result: {metrics}")

                # Replace .json with .pdf in doc names for CSV output
                docs_for_csv = [doc.replace(".json", ".pdf") for doc in retrieved_ids]

                if benchmark_mode:
                    return {
                        "question": question,
                        "answer": generated_answer,
                        "document": json.dumps(docs_for_csv, ensure_ascii=False),
                    }

                # Parse target_context from CSV if present
                target_context_raw = row.get("target_context")
                if target_context_raw:
                    # Try to parse as list if it looks like one
                    try:
                        target_context_list = ast.literal_eval(target_context_raw)
                        if not isinstance(target_context_list, list):
                            target_context_list = [target_context_raw]
                    except (ValueError, SyntaxError):
                        target_context_list = [target_context_raw]
                else:
                    target_context_list = []

                return {
                    # CSV output fields
                    "question": question,
                    "docs": json.dumps(docs_for_csv, ensure_ascii=False),
                    "chunks": json.dumps(chunks, ensure_ascii=False),
                    "answer": generated_answer,
                    # Metrics (for JSON response only)
                    "_target_answer": row.get("answer"),
                    "_relevant_context": relevant_ids,
                    "_retrieved_context": retrieved_ids,
                    "_chunks_raw": chunks,  # raw list for evaluation service
                    "_target_context_raw": target_context_list,  # for evaluation service
                    "_precision": metrics["precision"],
                    "_recall": metrics["recall"],
                    "_accuracy": metrics["accuracy"],
                    "_rag_response_time": rag_response_time,
                }
            except Exception as e:
                return {"question": question, "error": str(e)}

    rag_start_time = time.time()
    tasks = [process_row(row) for row in rows]
    results = await asyncio.gather(*tasks)
    rag_total_time = time.time() - rag_start_time

    # Filter valid results
    valid_results = [r for r in results if r]

    if benchmark_mode:
        output_fields = ["question", "answer", "document"]
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=output_fields, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(valid_results)

            rag_times = [
                r.get("_rag_response_time", 0)
                for r in valid_results
                if r.get("_rag_response_time") is not None
            ]
            avg_rag_response_time = sum(rag_times) / len(rag_times) if rag_times else 0

            return json.dumps(
                {
                    "status": "success",
                    "mode": "benchmark",
                    "processed_rows": len(valid_results),
                    "output_file": str(output_path),
                    "timing": {
                        "rag_total_time_sec": round(rag_total_time, 2),
                        "avg_rag_response_time_sec": round(avg_rag_response_time, 3),
                    },
                },
                ensure_ascii=False,
                indent=2,
            )
        except Exception as e:
            return json.dumps(
                {
                    "error": "Failed to save results",
                    "detail": str(e),
                },
                ensure_ascii=False,
                indent=2,
            )

    # Call evaluation service for generation metrics (faithfulness, correctness, donkit_score)
    eval_total_time = 0.0
    if args.evaluation_service_url:
        logger.debug(f"Calling evaluation service at {args.evaluation_service_url}...")
        eval_start_time = time.time()
        async with httpx.AsyncClient() as eval_client:
            valid_results = await call_evaluation_service(
                eval_client,
                args.evaluation_service_url,
                valid_results,
                output_dir=output_path.parent,
            )
        eval_total_time = time.time() - eval_start_time

    # Calculate aggregates (using internal metric fields)
    total_acc = sum(r.get("_accuracy", 0) for r in valid_results if "_accuracy" in r)
    total_prec = sum(r.get("_precision", 0) for r in valid_results if "_precision" in r)
    total_rec = sum(r.get("_recall", 0) for r in valid_results if "_recall" in r)
    count = len([r for r in valid_results if "_accuracy" in r])

    # Calculate average RAG response time
    rag_times = [r.get("_rag_response_time", 0) for r in valid_results if "_rag_response_time" in r]
    avg_rag_response_time = sum(rag_times) / len(rag_times) if rag_times else 0

    # Generation metrics from evaluation service (prefixed with _)
    answer_accuracy_vals = [
        float(r.get("_answer_accuracy", 0)) for r in valid_results if r.get("_answer_accuracy")
    ]
    simple_criteria_vals = [
        float(r.get("_simple_criteria", 0)) for r in valid_results if r.get("_simple_criteria")
    ]
    rubric_score_vals = [
        float(r.get("_rubric_score", 0)) for r in valid_results if r.get("_rubric_score")
    ]
    faithfulness_vals = [
        float(r.get("_faithfulness", 0)) for r in valid_results if r.get("_faithfulness")
    ]
    donkit_vals = [
        float(r.get("_donkit_score", 0)) for r in valid_results if r.get("_donkit_score")
    ]

    aggregates = {
        "mean_accuracy": total_acc / count if count > 0 else 0,
        "mean_precision": total_prec / count if count > 0 else 0,
        "mean_recall": total_rec / count if count > 0 else 0,
        "mean_answer_accuracy": sum(answer_accuracy_vals) / len(answer_accuracy_vals)
        if answer_accuracy_vals
        else None,
        "mean_simple_criteria": sum(simple_criteria_vals) / len(simple_criteria_vals)
        if simple_criteria_vals
        else None,
        "mean_rubric_score": sum(rubric_score_vals) / len(rubric_score_vals)
        if rubric_score_vals
        else None,
        "mean_faithfulness": sum(faithfulness_vals) / len(faithfulness_vals)
        if faithfulness_vals
        else None,
        "mean_donkit_score": sum(donkit_vals) / len(donkit_vals) if donkit_vals else None,
    }

    # Save to CSV (only required fields, no metrics)
    output_fields = ["question", "docs", "chunks", "answer"]

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=output_fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(valid_results)

        return json.dumps(
            {
                "status": "success",
                "processed_rows": len(valid_results),
                "output_file": str(output_path),
                "metrics": aggregates,
                "timing": {
                    "rag_total_time_sec": round(rag_total_time, 2),
                    "avg_rag_response_time_sec": round(avg_rag_response_time, 3),
                    "evaluation_time_sec": round(eval_total_time, 2),
                },
            },
            ensure_ascii=False,
            indent=2,
        )

    except Exception as e:
        return json.dumps(
            {"error": "Failed to save results", "detail": str(e)}, ensure_ascii=False, indent=2
        )


def main() -> None:
    server.run(
        transport="stdio",
        log_level=os.getenv("RAGOPS_LOG_LEVEL", "CRITICAL"),
        show_banner=False,
    )


if __name__ == "__main__":
    main()
