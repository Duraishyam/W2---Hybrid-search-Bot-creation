"""Run the APS Tamil School evaluation set and report first-contact resolution.

Metrics:
  FCR (first-contact resolution): fraction of all questions correctly handled
  on the first turn. Answer-expected questions must be answered with the
  expected top document and required phrases; escalation-expected questions
  must be escalated.

  The failure breakdown distinguishes false answers (hallucination risk) from
  safe but unhelpful false escalations.

Run:
    python eval.py
    python eval.py --quiet
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass

from eval_set import EVAL_CASES, EvalCase
from bot_logic import run_bot
from retriever import HybridRetriever, build_retriever


# Must match the confidence gate in streamlit_app.py.
CONFIDENCE_THRESHOLD = 0.55


@dataclass
class RowResult:
    case: EvalCase
    bot_decision: str
    top_doc_id: str
    top_sim: float
    answer: str
    judge_used: bool
    correct: bool
    failure_mode: str  # "" | wrong_doc | missing_phrase | false_answer | false_escalate


def ask(retriever: HybridRetriever, query: str) -> dict:
    """Run the same retrieve-score-answer/escalate flow used by the app."""
    result = run_bot(retriever, query)
    result["judge_used"] = False
    return result


def grade(case: EvalCase, output: dict) -> RowResult:
    """Compare one bot result against its expected answer or escalation."""
    decision = output.get("decision", "unknown")
    scored = output.get("scored") or []
    top_doc_id = str(scored[0][0].metadata.get("id", "")) if scored else ""
    top_sim = float(output.get("top_sim", 0.0))
    answer = str(output.get("answer", ""))
    judge_used = bool(output.get("judge_used", False))

    correct = False
    failure_mode = ""
    if case.expected == "escalate":
        if decision == "escalate":
            correct = True
        else:
            failure_mode = "false_answer"
    elif decision == "escalate":
        failure_mode = "false_escalate"
    elif top_doc_id != case.expected_doc_id:
        failure_mode = "wrong_doc"
    else:
        missing = [phrase for phrase in case.must_include if phrase.lower() not in answer.lower()]
        if missing:
            failure_mode = f"missing_phrase: {missing}"
        else:
            correct = True

    return RowResult(
        case=case,
        bot_decision=decision,
        top_doc_id=top_doc_id,
        top_sim=top_sim,
        answer=answer,
        judge_used=judge_used,
        correct=correct,
        failure_mode=failure_mode,
    )


def run(verbose: bool = True) -> list[RowResult]:
    """Run all cases and optionally print per-query results."""
    print("Building hybrid retriever (loads the embedding model on first run)...", flush=True)
    start_time = time.time()
    retriever = build_retriever()
    print(f"Ready in {time.time() - start_time:.1f}s.\n", flush=True)

    rows: list[RowResult] = []
    for index, case in enumerate(EVAL_CASES, start=1):
        row = grade(case, ask(retriever, case.query))
        rows.append(row)
        if verbose:
            mark = "PASS" if row.correct else "FAIL"
            print(
                f"[{index:2d}] {mark}  expected={case.expected:8s} "
                f"got={row.bot_decision:8s} top={row.top_doc_id or '-':14s} "
                f"sim={row.top_sim:.2f} judge={'Y' if row.judge_used else 'N'}"
            )
            print(f"     Q: {case.query}")
            preview = row.answer[:160]
            print(f"     A: {preview}{'...' if len(row.answer) > 160 else ''}")
            if row.failure_mode:
                print(f"     !! {row.failure_mode}")
            print()
    return rows


def report(rows: list[RowResult]) -> None:
    """Print FCR and the evaluation confusion/failure breakdown."""
    total = len(rows)
    correct = sum(row.correct for row in rows)
    fcr = correct / total if total else 0.0
    answer_expected = [row for row in rows if row.case.expected == "answer"]
    escalate_expected = [row for row in rows if row.case.expected == "escalate"]
    answered_correctly = sum(row.correct for row in answer_expected)
    escalated_correctly = sum(row.correct for row in escalate_expected)
    false_answers = sum(row.failure_mode == "false_answer" for row in rows)
    false_escalates = sum(row.failure_mode == "false_escalate" for row in rows)
    wrong_documents = sum(row.failure_mode == "wrong_doc" for row in rows)
    missing_phrases = sum(row.failure_mode.startswith("missing_phrase") for row in rows)

    print("=" * 68)
    print("RESULTS")
    print("=" * 68)
    print(f"First-Contact Resolution (FCR):   {correct}/{total}  = {fcr * 100:.1f}%")
    print(f"  answer-expected correct:        {answered_correctly}/{len(answer_expected)}")
    print(f"  escalate-expected correct:      {escalated_correctly}/{len(escalate_expected)}")
    print()
    print("Failure breakdown:")
    print(f"  false-answer (should escalate): {false_answers}   <-- hallucination risk")
    print(f"  false-escalate (lost FCR):      {false_escalates}")
    print(f"  wrong retrieved doc:            {wrong_documents}")
    print(f"  answer missing key phrase:      {missing_phrases}")
    print("=" * 68)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quiet", action="store_true", help="Show only the final report.")
    args = parser.parse_args()

    rows = run(verbose=not args.quiet)
    report(rows)
    fcr = sum(row.correct for row in rows) / len(rows) if rows else 0.0
    return 0 if fcr >= 0.70 else 1


if __name__ == "__main__":
    sys.exit(main())
