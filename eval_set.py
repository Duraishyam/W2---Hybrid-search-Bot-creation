"""Evaluation cases for the APS Online Tamil School chatbot.

The set contains realistic parent questions with expected outcomes. Keyword-heavy
and paraphrased questions are intentionally mixed to evaluate both BM25 and
semantic retrieval in the hybrid retriever.

``expected_doc_id`` identifies the canonical knowledge-base record that should
ground an answer. It is ``None`` for questions that should be escalated.
``must_include`` contains optional lower-case phrases for loose answer checks.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvalCase:
    query: str
    expected: str  # "answer" or "escalate"
    expected_doc_id: str | None
    must_include: tuple[str, ...] = ()


EVAL_CASES: list[EvalCase] = [
    # ---- Should answer: FAQ-grounded ----
    EvalCase(
        "My child has never learned Tamil. Can they still join?",
        "answer",
        "faq-003",
        ("beginner", "tamil alphabet"),
    ),
    EvalCase(
        "How will you decide which Tamil class level is right for my daughter?",
        "answer",
        "faq-004",
        ("placement", "speaking"),
    ),
    EvalCase(
        "How many times per week are the online lessons?",
        "answer",
        "faq-005",
        ("once or twice",),
    ),
    EvalCase(
        "Do I need a laptop with a webcam for Tamil class?",
        "answer",
        "faq-006",
        ("camera", "microphone"),
    ),
    EvalCase(
        "Will my son be able to talk to the teacher live, or are lessons only videos?",
        "answer",
        "faq-007",
        ("live",),
    ),
    EvalCase(
        "Are worksheets included with registration, or do we have to buy them?",
        "answer",
        "faq-009",
        ("digital worksheets",),
    ),
    EvalCase(
        "My child was sick and missed Tamil school. Can they catch up?",
        "answer",
        "faq-012",
        ("catch up", "make-up"),
    ),
    EvalCase(
        "How can I find out whether my child is improving in Tamil?",
        "answer",
        "faq-014",
        ("feedback", "progress"),
    ),
    # ---- Should answer: School-manual-grounded ----
    EvalCase(
        "Should students join class before the scheduled start time?",
        "answer",
        "manual-003",
        ("five minutes",),
    ),
    EvalCase(
        "What behaviour is expected in the virtual classroom?",
        "answer",
        "manual-004",
        ("respectful", "raise a hand"),
    ),
    EvalCase(
        "Can I record my child's online Tamil lesson and share it with family?",
        "answer",
        "manual-009",
        ("written permission",),
    ),
    # ---- Should escalate: parent service requests need staff action ----
    EvalCase(
        "We have a recurring conflict on Saturdays. Can my child switch class times?",
        "escalate",
        None,
    ),
    EvalCase(
        "I need to tell the school that Arjun will not attend next week's lesson.",
        "escalate",
        None,
    ),
    EvalCase(
        "My child cannot open the Zoom link and has no audio in class.",
        "escalate",
        None,
    ),
    EvalCase(
        "Please send me a receipt for the tuition payment.",
        "escalate",
        None,
    ),
    # ---- Should escalate: outside the available knowledge base ----
    EvalCase("Do you offer online Hindi or Telugu classes too?", "escalate", None),
    EvalCase("Can you guarantee that my child will become fluent in three months?", "escalate", None),
    EvalCase("Do you have a scholarship or sibling-discount program?", "escalate", None),
    EvalCase("Can you arrange school bus transportation for in-person lessons?", "escalate", None),
    EvalCase("Which teacher will be assigned to my child's class next term?", "escalate", None),
]


def summary() -> None:
    """Print the number of expected-answer and expected-escalation cases."""
    answer_cases = sum(case.expected == "answer" for case in EVAL_CASES)
    escalation_cases = sum(case.expected == "escalate" for case in EVAL_CASES)
    print(
        f"{len(EVAL_CASES)} cases: {answer_cases} answer-expected, "
        f"{escalation_cases} escalate-expected"
    )


if __name__ == "__main__":
    summary()
