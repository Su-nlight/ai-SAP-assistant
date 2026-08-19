"""
Evaluation pipeline — custom evaluator implementation.

This increment ships fast, dependency-light heuristic evaluators so the
pipeline (compute -> store -> aggregate) is real and working end-to-end.
Ragas/DeepEval (LLM-graded context precision/recall, faithfulness) are
heavier, slower, and cost real LLM calls per evaluation — they are a
documented next increment (see docs/ARCHITECTURE.md) that plugs into the
exact same `EvaluationResult` table and `/api/evaluate` contract; only
`_score_*` internals below would change.
"""
from __future__ import annotations

import uuid

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import NotFoundError
from app.models.audit_feedback import EvaluationResult
from app.models.chat import Message

_STOPWORDS = {"the", "a", "an", "is", "are", "of", "to", "in", "for", "on", "and", "or", "what", "how", "does"}


def _tokenize(text: str) -> set[str]:
    return {w.strip(".,?!").lower() for w in text.split() if w.strip(".,?!").lower() not in _STOPWORDS}


def _score_answer_relevancy(question: str, answer: str) -> float:
    """Lexical overlap between question and answer content words — a crude
    but zero-cost proxy for topical relevance."""
    q_tokens, a_tokens = _tokenize(question), _tokenize(answer)
    if not q_tokens:
        return 0.0
    overlap = len(q_tokens & a_tokens)
    return min(1.0, overlap / len(q_tokens))


def _score_groundedness(answer: str, citation_count: int) -> float:
    """Answers that cite at least one source are considered grounded; more
    citations relative to answer length indicates denser grounding."""
    if citation_count == 0:
        return 0.0
    sentence_count = max(1, answer.count(".") + answer.count("!") + answer.count("?"))
    return min(1.0, citation_count / sentence_count)


def _score_faithfulness(answer: str, citation_count: int) -> float:
    """Proxy: an ungrounded ('no answer found') response is trivially
    faithful; a substantive claim is only as faithful as it is cited."""
    no_answer_markers = ("do not contain", "does not contain", "cannot find", "no relevant")
    if any(marker in answer.lower() for marker in no_answer_markers):
        return 1.0
    return 1.0 if citation_count > 0 else 0.3


class EvaluationService:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def evaluate_message(self, message_id: uuid.UUID) -> EvaluationResult:
        message = await self._session.get(Message, message_id)
        if message is None:
            raise NotFoundError(f"Message {message_id} not found.")

        citation_count = len(message.citations_json or [])
        answer = message.answer or ""
        question = message.question or ""

        result = EvaluationResult(
            message_id=message.id,
            answer_relevancy=_score_answer_relevancy(question, answer),
            groundedness=_score_groundedness(answer, citation_count),
            faithfulness=_score_faithfulness(answer, citation_count),
            context_precision=None,  # requires ground-truth relevant-chunk labels — next increment
            context_recall=None,  # requires ground-truth relevant-chunk labels — next increment
            latency_ms=float(message.latency_ms) if message.latency_ms is not None else None,
            total_tokens=message.total_tokens,
            evaluator="custom-heuristic-v1",
        )
        self._session.add(result)
        await self._session.commit()
        return result

    async def get_aggregate_metrics(self) -> dict[str, float | int | None]:
        stmt = select(
            func.avg(EvaluationResult.answer_relevancy),
            func.avg(EvaluationResult.groundedness),
            func.avg(EvaluationResult.faithfulness),
            func.avg(EvaluationResult.latency_ms),
            func.avg(EvaluationResult.total_tokens),
            func.count(EvaluationResult.id),
        )
        row = (await self._session.execute(stmt)).one()
        return {
            "avg_answer_relevancy": row[0],
            "avg_groundedness": row[1],
            "avg_faithfulness": row[2],
            "avg_latency_ms": row[3],
            "avg_total_tokens": row[4],
            "evaluation_count": row[5],
        }
