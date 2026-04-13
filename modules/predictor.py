"""
FarrahAI — Module 8: Question Paper Prediction
================================================
Combines teacher profile + subject knowledge base
to predict probable exam topics and generate a sample paper.

Technically this is:
  - trend-based topic forecasting
  - frequency + recency + marks-weight scoring
  - retrieval of supporting notes for each predicted topic

NOT: "magically predicting the exact paper"
YES: "ranking topics by historical probability + content coverage"
"""

import logging
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


def score_topics(topic_frequency: dict,
                 recency_weight: float = 0.3,
                 freq_weight: float = 0.7) -> list[dict]:
    """
    Score topics by frequency and recency.

    recency_weight: recent papers count more
    freq_weight: raw count matters

    Returns sorted list of { 'topic', 'score', 'rank' }
    """
    if not topic_frequency:
        return []

    topics = list(topic_frequency.items())
    counts = np.array([c for _, c in topics], dtype=float)

    # Normalize
    norm_counts = counts / counts.max() if counts.max() > 0 else counts

    # Simple combined score (can be improved with actual recency data)
    scores = freq_weight * norm_counts + recency_weight * norm_counts  # placeholder

    ranked = sorted(
        zip([t for t, _ in topics], scores.tolist()),
        key=lambda x: x[1],
        reverse=True
    )

    return [
        {"rank": i+1, "topic": t, "score": round(s, 4)}
        for i, (t, s) in enumerate(ranked)
    ]


def predict_important_topics(teacher_name: str,
                               subject: str,
                               teacher_db: dict,
                               index_dir: str,
                               top_n: int = 10,
                               model_name: str = "all-MiniLM-L6-v2") -> list[dict]:
    """
    Predict top N important topics for a teacher + subject combination.

    For each predicted topic:
      - shows score (based on historical frequency)
      - retrieves relevant note chunks from the subject KB

    Returns:
        list of { 'rank', 'topic', 'score', 'supporting_notes' }
    """
    from modules.teacher_profile import get_top_topics
    from modules.retriever import retrieve, format_retrieved_context

    profile = teacher_db.get(teacher_name)
    if not profile:
        raise ValueError(f"No profile for teacher: {teacher_name}")

    # Get top topics from teacher's history
    top_topics = get_top_topics(teacher_name, teacher_db, top_n=top_n)

    results = []
    for item in top_topics:
        topic = item["topic"]

        # Try to retrieve relevant notes for this topic
        try:
            notes = retrieve(
                query=topic,
                subject=subject,
                index_dir=index_dir,
                top_k=3,
                model_name=model_name
            )
            note_text = format_retrieved_context(notes)
        except Exception as e:
            logger.warning(f"Could not retrieve notes for topic '{topic}': {e}")
            note_text = "No notes found for this topic."

        results.append({
            "rank":             item["rank"],
            "topic":            topic,
            "historical_count": item["count"],
            "supporting_notes": note_text,
        })

    return results


def generate_sample_paper(teacher_name: str,
                            subject: str,
                            teacher_db: dict,
                            total_marks: int = 100,
                            paper_type: str = "endsem") -> dict:
    """
    Generate a sample predicted question paper.

    Uses teacher's marks distribution pattern to decide
    how many 2-mark, 5-mark, 10-mark questions to include.

    Returns a structured paper dict.
    """
    from modules.teacher_profile import get_top_topics, get_marks_pattern

    profile = teacher_db.get(teacher_name)
    if not profile:
        raise ValueError(f"No profile found for teacher: {teacher_name}")

    top_topics   = get_top_topics(teacher_name, teacher_db, top_n=15)
    marks_pattern = get_marks_pattern(teacher_name, teacher_db)

    # Decide section structure
    # Default fallback if no marks pattern available
    if not marks_pattern:
        structure = [
            {"marks": 2,  "count": 10},
            {"marks": 5,  "count": 6},
            {"marks": 10, "count": 4},
        ]
    else:
        # Build from teacher's actual pattern
        structure = _infer_paper_structure(marks_pattern, total_marks)

    # Assign topics to questions
    paper = {
        "teacher":    teacher_name,
        "subject":    subject,
        "paper_type": paper_type,
        "total_marks": total_marks,
        "sections":   [],
        "disclaimer": (
            "This is a predicted sample paper based on historical topic analysis. "
            "It is not a guarantee of the actual exam content."
        ),
    }

    topic_pool = [t["topic"] for t in top_topics]
    topic_idx  = 0

    for section in structure:
        marks   = section["marks"]
        count   = section["count"]
        q_list  = []

        for i in range(count):
            topic = topic_pool[topic_idx % len(topic_pool)]
            topic_idx += 1
            q_list.append({
                "q_no":  i + 1,
                "topic": topic,
                "marks": marks,
                "note":  f"Based on {teacher_name}'s historical emphasis on '{topic}'",
            })

        paper["sections"].append({
            "section_label": f"Section ({marks} marks each)",
            "marks_each":    marks,
            "questions":     q_list,
        })

    return paper


def _infer_paper_structure(marks_pattern: dict, total_marks: int) -> list[dict]:
    """
    Infer section structure from teacher's marks distribution.
    Simple heuristic: top 3 marks values → create sections.
    """
    sorted_marks = sorted(marks_pattern.items(),
                          key=lambda x: int(x[0]))

    structure = []
    for m_str, count in sorted_marks[:3]:
        m = int(m_str)
        # normalize count to reasonable number
        q_count = max(2, min(count, total_marks // m))
        structure.append({"marks": m, "count": q_count})

    return structure


def format_paper_output(paper: dict) -> str:
    """Human-readable formatted paper for CLI / notebook display."""
    lines = [
        f"╔══════════════════════════════════════════════════╗",
        f"  FARRAHAI PREDICTED SAMPLE PAPER",
        f"  Subject:    {paper['subject']}",
        f"  Teacher:    {paper['teacher']}",
        f"  Paper Type: {paper['paper_type']}",
        f"  Total Marks:{paper['total_marks']}",
        f"╚══════════════════════════════════════════════════╝",
        "",
        f"⚠ {paper['disclaimer']}",
        "",
    ]

    for section in paper["sections"]:
        lines.append(f"── {section['section_label']} ──────────────────────────")
        for q in section["questions"]:
            lines.append(f"  Q{q['q_no']}. [{q['topic']}]  ({q['marks']} marks)")
        lines.append("")

    return "\n".join(lines)
