"""
FarrahAI — Module 7: Teacher Profile (Global)
==============================================
Teachers are GLOBAL across all subject rooms.
Each teacher has a profile built from their past question papers.

A teacher profile stores:
  - name
  - subjects taught
  - topic frequency in past papers
  - marks distribution patterns
  - question style patterns
  - paper difficulty trends

When a student asks "predict the paper for Prof. Sharma",
the system pulls Prof. Sharma's global profile and combines it
with the current subject's knowledge base.

If a teacher's pattern changes (e.g., new topic emphasis),
that update propagates globally to all subjects that teacher handles.
"""

import json
import logging
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime

logger = logging.getLogger(__name__)


# ── Teacher Profile Structure ─────────────────────────────────────────────────

def empty_profile(teacher_name: str) -> dict:
    return {
        "name":             teacher_name,
        "subjects":         [],
        "papers":           [],          # list of analyzed paper records
        "topic_frequency":  {},          # topic → count across all papers
        "marks_distribution": {},        # marks_type → avg_marks
        "question_patterns": [],         # recurring question styles
        "last_updated":     None,
    }


# ── Load / Save Teacher DB ────────────────────────────────────────────────────

def load_teacher_db(db_path: str) -> dict:
    """Load the global teacher database (JSON file)."""
    path = Path(db_path)
    if not path.exists():
        logger.info("No teacher DB found, creating empty one.")
        return {}
    with open(path, 'r') as f:
        return json.load(f)


def save_teacher_db(db: dict, db_path: str):
    """Save the global teacher database."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(db, f, indent=2)
    logger.info(f"Teacher DB saved: {path}")


# ── Create / Update Teacher Profile ──────────────────────────────────────────

def get_or_create_teacher(db: dict, teacher_name: str) -> dict:
    """Get existing teacher profile or create a new one."""
    if teacher_name not in db:
        db[teacher_name] = empty_profile(teacher_name)
        logger.info(f"Created new teacher profile: {teacher_name}")
    return db[teacher_name]


def add_paper_to_teacher(db: dict,
                          teacher_name: str,
                          paper_data: dict,
                          db_path: str = None) -> dict:
    """
    Add a question paper to a teacher's profile and update their stats.

    paper_data format:
    {
        "subject":    "AI_ML",
        "paper_type": "internal" | "endsem" | "midsem",
        "year":       2024,
        "semester":   "odd" | "even",
        "topics":     [{"name": str, "marks": int, "question": str}],
        "total_marks": int,
    }

    This is the function that makes teacher profiles "global" —
    any change here reflects everywhere the teacher's profile is used.
    """
    profile = get_or_create_teacher(db, teacher_name)

    # Add to subject list
    subj = paper_data.get("subject", "unknown")
    if subj not in profile["subjects"]:
        profile["subjects"].append(subj)

    # Add paper record
    profile["papers"].append({
        "subject":    subj,
        "paper_type": paper_data.get("paper_type", "unknown"),
        "year":       paper_data.get("year", datetime.now().year),
        "semester":   paper_data.get("semester", "unknown"),
        "total_marks":paper_data.get("total_marks", 0),
        "topic_count":len(paper_data.get("topics", [])),
    })

    # Update topic frequency
    for t in paper_data.get("topics", []):
        topic = t.get("name", "unknown").strip().lower()
        profile["topic_frequency"][topic] = profile["topic_frequency"].get(topic, 0) + 1

    # Update marks distribution (ignore missing/invalid/non-positive marks)
    for t in paper_data.get("topics", []):
        raw_marks = t.get("marks", None)
        try:
            marks_val = int(raw_marks)

        except (TypeError, ValueError):
            continue
        if marks_val <= 0:
            continue

        marks_key = str(marks_val)
        if marks_key not in profile["marks_distribution"]:
            profile["marks_distribution"][marks_key] = 0
        profile["marks_distribution"][marks_key] += 1

    profile["last_updated"] = datetime.now().isoformat()

    if db_path:
        save_teacher_db(db, db_path)

    logger.info(f"Updated profile for {teacher_name}: "
                f"{len(profile['papers'])} papers, "
                f"{len(profile['topic_frequency'])} unique topics")
    return profile


# ── Analysis ──────────────────────────────────────────────────────────────────

def get_top_topics(teacher_name: str, db: dict, top_n: int = 10) -> list[dict]:
    """
    Get the most frequently asked topics for a teacher.
    Returns sorted list of { 'topic', 'count', 'rank' }
    """
    profile = db.get(teacher_name)
    if not profile:
        raise ValueError(f"Teacher '{teacher_name}' not found in DB.")

    freq = profile["topic_frequency"]
    sorted_topics = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:top_n]

    return [
        {"rank": i+1, "topic": t, "count": c}
        for i, (t, c) in enumerate(sorted_topics)
    ]


def get_marks_pattern(teacher_name: str, db: dict) -> dict:
    """
    Get the marks distribution pattern for a teacher.
    Shows how often they ask 2-mark, 5-mark, 10-mark questions.
    """
    profile = db.get(teacher_name)
    if not profile:
        raise ValueError(f"Teacher '{teacher_name}' not found.")

    return profile["marks_distribution"]


def get_teacher_summary(teacher_name: str, db: dict) -> str:
    """Human-readable summary of a teacher's paper pattern."""
    profile = db.get(teacher_name)
    if not profile:
        return f"No profile found for {teacher_name}"

    top_topics = get_top_topics(teacher_name, db, top_n=5)
    marks = get_marks_pattern(teacher_name, db)

    lines = [
        f"Teacher: {teacher_name}",
        f"Subjects: {', '.join(profile['subjects'])}",
        f"Papers analyzed: {len(profile['papers'])}",
        f"Last updated: {profile['last_updated']}",
        "",
        "Top 5 topics by frequency:",
    ]
    for t in top_topics:
        lines.append(f"  {t['rank']}. {t['topic']} (asked {t['count']} time(s))")

    lines += ["", "Marks distribution:"]
    valid_marks = []
    for marks_val, count in marks.items():
        try:
            m = int(marks_val)
        except (TypeError, ValueError):
            continue
        if m <= 0:
            continue
        valid_marks.append((m, count))

    for marks_val, count in sorted(valid_marks, key=lambda x: x[0]):
        lines.append(f"  {marks_val} marks: {count} question(s)")

    return "\n".join(lines)


def list_teachers(db: dict) -> list[str]:
    """List all teacher names in the global DB."""
    return sorted(db.keys())
