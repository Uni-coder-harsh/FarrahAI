"""
FarrahAI — Main CLI Entry Point
=================================
Usage:

  # Create a subject room
  python main.py --action create_room --subject AI_ML --teacher "Prof_Sharma"

  # Upload a file to a room
  python main.py --action upload --subject AI_ML --file notes.jpg

  # Ask a question
  python main.py --action ask --subject AI_ML --query "What is backpropagation?"

  # Predict important topics
  python main.py --action predict --subject AI_ML --teacher "Prof_Sharma"

  # Generate sample paper
  python main.py --action sample_paper --subject AI_ML --teacher "Prof_Sharma"

  # List all rooms
  python main.py --action list_rooms

  # Check Ollama status
  python main.py --action check_ollama
"""

import argparse
import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import (
    DATA_DIR, EMBEDDINGS_DIR, TEACHER_DB_PATH,
    OLLAMA_MODEL, TOP_K_RETRIEVAL, LOG_LEVEL
)

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("farrahai")


def cmd_create_room(args):
    from modules.room_manager import create_room
    from modules.teacher_profile import load_teacher_db, get_or_create_teacher, save_teacher_db
    db = load_teacher_db(str(TEACHER_DB_PATH))
    get_or_create_teacher(db, args.teacher)
    save_teacher_db(db, str(TEACHER_DB_PATH))
    create_room(args.subject, args.teacher, str(DATA_DIR))


def cmd_upload(args):
    from modules.room_manager import upload_and_index
    upload_and_index(
        file_path=args.file,
        subject=args.subject,
        base_dir=str(DATA_DIR)
    )


def cmd_ask(args):
    from modules.retriever import retrieve, format_retrieved_context
    from modules.ollama_chat import answer_from_notes, is_ollama_running

    print(f"\n📚 Searching subject: {args.subject}")
    print(f"❓ Query: {args.query}\n")

    results = retrieve(
        query=args.query,
        subject=args.subject,
        index_dir=str(EMBEDDINGS_DIR),
        top_k=TOP_K_RETRIEVAL
    )

    if not results:
        print("No relevant notes found. Upload more notes first.")
        return

    context = format_retrieved_context(results)

    if is_ollama_running():
        print(f"🤖 Generating answer via Ollama [{OLLAMA_MODEL}]...\n")
        answer = answer_from_notes(args.query, context, model=OLLAMA_MODEL)
        print("── ANSWER ──────────────────────────────────────")
        print(answer)
    else:
        print("⚠ Ollama not running. Showing raw retrieved notes instead.\n")
        print("── RETRIEVED NOTES ──────────────────────────────")
        print(context)

    print(f"\n── Top {len(results)} relevant note chunks retrieved ──")
    for r in results:
        print(f"  [{r['rank']}] Score: {r['score']} | {r['text'][:80]}...")


def cmd_predict(args):
    from modules.teacher_profile import load_teacher_db, get_teacher_summary
    from modules.predictor import predict_important_topics, format_paper_output
    from modules.ollama_chat import explain_prediction, is_ollama_running

    db = load_teacher_db(str(TEACHER_DB_PATH))

    if args.teacher not in db:
        print(f"Teacher '{args.teacher}' not found in DB.")
        print(f"Available teachers: {list(db.keys())}")
        return

    print(f"\n── Teacher Profile: {args.teacher} ──────────────────")
    print(get_teacher_summary(args.teacher, db))

    print(f"\n── Predicted Important Topics ───────────────────────")
    try:
        topics = predict_important_topics(
            teacher_name=args.teacher,
            subject=args.subject,
            teacher_db=db,
            index_dir=str(EMBEDDINGS_DIR),
        )
        for t in topics:
            print(f"  {t['rank']}. {t['topic']}  (seen {t['historical_count']}x)")

        if is_ollama_running():
            print("\n── Study Recommendation ──────────────────────────")
            rec = explain_prediction(topics, args.teacher, model=OLLAMA_MODEL)
            print(rec)

    except Exception as e:
        print(f"Error: {e}")


def cmd_sample_paper(args):
    from modules.teacher_profile import load_teacher_db
    from modules.predictor import generate_sample_paper, format_paper_output

    db = load_teacher_db(str(TEACHER_DB_PATH))

    if args.teacher not in db:
        print(f"Teacher '{args.teacher}' not in DB.")
        return

    paper = generate_sample_paper(
        teacher_name=args.teacher,
        subject=args.subject,
        teacher_db=db,
        total_marks=100,
    )
    print(format_paper_output(paper))

    # Save to outputs
    from config.settings import OUTPUTS_DIR
    import json
    out_path = OUTPUTS_DIR / f"predicted_paper_{args.subject}_{args.teacher}.json"
    with open(out_path, 'w') as f:
        json.dump(paper, f, indent=2)
    print(f"\nSaved to: {out_path}")


def cmd_list_rooms(args):
    from modules.room_manager import list_rooms
    rooms = list_rooms(str(DATA_DIR))
    if not rooms:
        print("No subject rooms found. Create one with --action create_room")
        return
    print(f"\n── Subject Rooms ({len(rooms)}) ─────────────────────")
    for r in rooms:
        print(f"  • {r['subject']}  (Teacher: {r['teacher']})")


def cmd_check_ollama(args):
    from modules.ollama_chat import is_ollama_running, list_available_models
    running = is_ollama_running()
    print(f"Ollama running: {'✓ YES' if running else '✗ NO'}")
    if running:
        models = list_available_models()
        print(f"Available models: {models}")
        print(f"Configured model: {OLLAMA_MODEL}")


# ── Argument Parser ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="FarrahAI — Subject-Wise Exam Intelligence System"
    )
    parser.add_argument("--action", required=True,
        choices=["create_room", "upload", "ask", "predict",
                 "sample_paper", "list_rooms", "check_ollama"],
        help="Action to perform"
    )
    parser.add_argument("--subject", help="Subject room name")
    parser.add_argument("--teacher", help="Teacher name")
    parser.add_argument("--file",    help="Path to file for upload")
    parser.add_argument("--query",   help="Question to ask")

    args = parser.parse_args()

    actions = {
        "create_room":  cmd_create_room,
        "upload":       cmd_upload,
        "ask":          cmd_ask,
        "predict":      cmd_predict,
        "sample_paper": cmd_sample_paper,
        "list_rooms":   cmd_list_rooms,
        "check_ollama": cmd_check_ollama,
    }

    try:
        actions[args.action](args)
    except KeyboardInterrupt:
        print("\nAborted.")
    except Exception as e:
        logger.error(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
