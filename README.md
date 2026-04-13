# FarrahAI 

**Subject-Wise Exam Intelligence System**

> Upload notes. Ask questions. Predict what's coming in the exam.

---

## What is FarrahAI?

FarrahAI is a subject-wise AI system where students join a **server room** for their subject, upload notes and past question papers, and get:

- ✅ Answers grounded in uploaded notes (not hallucinated)
- ✅ Topic importance ranking based on past papers
- ✅ Teacher profile-based question paper prediction
- ✅ OCR quality measurement (CER / WER)
- ✅ ML-based topic classification (XGBoost, Logistic Regression)
- ✅ Unsupervised clustering of questions and topics

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        FarrahAI                             │
│                                                             │
│  ┌──────────────┐   ┌──────────────┐   ┌───────────────┐   │
│  │  Server Room │   │  Teacher DB  │   │  Global Store │   │
│  │  (per subj.) │   │  (global)    │   │  question ppr │   │
│  └──────┬───────┘   └──────┬───────┘   └───────┬───────┘   │
│         │                  │                   │           │
│         ▼                  ▼                   ▼           │
│  ┌─────────────────────────────────────────────────────┐   │
│  │               Pipeline                              │   │
│  │  Upload → Preprocess → OCR → Clean → Chunk         │   │
│  │       → Embed → FAISS Index → Retrieve             │   │
│  └──────────────────────┬──────────────────────────────┘   │
│                         │                                   │
│         ┌───────────────┼───────────────┐                  │
│         ▼               ▼               ▼                  │
│  ┌────────────┐ ┌──────────────┐ ┌──────────────────┐      │
│  │  Q&A via   │ │ Topic Class. │ │ Paper Prediction │      │
│  │  Ollama    │ │  XGBoost/LR  │ │  Teacher Profile │      │
│  └────────────┘ └──────────────┘ └──────────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
FarrahAI/
├── config/
│   └── settings.py          # Paths, model names, constants
├── data/
│   ├── raw/                 # Original uploaded files
│   ├── processed/           # Cleaned text after OCR
│   ├── embeddings/          # FAISS indexes per subject
│   ├── knowledge_base/      # Chunked text per subject
│   └── question_papers/     # Past papers per teacher/subject
├── modules/
│   ├── preprocess.py        # OpenCV image preprocessing
│   ├── ocr.py               # OCR + quality metrics (CER/WER)
│   ├── chunker.py           # Text cleaning and chunking
│   ├── embedder.py          # Sentence embeddings + FAISS
│   ├── retriever.py         # Query → retrieve relevant chunks
│   ├── ml_models.py         # XGBoost, LR, clustering
│   ├── teacher_profile.py   # Teacher DB + pattern analysis
│   ├── predictor.py         # Question paper prediction
│   └── ollama_chat.py       # Ollama integration for Q&A
├── notebooks/
│   ├── 01_ocr_evaluation.ipynb
│   ├── 02_topic_classification.ipynb
│   ├── 03_clustering.ipynb
│   ├── 04_paper_prediction.ipynb
│   └── 05_full_pipeline_demo.ipynb
├── models/                  # Saved ML models (.pkl)
├── outputs/                 # Reports, charts, predictions
├── main.py                  # CLI entry point
├── requirements.txt
└── README.md
```

---

## How Server Rooms Work

- Each **subject** gets its own server room (folder + FAISS index + knowledge base)
- Students upload notes → processed into that subject's knowledge base
- **Teacher profiles are global** — one teacher can teach multiple subjects
- Question paper prediction uses: teacher profile + subject knowledge base

---

## Setup

```bash
git clone https://github.com/Uni-coder-harsh/FarrahAI.git
cd FarrahAI
pip install -r requirements.txt

# Make sure Ollama is running
ollama serve
ollama pull mistral   # or any model you prefer
```

---

## Run

```bash
# Create a subject room
python main.py --action create_room --subject "AI_ML"

# Upload notes to a room
python main.py --action upload --subject "AI_ML" --file path/to/notes.jpg

# Ask a question
python main.py --action ask --subject "AI_ML" --query "What is backpropagation?"

# Predict important topics
python main.py --action predict --subject "AI_ML" --teacher "Dr_Rakesh_Godi"
```

---

## ML Models Used

| Task | Algorithm | Metric |
|------|-----------|--------|
| Topic classification | XGBoost, Logistic Regression | Accuracy, F1 |
| Question type classification | Random Forest | Precision, Recall |
| Topic importance prediction | XGBoost | Top-k Accuracy |
| Topic discovery | K-Means, Hierarchical | Silhouette Score |
| Answer retrieval | FAISS + Cosine Similarity | Recall@k, MRR |
| OCR quality | CER / WER | Lower = Better |

---

## Tech Stack

| Layer | Tool |
|-------|------|
| Image processing | OpenCV |
| OCR | PaddleOCR / Tesseract |
| ML | scikit-learn, XGBoost |
| Embeddings | sentence-transformers |
| Vector search | FAISS |
| LLM | Ollama (local) |
| Notebooks | Jupyter in VS Code |
| Storage | SQLite + local files |

---

*Built for AI/ML course project — but designed like a real system.*
