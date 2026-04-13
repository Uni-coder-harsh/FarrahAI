# FarrahAI — Setup Guide (Kali Linux)

## Prerequisites

### 1. Python 3.10+
```bash
python3 --version
# Should be 3.10 or higher
```

### 2. pip
```bash
pip3 --version
```

### 3. Tesseract (system package)
```bash
sudo apt update
sudo apt install tesseract-ocr tesseract-ocr-eng -y
tesseract --version
```

### 4. Ollama (already installed per your setup)
```bash
# Check it works
ollama list

# Start the server (keep this running in a terminal)
ollama serve

# Pull a model (mistral is fast and good)
ollama pull mistral

# Or phi3 if you want something lighter
ollama pull phi3
```

---

## Install Python Dependencies

```bash
cd /home/harsh/Desktop/CUK/FarrahAI

# Create virtual environment (recommended)
python3 -m venv farrahai_env
source farrahai_env/bin/activate

# Install all dependencies
pip install -r requirements.txt
```

> ⚠ PaddleOCR can take a while to install (it pulls paddle framework).
> If it fails, Tesseract is the fallback — that works fine too.

---

## Download the Embedding Model

The `all-MiniLM-L6-v2` model (~80MB) downloads automatically on first use.
To pre-download it:

```bash
python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

---

## Verify Everything Works

```bash
# Check Ollama
python main.py --action check_ollama

# Create a test room
python main.py --action create_room --subject AI_ML --teacher Prof_Sharma

# List rooms
python main.py --action list_rooms
```

---

## Project Order for Demo

Run notebooks in this order:

1. `01_ocr_evaluation.ipynb` — show OCR accuracy (CER/WER)
2. `02_topic_classification.ipynb` — XGBoost/LR/RF comparison
3. `03_clustering.ipynb` — K-Means + hierarchical + 2D plot
4. `04_paper_prediction.ipynb` — teacher profile + sample paper
5. `05_full_pipeline_demo.ipynb` — live end-to-end demo

---

## Common Issues

**PaddleOCR installation fails**
```bash
pip install paddlepaddle -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
pip install paddleocr
```
Or just use Tesseract — change `OCR_ENGINE = "tesseract"` in `config/settings.py`.

**FAISS not found**
```bash
pip install faiss-cpu
```

**Ollama connection refused**
```bash
# Make sure ollama is running first
ollama serve
```

**sentence-transformers slow first run**
Normal — it's downloading the model (~80MB). Only happens once.
