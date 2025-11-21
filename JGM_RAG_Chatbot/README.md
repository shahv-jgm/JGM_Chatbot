# JGM RAG Chatbot (ADK + Folium)

This is a ready-to-run chatbot that:
- Indexes your CSV/Excel/JSON tables + notebooks + code + graph captions (RAG)
- Answers natural-language questions about the datasets
- Auto-generates charts if your question asks for graphs
- Builds an **offline Folium map** (HTML) if your data has latitude/longitude columns
- Wraps everything inside **Google Agent Development Kit (ADK)** as callable tools

## Quick Start

```bash
# 1) Create a venv (recommended)
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

# 2) Install deps
pip install -r requirements.txt
```

> **Optional (LLM)**: If you want the ADK agent to use a Gemini model for replies beyond tool outputs, create a `.env` file with:
```
GOOGLE_API_KEY=YOUR_KEY
```

### Add your data
Drop files into:
```
jgm_workspace/data/    # CSV, TSV, XLSX, JSON
jgm_workspace/graphs/  # PNG/JPG/SVG/PDF (optional .txt captions with same stem)
jgm_workspace/code/    # ipynb or .py/.md
```

### Run the ADK agent (CLI)
```bash
# From this folder:
adk run .
# then select 'root_agent' when prompted
```

### Run the ADK Web UI
```bash
# in one terminal
adk web --port 8000
# open http://localhost:8000
# choose this folder and the 'root_agent'
```

### Use the tools via chat
Start with:
- `greet` → onboarding prompt
- `set_profile` → pass first_name/last_name/role/contact
- Ask questions like: _"dropout trends by region in 2020?"_
- _"chart dropout rate by region"_ → saves a PNG under `jgm_workspace/`
- _"map"_ → tries to build `jgm_map.html` under `jgm_workspace/`

### Notes
- Auto-charting is heuristic. For better results, name columns clearly and hint in your question (e.g., "chart dropout_rate by region using applicants_2025.csv").
- Folium map requires columns named like `lat/latitude` and `lon/lng/longitude`.
- You can customize retrieval by swapping TF‑IDF with embeddings later.
