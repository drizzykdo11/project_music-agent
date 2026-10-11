# Music Q&A Agent

A Flask web app that answers music questions using **RAG** (retrieval-augmented
generation) and **LLM tool calling**.

## What it does
- Factual questions (genres, theory, instruments, history) are answered from a
  small knowledge base using semantic search.
- Calculation questions ("How long is an eighth note at 90 BPM?") are handled
  by a tool the model calls.
- The model decides per question whether to use retrieved context, the tool,
  both, or neither.

## Tech
Python, Flask, LangChain (Anthropic Claude), sentence-transformers, ChromaDB

## Project structure
```
project_music agent/
├── app.py          # Flask routes
├── agent.py        # LLM agent: combines retrieval and tool calling
├── rag.py          # embeddings + vector search with relevance threshold
├── tool.py         # BPM-to-note-duration calculator tool
├── knowledge.py    # knowledge base documents
├── requirements.txt    # requirements for installing
└── templates/
    └── index1.html
```

## Setup
1. `pip install -r requirements.txt`
2. Copy `.env.example` to `.env` and add your `ANTHROPIC_API_KEY`
   (requires an Anthropic API account with credit).
3. `python app.py`, then open http://127.0.0.1:5000

## Design note: relevance threshold
Vector search always returns the nearest documents, even when nothing relevant
exists. In testing, a relevant question scored a distance of about 0.5, while
an off-topic one ("capital of Japan") scored about 1.9, so results with a
distance of 1.5 or more are discarded. The tradeoff: vague questions land near
the cutoff (the right document for "How long is a quarter note?" ranked second),
so the agent retrieves two candidates instead of one. When nothing passes the
threshold, the model is told no context was found, so it declines instead of
inventing an answer.