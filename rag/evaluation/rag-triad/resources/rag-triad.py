"""
RAG Triad — end-to-end demo on Postgres + pgvector.

What this single file does, in order, when you run `python rag-triad.py`:

    1. SETUP      Connect to Postgres, enable pgvector, (re)create the `documents` table.
    2. INGEST     Read sample.csv, embed each row with OpenAI, store the vectors.
    3. QUERY x5   For 5 questions: retrieve context -> generate an answer (basic RAG).
    4. JUDGE      Score every answer with an LLM judge on the RAG Triad:
                    - Context Relevance   (Q <-> retrieved context)
                    - Groundedness        (context <-> answer)  [hallucination check]
                    - Answer Relevance    (answer <-> Q)
    5. REPORT     Print everything to the terminal, plus a summary table.

The RAG Triad is reference-free: it needs no human-written "correct answer",
so the exact same judge can run offline (like here) or on live production traffic.
See ../Introduction.md for the full concept write-up.

Config is read from the repo-root .env:
    OPENAI_API_KEY   used for BOTH embeddings and generation/judging
    DATABASE_URL     e.g. postgresql://user:pass@localhost:5432/appliedaidb
"""

from __future__ import annotations

import csv
import json
import sys
import textwrap
from pathlib import Path

import numpy as np
import psycopg2
from dotenv import load_dotenv
from openai import OpenAI
from pgvector.psycopg2 import register_vector

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

# The repo root is 4 levels up from this file (resources/ -> rag-triad/ ->
# evaluation/ -> rag/ -> repo root), where the shared .env lives.
ROOT = Path(__file__).resolve().parents[4]
load_dotenv(ROOT / ".env")

import os  # noqa: E402  (imported after load_dotenv so os.environ is populated)

CSV_PATH = Path(__file__).parent / "sample.csv"

EMBED_MODEL = "text-embedding-3-small"   # 1536-dim embeddings
EMBED_DIM = 1536
CHAT_MODEL = "gpt-4o-mini"               # used for both answering and judging
TOP_K = 3                                # how many chunks to retrieve per query

# The 5 questions we ask our little knowledge base.
QUERIES = [
    "How long does standard shipping take, and is it ever free?",
    "What is your return policy, and are there any restocking fees?",
    "What do I get with the NovaMart Plus membership?",
    "Which payment methods can I use, and do you offer any pay-later options?",
    "Tell me about the AuroraBook laptop's specs and price.",
]

# Pass/attention thresholds for each leg of the triad (0..1).
THRESHOLDS = {
    "context_relevance": 0.70,
    "groundedness": 0.80,
    "answer_relevance": 0.70,
}

# --------------------------------------------------------------------------- #
# Tiny terminal-formatting helpers (no extra dependencies)
# --------------------------------------------------------------------------- #

BOLD, CYAN, GREEN, YELLOW, RED, DIM, RESET = (
    "\033[1m", "\033[36m", "\033[32m", "\033[33m", "\033[31m", "\033[2m", "\033[0m",
)


def banner(title: str) -> None:
    line = "═" * 70
    print(f"\n{CYAN}{line}{RESET}")
    print(f"{CYAN}{BOLD}  {title}{RESET}")
    print(f"{CYAN}{line}{RESET}")


def step(msg: str) -> None:
    print(f"{DIM}  ->{RESET} {msg}")


def wrap(text: str, indent: str = "     ") -> str:
    return textwrap.fill(text, width=90, initial_indent=indent, subsequent_indent=indent)


def score_badge(score: float, threshold: float) -> str:
    """Colour + PASS/WARN label for a triad score."""
    if score >= threshold:
        return f"{GREEN}{score:.2f}  PASS{RESET}"
    return f"{YELLOW}{score:.2f}  WARN{RESET}"


# --------------------------------------------------------------------------- #
# Step 1 — Database setup
# --------------------------------------------------------------------------- #

def connect_db() -> "psycopg2.extensions.connection":
    url = os.environ.get("DATABASE_URL")
    if not url:
        sys.exit(f"{RED}DATABASE_URL is not set in {ROOT / '.env'}{RESET}")
    conn = psycopg2.connect(url)
    conn.autocommit = True
    return conn


def setup_schema(conn) -> None:
    """Enable pgvector and (re)create a clean `documents` table."""
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        # Register the vector type on THIS connection so Python lists/arrays
        # adapt to the pgvector column automatically.
        register_vector(conn)
        cur.execute("DROP TABLE IF EXISTS documents;")
        cur.execute(
            f"""
            CREATE TABLE documents (
                id        SERIAL PRIMARY KEY,
                category  TEXT NOT NULL,
                content   TEXT NOT NULL,
                embedding VECTOR({EMBED_DIM})
            );
            """
        )
    step("pgvector extension ready; `documents` table created fresh.")


# --------------------------------------------------------------------------- #
# Step 2 — Ingest: read CSV, embed, store
# --------------------------------------------------------------------------- #

def embed_texts(client: OpenAI, texts: list[str]) -> list[np.ndarray]:
    """One batched embedding call for a list of texts."""
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    # resp.data is returned in the same order as the input.
    return [np.array(d.embedding, dtype=np.float32) for d in resp.data]


def ingest_csv(conn, client: OpenAI) -> int:
    rows = list(csv.DictReader(CSV_PATH.open(encoding="utf-8")))
    step(f"Loaded {len(rows)} rows from {CSV_PATH.name}.")

    # Each CSV row is one "chunk". We embed the content (category prepended so
    # the category words are part of the semantic signal).
    chunks = [f"[{r['category']}] {r['content']}" for r in rows]
    step(f"Embedding {len(chunks)} chunks with {EMBED_MODEL} ...")
    vectors = embed_texts(client, chunks)

    with conn.cursor() as cur:
        for r, vec in zip(rows, vectors):
            cur.execute(
                "INSERT INTO documents (category, content, embedding) VALUES (%s, %s, %s);",
                (r["category"], r["content"], vec),
            )
    step(f"Stored {len(rows)} embeddings in Postgres.")
    return len(rows)


# --------------------------------------------------------------------------- #
# Step 3 — Retrieve + Generate (the actual RAG)
# --------------------------------------------------------------------------- #

def retrieve(conn, client: OpenAI, query: str, k: int = TOP_K) -> list[str]:
    """Embed the query and return the k most similar chunks (cosine distance)."""
    q_vec = embed_texts(client, [query])[0]
    with conn.cursor() as cur:
        cur.execute(
            "SELECT content FROM documents ORDER BY embedding <=> %s LIMIT %s;",
            (q_vec, k),
        )
        return [row[0] for row in cur.fetchall()]


def generate_answer(client: OpenAI, query: str, contexts: list[str]) -> str:
    """Answer the query using ONLY the retrieved context."""
    context_block = "\n".join(f"- {c}" for c in contexts)
    system = (
        "You are a helpful customer-support assistant for the online store NovaMart. "
        "Answer the user's question using ONLY the information in the provided context. "
        "If the context does not contain the answer, say you don't have that information. "
        "Be concise and specific."
    )
    user = f"Context:\n{context_block}\n\nQuestion: {query}"
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
    )
    return resp.choices[0].message.content.strip()


# --------------------------------------------------------------------------- #
# Step 4 — The LLM-as-a-Judge: the RAG Triad
# --------------------------------------------------------------------------- #

JUDGE_SYSTEM = """You are a strict, impartial evaluator of a RAG (retrieval-augmented
generation) system. You score three things on a 0.0 to 1.0 scale and you MUST justify
each score in one short sentence. Be critical: only give high scores when clearly deserved.

Score these three legs of the RAG Triad:

1. context_relevance: Of the retrieved context passages, what fraction are actually
   relevant to answering the QUESTION? (1.0 = every passage is on-topic, 0.0 = none are.)

2. groundedness: Is every claim in the ANSWER supported by the CONTEXT? Break the answer
   into individual claims; score = (claims supported by context) / (total claims).
   This detects hallucination. (1.0 = fully grounded, 0.0 = fabricated.)

3. answer_relevance: Does the ANSWER actually address what the QUESTION asked?
   (1.0 = directly and completely answers it, 0.0 = off-topic or evasive.)

Return ONLY a JSON object with this exact shape:
{
  "context_relevance": {"score": <float>, "reason": "<one sentence>"},
  "groundedness":      {"score": <float>, "reason": "<one sentence>"},
  "answer_relevance":  {"score": <float>, "reason": "<one sentence>"}
}"""


def judge_triad(client: OpenAI, query: str, contexts: list[str], answer: str) -> dict:
    context_block = "\n".join(f"- {c}" for c in contexts)
    user = (
        f"QUESTION:\n{query}\n\n"
        f"CONTEXT:\n{context_block}\n\n"
        f"ANSWER:\n{answer}"
    )
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": JUDGE_SYSTEM},
                  {"role": "user", "content": user}],
    )
    return json.loads(resp.choices[0].message.content)


# --------------------------------------------------------------------------- #
# Step 5 — Orchestration + reporting
# --------------------------------------------------------------------------- #

def run_query(conn, client: OpenAI, idx: int, query: str) -> dict:
    banner(f"QUERY {idx}:  {query}")

    step("Retrieving top context from pgvector ...")
    contexts = retrieve(conn, client, query)
    print(f"\n  {BOLD}Retrieved context ({len(contexts)} chunks):{RESET}")
    for i, c in enumerate(contexts, 1):
        print(f"    {DIM}[{i}]{RESET} {c}")

    step("Generating answer with " + CHAT_MODEL + " ...")
    answer = generate_answer(client, query, contexts)
    print(f"\n  {BOLD}Answer:{RESET}")
    print(wrap(answer))

    step("Judging with the RAG Triad ...")
    verdict = judge_triad(client, query, contexts, answer)

    print(f"\n  {BOLD}RAG Triad scores:{RESET}")
    for leg in ("context_relevance", "groundedness", "answer_relevance"):
        s = float(verdict[leg]["score"])
        reason = verdict[leg]["reason"]
        label = leg.replace("_", " ").title()
        print(f"    {label:<20} {score_badge(s, THRESHOLDS[leg])}")
        print(f"    {DIM}{reason}{RESET}")

    return {
        "query": query,
        "scores": {leg: float(verdict[leg]["score"])
                   for leg in THRESHOLDS},
    }


def print_summary(results: list[dict]) -> None:
    banner("SUMMARY — RAG Triad across all queries")
    header = f"  {'#':<5}{'Context Rel.':<15}{'Groundedness':<15}{'Answer Rel.':<15}Query"
    print(f"{BOLD}{header}{RESET}")
    print(f"  {'-' * 96}")
    for i, r in enumerate(results, 1):
        cr = r["scores"]["context_relevance"]
        gr = r["scores"]["groundedness"]
        ar = r["scores"]["answer_relevance"]
        q = (r["query"][:45] + "...") if len(r["query"]) > 48 else r["query"]
        print(f"  {i:<5}{cr:<15.2f}{gr:<15.2f}{ar:<15.2f}{q}")

    # Averages — this is what you'd track over time / alert on.
    n = len(results)
    print(f"  {'-' * 96}")
    avg = {leg: sum(r["scores"][leg] for r in results) / n for leg in THRESHOLDS}
    print(f"  {BOLD}{'AVG':<5}"
          f"{avg['context_relevance']:<15.2f}"
          f"{avg['groundedness']:<15.2f}"
          f"{avg['answer_relevance']:<15.2f}{RESET}{DIM}(average across the test set){RESET}")
    print()


def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit(f"{RED}OPENAI_API_KEY is not set in {ROOT / '.env'}{RESET}")

    client = OpenAI()

    banner("STEP 1 — Database setup (Postgres + pgvector)")
    conn = connect_db()
    setup_schema(conn)

    banner("STEP 2 — Ingest CSV -> embeddings -> pgvector")
    ingest_csv(conn, client)

    # Steps 3 + 4 happen inside run_query for each question.
    results = [run_query(conn, client, i, q) for i, q in enumerate(QUERIES, 1)]

    print_summary(results)
    conn.close()
    step("Done. Connection closed.")


if __name__ == "__main__":
    main()
