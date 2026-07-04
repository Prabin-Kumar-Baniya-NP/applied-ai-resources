"""
E-commerce Semantic Cache + Automatic Category Classifier (Redis + RedisVL)
===========================================================================

An e-commerce assistant answers three KINDS of questions:

    • faq      → general store questions   ("What is your return policy?")
    • product  → questions about a product ("Is the headphone waterproof?")
    • order    → questions about MY order  ("Where is my order?")  ← personalized!

This script does two things, end to end:

    1. CLASSIFY   Decide which category a question belongs to — automatically.
                  We hard-code a small set of LABELLED example questions, store them
                  in a Redis vector index, and classify any new question by k-NN:
                  embed it, find the nearest labelled examples, take a weighted vote.
                  (The incoming demo questions carry NO category — the classifier
                  figures it out.)

    2. CACHE      Serve the answer from a semantic cache, scoped per category (and
                  per user for personalized 'order' questions), with per-category
                  similarity thresholds and TTLs. Off-topic questions bypass the
                  cache and go straight to the LLM (safe default).

See EXPLANATION.md for the architecture and diagrams.

RUN (from the repo root):
    python3 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    cp .env.example .env          # put your OPENAI_API_KEY inside .env
    python context-engineering/Semantic-Caching/resources/redis_semantic_cache.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from time import perf_counter

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())  # loads OPENAI_API_KEY + REDIS_URL from the repo-root .env

import os

from openai import OpenAI
from redisvl.index import SearchIndex
from redisvl.query import VectorQuery
from redisvl.schema import IndexSchema
from redisvl.utils.vectorize import OpenAITextVectorizer

try:  # RedisVL moved this class between versions
    from redisvl.extensions.cache.llm import SemanticCache
except ImportError:  # older redisvl
    from redisvl.extensions.llmcache import SemanticCache

# =========================================================================== #
# Configuration
# =========================================================================== #

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

EMBED_MODEL = "text-embedding-3-small"   # text -> 1536-dim vector
CHAT_MODEL = "gpt-4o-mini"               # generates answers on a cache MISS
EMBED_DIM = 1536

UNKNOWN = "unknown"   # category label for off-topic / low-confidence questions


@dataclass(frozen=True)
class CategoryConfig:
    """What makes one category's CACHE behave differently."""
    name: str
    distance_threshold: float  # RedisVL distance: lower = stricter match
    ttl_seconds: int           # how long an answer stays cached
    user_scoped: bool          # personalized? -> scope the cache key by user_id


# distance ≈ 1 - cosine_similarity  (0 = identical … 2 = opposite)
CATEGORIES: dict[str, CategoryConfig] = {
    #                        name        dist   ttl      user_scoped
    "faq":     CategoryConfig("faq",     0.20,  86_400,  False),  # loose, 1 day
    "product": CategoryConfig("product", 0.15,  3_600,   False),  # medium, 1 hour
    "order":   CategoryConfig("order",   0.10,  300,     True),   # strict, 5 min, per-user
}

# --- Classifier tuning knobs ---
KNN_K = 5            # how many nearest labelled examples to look at
SIM_FLOOR = 0.45     # nearest example must be at least this similar, else UNKNOWN
VOTE_FLOOR = 0.55    # winning category must hold at least this share of the vote

INTENT_INDEX_NAME = "intent_examples"


# =========================================================================== #
# 1) LABELLED TRAINING QUESTIONS  (hard-coded → stored in Redis)
#    These teach the classifier what each category "looks like". Add more to
#    make it smarter — no code changes needed elsewhere.
# =========================================================================== #

TRAINING_EXAMPLES: list[dict] = [
    # ---- faq ----
    {"text": "What is your return policy?",                 "category": "faq"},
    {"text": "How many days do I have to return an item?",  "category": "faq"},
    {"text": "Do you offer free shipping?",                 "category": "faq"},
    {"text": "What payment methods do you accept?",         "category": "faq"},
    {"text": "How can I contact customer support?",         "category": "faq"},
    {"text": "Do you ship internationally?",                "category": "faq"},
    {"text": "Can I use a discount code at checkout?",       "category": "faq"},

    # ---- product ----
    {"text": "Is the Aura wireless headphone waterproof?",  "category": "product"},
    {"text": "How long does the Aura headphone battery last?", "category": "product"},
    {"text": "What colors does the Aura headphone come in?", "category": "product"},
    {"text": "Does the Aura headphone support bluetooth?",  "category": "product"},
    {"text": "What is the warranty on the Aura headphone?", "category": "product"},
    {"text": "Is the laptop stand height adjustable?",      "category": "product"},
    {"text": "What material is the backpack made of?",      "category": "product"},

    # ---- order ----
    {"text": "Where is my order?",                          "category": "order"},
    {"text": "Track my package.",                           "category": "order"},
    {"text": "When will my order arrive?",                  "category": "order"},
    {"text": "Why is my delivery late?",                    "category": "order"},
    {"text": "Can I change the shipping address on my order?", "category": "order"},
    {"text": "How do I cancel my order?",                   "category": "order"},
]


# =========================================================================== #
# Shared singletons (embedding model + OpenAI client)
# =========================================================================== #

_vectorizer: OpenAITextVectorizer | None = None
_openai_client: OpenAI | None = None
_caches: dict[str, SemanticCache] = {}


def _vec() -> OpenAITextVectorizer:
    global _vectorizer
    if _vectorizer is None:
        _vectorizer = OpenAITextVectorizer(model=EMBED_MODEL)
    return _vectorizer


def _llm() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
    return _openai_client


# =========================================================================== #
# 2) THE CLASSIFIER  (k-NN over the Redis intent index)
# =========================================================================== #

def build_intent_index() -> SearchIndex:
    """Create the Redis vector index and load the labelled training questions."""
    schema = IndexSchema.from_dict({
        "index": {"name": INTENT_INDEX_NAME, "prefix": "intent", "storage_type": "hash"},
        "fields": [
            {"name": "text", "type": "text"},
            {"name": "category", "type": "tag"},
            {"name": "embedding", "type": "vector", "attrs": {
                "dims": EMBED_DIM, "distance_metric": "cosine",
                "algorithm": "flat", "datatype": "float32"}},
        ],
    })
    index = SearchIndex(schema, redis_url=REDIS_URL)
    index.create(overwrite=True, drop=True)

    texts = [ex["text"] for ex in TRAINING_EXAMPLES]
    vectors = _vec().embed_many(texts, as_buffer=True)   # batch-embed all examples
    index.load([
        {"text": ex["text"], "category": ex["category"], "embedding": v}
        for ex, v in zip(TRAINING_EXAMPLES, vectors)
    ])
    return index


@dataclass
class Classification:
    category: str
    confidence: float          # 0..1 share of the vote held by the winner
    neighbors: list[tuple[str, str, float]]  # (text, category, similarity)


def classify(question: str, index: SearchIndex, k: int = KNN_K) -> Classification:
    """Label a question by voting among its k nearest labelled examples."""
    qv = _vec().embed(question)   # list[float]
    results = index.query(VectorQuery(
        vector=qv, vector_field_name="embedding",
        return_fields=["text", "category"], num_results=k,
    ))
    if not results:
        return Classification(UNKNOWN, 0.0, [])

    neighbors = [(r["text"], r["category"], 1.0 - float(r["vector_distance"]))
                 for r in results]

    # Guard 1: if even the closest example is unlike the question, it's off-topic.
    if neighbors[0][2] < SIM_FLOOR:
        return Classification(UNKNOWN, neighbors[0][2], neighbors)

    # Weighted vote: each neighbor contributes its similarity to its category.
    votes: dict[str, float] = {}
    for _, cat, sim in neighbors:
        votes[cat] = votes.get(cat, 0.0) + sim
    winner = max(votes, key=lambda c: votes[c])
    confidence = votes[winner] / sum(votes.values())

    # Guard 2: if the winner isn't a clear majority, treat as ambiguous.
    if confidence < VOTE_FLOOR:
        return Classification(UNKNOWN, confidence, neighbors)

    return Classification(winner, confidence, neighbors)


# =========================================================================== #
# 3) THE SEMANTIC CACHE  (scoped per category / per user)
# =========================================================================== #

def scope_key(category: str, user_id: str | None) -> str:
    """The cache namespace. Personalized categories are scoped by user_id."""
    if CATEGORIES[category].user_scoped:
        return f"ecom:{category}:{user_id or 'anonymous'}"
    return f"ecom:{category}"


def get_cache(category: str, user_id: str | None) -> SemanticCache:
    """Return (creating if needed) the SemanticCache for this scope."""
    key = scope_key(category, user_id)
    if key not in _caches:
        cfg = CATEGORIES[category]
        _caches[key] = SemanticCache(
            name=key, redis_url=REDIS_URL, vectorizer=_vec(),
            distance_threshold=cfg.distance_threshold, ttl=cfg.ttl_seconds,
        )
    return _caches[key]


def generate_answer(question: str, category: str) -> str:
    """The expensive path: ask the LLM. Only called on a MISS or a BYPASS."""
    resp = _llm().chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content":
                f"You are an e-commerce assistant answering a {category} question. "
                f"Reply in one or two short sentences."},
            {"role": "user", "content": question},
        ],
    )
    return resp.choices[0].message.content.strip()


# =========================================================================== #
# 4) ONE REQUEST, END TO END:  classify -> route -> cache -> answer
# =========================================================================== #

@dataclass
class Outcome:
    status: str                 # "HIT" | "MISS" | "BYPASS"
    answer: str
    category: str
    confidence: float
    scope: str | None
    threshold: float | None
    distance: float | None
    matched_prompt: str | None
    elapsed_ms: float


def serve(question: str, category: str, confidence: float, user_id: str | None) -> Outcome:
    """Given a KNOWN category, run the cache lifecycle (or bypass if unknown)."""
    t0 = perf_counter()
    ms = lambda: (perf_counter() - t0) * 1000

    # Off-topic / ambiguous -> never cache. Straight to the LLM (safe default).
    if category not in CATEGORIES:
        answer = generate_answer(question, "general")
        return Outcome("BYPASS", answer, category, confidence,
                       None, None, None, None, ms())

    cfg = CATEGORIES[category]
    cache = get_cache(category, user_id)

    hits = cache.check(prompt=question, num_results=1)
    if hits:
        hit = hits[0]
        return Outcome("HIT", hit["response"], category, confidence,
                       scope_key(category, user_id), cfg.distance_threshold,
                       hit.get("vector_distance"), hit.get("prompt"), ms())

    answer = generate_answer(question, category)
    cache.store(prompt=question, response=answer)
    return Outcome("MISS", answer, category, confidence,
                   scope_key(category, user_id), cfg.distance_threshold,
                   None, None, ms())


# =========================================================================== #
# 5) DEMO  —  questions come in with NO category (user_id simulates the login)
# =========================================================================== #

DEMO_QUERIES: list[dict] = [
    # text                                                    logged-in user
    {"text": "What is your return policy?",                   "user": "user_42"},  # faq   → MISS (fills)
    {"text": "Can you explain your return policy?",           "user": "user_42"},  # faq   → HIT of #1
    {"text": "Do you offer free shipping?",                   "user": "user_42"},  # faq   → MISS
    {"text": "Do you have free shipping available?",          "user": "user_42"},  # faq   → HIT of #3
    {"text": "Is the Aura wireless headphone waterproof?",    "user": "user_99"},  # prod  → MISS
    {"text": "Can the Aura wireless headphones get wet?",     "user": "user_99"},  # prod  → HIT of #5
    {"text": "Where is my order?",                            "user": "user_42"},  # order → MISS
    {"text": "Where's my order?",                             "user": "user_42"},  # order → HIT of #7
    {"text": "Where is my order?",                            "user": "user_7"},   # order → MISS (same text, diff user) ⭐
    {"text": "What time does the football match start today?", "user": "user_42"}, # off-topic → BYPASS ⭐
]


# =========================================================================== #
# Presentation helpers
# =========================================================================== #

def _badge(status: str) -> str:
    return {"HIT": "✅ HIT   ", "MISS": "❌ MISS  ", "BYPASS": "⏭  BYPASS"}[status]


def print_training_summary() -> None:
    per_cat: dict[str, int] = {}
    for ex in TRAINING_EXAMPLES:
        per_cat[ex["category"]] = per_cat.get(ex["category"], 0) + 1
    counts = ", ".join(f"{c}={n}" for c, n in per_cat.items())
    print(f"Loaded {len(TRAINING_EXAMPLES)} labelled examples into Redis "
          f"index '{INTENT_INDEX_NAME}'  ({counts})")


def print_outcome(i: int, q: dict, c: Classification, o: Outcome) -> None:
    who = f" user={q['user']}"
    conf = f"{o.confidence:.0%}"
    print(f"\n[{i:02d}] {_badge(o.status)} | category={o.category} (conf {conf}){who}")
    print(f"     Q: {q['text']}")
    top = c.neighbors[0] if c.neighbors else ("—", "—", 0.0)
    print(f"     classified via nearest example: \"{top[0]}\" [{top[1]}] sim={top[2]:.2f}")
    if o.status == "HIT":
        print(f"     cache: scope='{o.scope}'  dist={o.distance:.4f} ≤ {o.threshold}  "
              f"→ matched \"{o.matched_prompt}\"")
    elif o.status == "MISS":
        print(f"     cache: scope='{o.scope}'  no match within {o.threshold}  → called LLM & stored")
    else:
        print(f"     cache: bypassed (off-topic/ambiguous) → called LLM, nothing stored")
    print(f"     time={o.elapsed_ms:.0f}ms   A: {o.answer[:90]}")


def reset_demo_caches(plan: list[tuple[dict, Classification]]) -> None:
    """Clear the cache scopes the demo will touch, for a reproducible run."""
    seen: set[str] = set()
    for q, c in plan:
        if c.category in CATEGORIES:
            key = scope_key(c.category, q.get("user"))
            if key not in seen:
                get_cache(c.category, q.get("user")).clear()
                seen.add(key)


def main() -> None:
    if not OPENAI_API_KEY:
        sys.exit("❌ OPENAI_API_KEY is empty. Put it in the repo-root .env and retry.")

    print("=" * 80)
    print("E-COMMERCE SEMANTIC CACHE + AUTO CATEGORY CLASSIFIER")
    print(f"Redis: {REDIS_URL}   Embeddings: {EMBED_MODEL}   LLM: {CHAT_MODEL}")
    print("=" * 80)

    # Step 1: teach the classifier (store labelled examples in Redis).
    intent_index = build_intent_index()
    print_training_summary()

    # Step 2: classify every incoming question (they carry NO category).
    plan = [(q, classify(q["text"], intent_index)) for q in DEMO_QUERIES]

    # Step 3: reset caches so the demo is reproducible, then serve each request.
    reset_demo_caches(plan)
    print("\nProcessing incoming questions (category is DETECTED, not given):")

    hits = 0
    for i, (q, c) in enumerate(plan, start=1):
        o = serve(q["text"], c.category, c.confidence, q.get("user"))
        hits += int(o.status == "HIT")
        print_outcome(i, q, c, o)

    total = len(DEMO_QUERIES)
    print("\n" + "=" * 80)
    print(f"SUMMARY: {hits}/{total} served from cache "
          f"→ {hits} LLM calls avoided ({hits / total * 100:.0f}% hit rate)")
    print("Tip: edit TRAINING_EXAMPLES (classifier) or DEMO_QUERIES, or tune the "
          "thresholds, and rerun.")
    print("=" * 80)


if __name__ == "__main__":
    main()
