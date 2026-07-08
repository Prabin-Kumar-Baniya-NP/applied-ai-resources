# NovaMart Supply-Chain Graph RAG — Runnable Project

A hands-on Graph RAG project on **Neo4j + OpenAI**. It builds a supply-chain knowledge graph
(products → components → suppliers → regions), extracts **risk events** from fake news articles
with an LLM, and answers plain-English questions by turning them into graph traversals
(**Text-to-Cypher**) — answering multi-hop questions that plain vector RAG cannot.

> Architecture & the "why" behind all of this: see [`../sample-project.md`](../sample-project.md).
> Neo4j / Cypher primer: see [`../neo4j.md`](../neo4j.md).

---

## What's in here

```
novamart-supply-graph/
├── docker-compose.yml     # Neo4j Community Edition (the database)
├── README.md              # you are here
├── db.py                  # shared: .env loading, Neo4j driver, schema description
├── 01_load_csv.py         # STEP 1: load structured CSVs → graph nodes/relationships
├── 02_extract_articles.py # STEP 2: LLM extracts risk events from articles → graph
├── 03_ask.py              # STEP 3: Graph RAG (Text-to-Cypher → traverse → answer)
├── reset.py               # wipe the graph
└── data/
    ├── products.csv       # products + monthly revenue
    ├── components.csv     # parts
    ├── suppliers.csv      # suppliers + region + reliability
    ├── bom.csv            # product → component  (MADE_OF)
    ├── sourcing.csv       # component → supplier (SUPPLIED_BY)
    ├── alternatives.csv   # supplier ↔ supplier  (ALTERNATIVE_TO)
    └── articles/          # fake news .txt files (one is a distractor)
```

The graph schema:

```
(:Product)-[:MADE_OF]->(:Component)-[:SUPPLIED_BY]->(:Supplier)-[:LOCATED_IN]->(:Region)
(:Supplier)-[:ALTERNATIVE_TO]->(:Supplier)
(:RiskEvent)-[:AFFECTS]->(:Region)
```

---

## Prerequisites

- **Docker** (to run Neo4j) — `docker --version` should work.
- **Python venv with deps** — from the repo root:
  ```bash
  python3 -m venv .venv && source .venv/bin/activate
  pip install -r requirements.txt      # includes neo4j, openai, python-dotenv
  ```
- **`.env` at the repo root** with your OpenAI key and the Neo4j creds (already added):
  ```
  OPENAI_API_KEY=sk-...
  NEO4J_URI=bolt://localhost:7687
  NEO4J_USER=neo4j
  NEO4J_PASSWORD=novamartgraph
  ```

---

## 1. Start Neo4j (Docker)

From **this folder**:

```bash
docker compose up -d          # pulls the image the first time, then starts Neo4j
docker compose logs -f        # (optional) watch until you see "Started."
```

- **Browser UI:** http://localhost:7474 — log in with `neo4j` / `novamartgraph`
- **Bolt (used by the scripts):** `bolt://localhost:7687`

Stop it later with `docker compose down` (keeps data) or `docker compose down -v` (wipes the
graph volume too).

---

## 2. Run the pipeline

Run these from **this folder** (with the venv activated). Steps 2 and 3 call OpenAI.

```bash
python 01_load_csv.py          # build the graph from the CSVs (Neo4j only)
python 02_extract_articles.py  # LLM extracts risk events from articles (OpenAI)
python 03_ask.py               # ask the 5 demo questions (OpenAI)
```

Ask your own question:

```bash
python 03_ask.py "If we lose ShenzhenParts Co, which products are affected?"
```

Reset and start over:

```bash
python reset.py                # empties the graph; re-run 01_load_csv.py to rebuild
```

---

## 3. See it in the browser

Open http://localhost:7474 and try some Cypher:

```cypher
// See the whole graph
MATCH (n) RETURN n LIMIT 100;

// The multi-hop question, by hand:
// which products are exposed to a Shenzhen risk event?
MATCH (e:RiskEvent)-[:AFFECTS]->(r:Region {name:'Shenzhen'})
MATCH (r)<-[:LOCATED_IN]-(s:Supplier)<-[:SUPPLIED_BY]-(c:Component)<-[:MADE_OF]-(p:Product)
RETURN DISTINCT p.name AS product, sum(p.monthly_revenue) AS revenue_at_risk;
```

---

## How the three LLM "jobs" map to the files

| Job | Where | What the LLM does |
|-----|-------|-------------------|
| **1. Extraction** | `02_extract_articles.py` | free text → `(:RiskEvent)-[:AFFECTS]->(:Region)` triples |
| **2. Text-to-Cypher** | `03_ask.py` → `to_cypher()` | English question → a Cypher traversal (given the schema) |
| **3. Answer generation** | `03_ask.py` → `to_answer()` | graph rows → grounded natural-language answer |

The generated Cypher is printed every time as the **audit trail** — you can always see exactly how
the answer was derived, which is a core advantage of Graph RAG over opaque vector similarity.

---

## Troubleshooting

- **`Could not connect to Neo4j`** — the container isn't up yet. Run `docker compose up -d` and
  wait ~15–30s for Neo4j to finish starting (`docker compose logs -f`).
- **Auth failure** — the password in `.env` (`NEO4J_PASSWORD`) must match `NEO4J_AUTH` in
  `docker-compose.yml` (`novamartgraph`). If you changed one, `docker compose down -v` and restart.
- **Cypher error in step 3** — the LLM occasionally writes imperfect Cypher; the script prints the
  error and moves on. Re-run, or rephrase the question. (Making this robust is a good exercise.)
- **Port already in use** — something else is on 7474/7687; stop it or change the ports in
  `docker-compose.yml`.
