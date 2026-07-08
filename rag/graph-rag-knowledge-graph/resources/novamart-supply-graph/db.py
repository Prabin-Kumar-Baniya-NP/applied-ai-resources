"""
Shared helpers for the NovaMart Supply-Chain Graph RAG project.

- Loads the repo-root .env (OPENAI_API_KEY + NEO4J_* credentials).
- Opens the Neo4j driver.
- Provides tiny run/query helpers and a colour-print utility so every script
  looks consistent in the terminal.

Nothing here talks to OpenAI — that lives in the scripts that need it.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase

# The repo root is 5 levels up from this file:
# novamart-supply-graph/[0] -> resources/[1] -> graph-rag-knowledge-graph/[2]
#   -> rag/[3] -> repo root[4]
ROOT = Path(__file__).resolve().parents[4]
load_dotenv(ROOT / ".env")

DATA_DIR = Path(__file__).parent / "data"

# ANSI colours (no dependency on `rich`)
BOLD, CYAN, GREEN, YELLOW, RED, DIM, RESET = (
    "\033[1m", "\033[36m", "\033[32m", "\033[33m", "\033[31m", "\033[2m", "\033[0m",
)


def banner(title: str) -> None:
    line = "═" * 70
    print(f"\n{CYAN}{line}{RESET}\n{CYAN}{BOLD}  {title}{RESET}\n{CYAN}{line}{RESET}")


def step(msg: str) -> None:
    print(f"{DIM}  ->{RESET} {msg}")


def get_driver():
    """Open a Neo4j driver from the NEO4J_* env vars, with a clear error if down."""
    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    pwd = os.environ.get("NEO4J_PASSWORD")
    if not pwd:
        sys.exit(f"{RED}NEO4J_PASSWORD is not set in {ROOT / '.env'}{RESET}")
    driver = GraphDatabase.driver(uri, auth=(user, pwd))
    try:
        driver.verify_connectivity()
    except Exception as exc:  # noqa: BLE001
        sys.exit(
            f"{RED}Could not connect to Neo4j at {uri}.{RESET}\n"
            f"{DIM}Is the container up? Run:  docker compose up -d{RESET}\n"
            f"{DIM}Details: {exc}{RESET}"
        )
    return driver


def run_write(driver, cypher: str, **params) -> None:
    """Run a write query (no result needed)."""
    with driver.session() as session:
        session.run(cypher, **params)


def run_query(driver, cypher: str, **params) -> list[dict]:
    """Run a read query and return rows as a list of dicts."""
    with driver.session() as session:
        return [record.data() for record in session.run(cypher, **params)]


def require_openai_key() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit(f"{RED}OPENAI_API_KEY is not set in {ROOT / '.env'}{RESET}")


# The graph schema, described in plain text. We hand this to the LLM so it knows
# how to write Cypher (Text-to-Cypher) and how to extract triples.
SCHEMA_DESCRIPTION = """
Node labels and properties:
  (:Product   {sku, name, monthly_revenue})
  (:Component {part_no, name})
  (:Supplier  {name, reliability_score})
  (:Region    {name, country})
  (:RiskEvent {type, severity, summary})

Relationships (all directed):
  (:Product)-[:MADE_OF]->(:Component)
  (:Component)-[:SUPPLIED_BY]->(:Supplier)
  (:Supplier)-[:LOCATED_IN]->(:Region)
  (:Supplier)-[:ALTERNATIVE_TO]->(:Supplier)
  (:RiskEvent)-[:AFFECTS]->(:Region)
""".strip()
