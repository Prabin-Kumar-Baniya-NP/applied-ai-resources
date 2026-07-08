"""
STEP 3 — Graph RAG: answer plain-English questions over the knowledge graph.

For each question this runs the three-step Graph RAG loop:
    1. Text-to-Cypher   : LLM turns the English question into a Cypher query
                          (given the schema). This is the clever core.
    2. Traverse         : Neo4j runs the Cypher and returns exact rows.
    3. Answer generation: LLM turns the rows into a natural-language answer.

The generated Cypher is printed as the AUDIT TRAIL — a key Graph RAG advantage.

By default it runs 5 built-in demo questions. Pass your own question as an arg:
    python 03_ask.py                       # runs the demo questions
    python 03_ask.py "your question here"  # runs a single custom question
"""

from __future__ import annotations

import json
import sys

from openai import OpenAI

from db import (
    BOLD, CYAN, GREEN, RED, RESET, SCHEMA_DESCRIPTION, YELLOW,
    banner, get_driver, require_openai_key, run_query, step,
)

CHAT_MODEL = "gpt-4o-mini"

DEMO_QUESTIONS = [
    "A flood hit Shenzhen. Which products are at risk and how much monthly revenue is exposed?",
    "Which single supplier, if lost, would put the most monthly revenue at risk?",
    "For components affected by the Shenzhen flood, is there an alternative supplier?",
    "Which components are single-sourced (only one supplier)?",
    "Which products are affected by the Taiwan earthquake?",
]

TEXT2CYPHER_SYSTEM = f"""You translate a user's question into a single Cypher query
for Neo4j. Use ONLY this schema:

{SCHEMA_DESCRIPTION}

Rules:
- Return ONLY JSON: {{"cypher": "<the query>"}}
- Use MATCH/WHERE/RETURN; aggregate with sum(), count(), collect() when useful.
- Revenue is p.monthly_revenue (an integer) on :Product. Use DISTINCT to avoid dupes.
- Never invent labels or relationships outside the schema.

IMPORTANT — avoid the two most common mistakes:
1. DIRECTION: to avoid arrow-direction errors, match relationships WITHOUT a
   direction — write -[:REL]- (no arrow), not -[:REL]-> . The label types are
   unambiguous, so undirected matching is safe here.
2. OVER-FILTERING: match a :RiskEvent by its `type` ONLY (e.g. {{type:'flood'}}).
   NEVER filter a RiskEvent on `severity` or `summary`. To find what an event
   affects, follow (:RiskEvent)-[:AFFECTS]-(:Region) — do not guess the region name
   from the question (an "earthquake in Taiwan" may be stored under region 'Taipei').

Canonical patterns (copy these shapes):
- Products exposed to an event of a given type:
  MATCH (e:RiskEvent {{type:'flood'}})-[:AFFECTS]-(r:Region)
  MATCH (r)-[:LOCATED_IN]-(s:Supplier)-[:SUPPLIED_BY]-(c:Component)-[:MADE_OF]-(p:Product)
  RETURN DISTINCT p.name AS product, sum(p.monthly_revenue) AS revenue_at_risk
- Products/revenue that depend on a specific supplier:
  MATCH (s:Supplier {{name:'ShenzhenParts Co'}})-[:SUPPLIED_BY]-(c:Component)-[:MADE_OF]-(p:Product)
  RETURN DISTINCT p.name, sum(p.monthly_revenue) AS revenue
- Revenue at risk per supplier (which supplier matters most):
  MATCH (s:Supplier)-[:SUPPLIED_BY]-(c:Component)-[:MADE_OF]-(p:Product)
  RETURN s.name AS supplier, sum(DISTINCT p.monthly_revenue) AS revenue
  ORDER BY revenue DESC
- Single-sourced components (only one supplier):
  MATCH (c:Component)-[:SUPPLIED_BY]-(s:Supplier)
  WITH c, count(DISTINCT s) AS suppliers WHERE suppliers = 1
  RETURN c.name
- Alternative supplier for a component:
  MATCH (c:Component)-[:SUPPLIED_BY]-(s:Supplier)-[:ALTERNATIVE_TO]-(alt:Supplier)
  RETURN DISTINCT c.name, alt.name"""

ANSWER_SYSTEM = """You are a supply-chain analyst assistant. Given the user's
question and the exact rows returned from the knowledge graph, write a concise,
factual answer. Use ONLY the provided rows — do not invent products, suppliers,
or numbers. Format revenue with a $ and thousands separators."""


def to_cypher(client: OpenAI, question: str) -> str:
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": TEXT2CYPHER_SYSTEM},
            {"role": "user", "content": question},
        ],
    )
    return json.loads(resp.choices[0].message.content)["cypher"]


def to_answer(client: OpenAI, question: str, rows: list[dict]) -> str:
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": ANSWER_SYSTEM},
            {"role": "user",
             "content": f"Question: {question}\n\nGraph rows (JSON):\n{json.dumps(rows, default=str)}"},
        ],
    )
    return resp.choices[0].message.content.strip()


def ask(driver, client: OpenAI, question: str) -> None:
    banner(f"Q: {question}")

    # 1. Text-to-Cypher
    step("Translating question → Cypher ...")
    cypher = to_cypher(client, question)
    print(f"\n  {BOLD}Generated Cypher (audit trail):{RESET}")
    print(f"{CYAN}{cypher}{RESET}")

    # 2. Traverse the graph
    step("Running the query against Neo4j ...")
    try:
        rows = run_query(driver, cypher)
    except Exception as exc:  # noqa: BLE001
        print(f"  {RED}Cypher failed:{RESET} {exc}")
        return
    print(f"\n  {BOLD}Rows returned ({len(rows)}):{RESET}")
    if rows:
        for r in rows:
            print(f"    {r}")
    else:
        print(f"    {YELLOW}(no rows){RESET}")

    # 3. Answer generation
    step("Generating natural-language answer ...")
    answer = to_answer(client, question, rows)
    print(f"\n  {GREEN}{BOLD}Answer:{RESET} {answer}")


def main() -> None:
    require_openai_key()
    client = OpenAI()
    driver = get_driver()

    questions = [" ".join(sys.argv[1:])] if len(sys.argv) > 1 else DEMO_QUESTIONS
    for q in questions:
        ask(driver, client, q)

    driver.close()
    print()
    step("Done.")


if __name__ == "__main__":
    main()
