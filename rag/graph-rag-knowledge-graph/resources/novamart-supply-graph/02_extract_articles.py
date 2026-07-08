"""
STEP 2 — LLM extraction: turn unstructured news articles into graph triples.

For each .txt file in ./data/articles, we ask OpenAI to extract any supply-chain
RISK EVENT and the REGION it affects, as JSON. Then we MERGE it into the graph as:
    (:RiskEvent {type, severity, summary})-[:AFFECTS]->(:Region {name})

This is "Job 1" from the sample-project architecture. Note one article is a pure
marketing update with NO risk event — the LLM should correctly extract nothing,
which demonstrates relevance filtering.

Run:  python 02_extract_articles.py   (uses OPENAI_API_KEY)
"""

from __future__ import annotations

import json

from openai import OpenAI

from db import (
    DATA_DIR, DIM, GREEN, RESET, YELLOW, banner, get_driver, require_openai_key,
    run_query, run_write, step,
)

CHAT_MODEL = "gpt-4o-mini"

EXTRACTION_SYSTEM = """You extract supply-chain risk events from news text for a
knowledge graph. Return ONLY JSON.

If the article describes a real supply-chain disruption (flood, earthquake, strike,
fire, ban, shortage, etc.), return:
{
  "has_risk_event": true,
  "type": "<flood|earthquake|strike|fire|ban|shortage|other>",
  "region": "<the city/region name affected>",
  "severity": "<low|moderate|high>",
  "summary": "<one short sentence>"
}

If the article is NOT about a supply-chain disruption (e.g. marketing, finance,
general news), return exactly:
{ "has_risk_event": false }"""


def extract_article(client: OpenAI, text: str) -> dict:
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": EXTRACTION_SYSTEM},
            {"role": "user", "content": text},
        ],
    )
    return json.loads(resp.choices[0].message.content)


def store_event(driver, ev: dict) -> None:
    """MERGE the RiskEvent and its AFFECTS edge to an existing/new Region."""
    run_write(
        driver,
        """
        MERGE (r:Region {name:$region})
        MERGE (e:RiskEvent {type:$type, region:$region})
        SET e.severity=$severity, e.summary=$summary
        MERGE (e)-[:AFFECTS]->(r)
        """,
        region=ev["region"], type=ev["type"],
        severity=ev.get("severity", "unknown"), summary=ev.get("summary", ""),
    )


def main() -> None:
    banner("STEP 2 — LLM extracts risk events from articles → graph")
    require_openai_key()
    client = OpenAI()
    driver = get_driver()

    articles = sorted((DATA_DIR / "articles").glob("*.txt"))
    step(f"Found {len(articles)} articles.")

    for path in articles:
        text = path.read_text(encoding="utf-8")
        ev = extract_article(client, text)
        print(f"\n  {DIM}{path.name}{RESET}")
        if ev.get("has_risk_event"):
            store_event(driver, ev)
            print(f"    {GREEN}RISK EVENT{RESET} "
                  f"{ev['type']} in {ev['region']} (severity: {ev.get('severity')})")
            print(f"    {DIM}{ev.get('summary','')}{RESET}")
        else:
            print(f"    {YELLOW}no risk event{RESET} (correctly skipped)")

    events = run_query(
        driver,
        "MATCH (e:RiskEvent)-[:AFFECTS]->(r:Region) "
        "RETURN e.type AS type, r.name AS region, e.severity AS severity ORDER BY region",
    )
    print("\n  RiskEvents now in the graph:")
    for e in events:
        print(f"    {e['type']:<12} → {e['region']:<14} ({e['severity']})")
    driver.close()
    step("Done. Risk events linked to regions via AFFECTS.")


if __name__ == "__main__":
    main()
