"""
STEP 1 — Load the structured CSV data into Neo4j.

Reads the 6 CSVs in ./data and builds the graph:
    products, components, suppliers, regions   (nodes)
    MADE_OF, SUPPLIED_BY, LOCATED_IN, ALTERNATIVE_TO   (relationships)

We load through the Python Bolt driver (not Neo4j's LOAD CSV) so it works
regardless of the container's import-folder setup. Everything uses MERGE, so
this script is safe to run repeatedly without creating duplicates.

Run:  python 01_load_csv.py
"""

from __future__ import annotations

import csv

from db import DATA_DIR, banner, get_driver, run_query, run_write, step


def read_csv(name: str) -> list[dict]:
    with (DATA_DIR / name).open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def create_constraints(driver) -> None:
    """Uniqueness constraints = the database-level guard against duplicate entities."""
    constraints = [
        "CREATE CONSTRAINT product_sku IF NOT EXISTS FOR (p:Product) REQUIRE p.sku IS UNIQUE",
        "CREATE CONSTRAINT component_pn IF NOT EXISTS FOR (c:Component) REQUIRE c.part_no IS UNIQUE",
        "CREATE CONSTRAINT supplier_name IF NOT EXISTS FOR (s:Supplier) REQUIRE s.name IS UNIQUE",
        "CREATE CONSTRAINT region_name IF NOT EXISTS FOR (r:Region) REQUIRE r.name IS UNIQUE",
    ]
    for c in constraints:
        run_write(driver, c)
    step("Uniqueness constraints ready.")


def load_products(driver) -> None:
    for row in read_csv("products.csv"):
        run_write(
            driver,
            "MERGE (p:Product {sku:$sku}) "
            "SET p.name=$name, p.monthly_revenue=toInteger($rev)",
            sku=row["sku"], name=row["name"], rev=row["monthly_revenue"],
        )
    step("Loaded products.")


def load_components(driver) -> None:
    for row in read_csv("components.csv"):
        run_write(
            driver,
            "MERGE (c:Component {part_no:$pn}) SET c.name=$name",
            pn=row["part_no"], name=row["name"],
        )
    step("Loaded components.")


def load_suppliers_and_regions(driver) -> None:
    for row in read_csv("suppliers.csv"):
        # Supplier node
        run_write(
            driver,
            "MERGE (s:Supplier {name:$name}) SET s.reliability_score=toFloat($rel)",
            name=row["name"], rel=row["reliability_score"],
        )
        # Region node + LOCATED_IN edge
        run_write(
            driver,
            "MERGE (r:Region {name:$region}) SET r.country=$country "
            "WITH r "
            "MATCH (s:Supplier {name:$name}) "
            "MERGE (s)-[:LOCATED_IN]->(r)",
            region=row["region"], country=row["country"], name=row["name"],
        )
    step("Loaded suppliers + regions (LOCATED_IN).")


def load_bom(driver) -> None:
    for row in read_csv("bom.csv"):
        run_write(
            driver,
            "MATCH (p:Product {sku:$sku}), (c:Component {part_no:$pn}) "
            "MERGE (p)-[:MADE_OF]->(c)",
            sku=row["product_sku"], pn=row["component_part_no"],
        )
    step("Loaded bill-of-materials (MADE_OF).")


def load_sourcing(driver) -> None:
    for row in read_csv("sourcing.csv"):
        run_write(
            driver,
            "MATCH (c:Component {part_no:$pn}), (s:Supplier {name:$sup}) "
            "MERGE (c)-[:SUPPLIED_BY]->(s)",
            pn=row["component_part_no"], sup=row["supplier_name"],
        )
    step("Loaded sourcing (SUPPLIED_BY).")


def load_alternatives(driver) -> None:
    for row in read_csv("alternatives.csv"):
        run_write(
            driver,
            "MATCH (a:Supplier {name:$a}), (b:Supplier {name:$b}) "
            "MERGE (a)-[:ALTERNATIVE_TO]->(b)",
            a=row["supplier_name"], b=row["alternative_supplier_name"],
        )
    step("Loaded alternatives (ALTERNATIVE_TO).")


def summarize(driver) -> None:
    counts = run_query(
        driver,
        "MATCH (n) RETURN labels(n)[0] AS label, count(*) AS n ORDER BY label",
    )
    rels = run_query(
        driver,
        "MATCH ()-[r]->() RETURN type(r) AS rel, count(*) AS n ORDER BY rel",
    )
    print("\n  Nodes:")
    for c in counts:
        print(f"    {c['label']:<12} {c['n']}")
    print("  Relationships:")
    for r in rels:
        print(f"    {r['rel']:<16} {r['n']}")


def main() -> None:
    banner("STEP 1 — Load structured CSV data into Neo4j")
    driver = get_driver()
    create_constraints(driver)
    load_products(driver)
    load_components(driver)
    load_suppliers_and_regions(driver)
    load_bom(driver)
    load_sourcing(driver)
    load_alternatives(driver)
    summarize(driver)
    driver.close()
    step("Done. Open http://localhost:7474 and run:  MATCH (n) RETURN n")


if __name__ == "__main__":
    main()
