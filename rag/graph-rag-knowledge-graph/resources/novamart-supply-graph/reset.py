"""
Wipe the whole graph (all nodes + relationships). Handy between experiments.

Run:  python reset.py
"""

from db import banner, get_driver, run_write, step


def main() -> None:
    banner("RESET — delete all nodes and relationships")
    driver = get_driver()
    run_write(driver, "MATCH (n) DETACH DELETE n")
    driver.close()
    step("Graph is now empty. Re-run 01_load_csv.py to rebuild.")


if __name__ == "__main__":
    main()
