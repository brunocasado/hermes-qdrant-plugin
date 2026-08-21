#!/usr/bin/env python3
"""Run file-discovery retrieval benchmarks and report Recall@K + MRR."""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

import yaml

PLUGIN_ROOT = Path(__file__).resolve().parent.parent
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

import core
import registry


def compute_metrics(rows: list[dict]) -> dict[str, float]:
    if not rows:
        return {"Recall@1": 0.0, "Recall@3": 0.0, "Recall@5": 0.0, "MRR": 0.0}
    totals = {1: 0, 3: 0, 5: 0}
    reciprocal = 0.0
    for row in rows:
        expected = set(row["expected"])
        returned = row["returned"]
        first_rank = next((rank for rank, path in enumerate(returned, 1) if path in expected), None)
        for k in totals:
            if first_rank is not None and first_rank <= k:
                totals[k] += 1
        if first_rank is not None:
            reciprocal += 1.0 / first_rank
    n = len(rows)
    return {
        "Recall@1": totals[1] / n,
        "Recall@3": totals[3] / n,
        "Recall@5": totals[5] / n,
        "MRR": reciprocal / n,
    }


async def run_queries(entries: list[dict]) -> list[dict]:
    rows = []
    for entry in entries:
        root = str(PLUGIN_ROOT if entry.get("project") == "self" else Path(entry["root"]).expanduser().resolve())
        collection = entry.get("collection") or registry.collection_for_root(root)
        if not collection:
            raise RuntimeError(f"No collection registered for {root}; reindex it first")
        hits = await core.search_qdrant(collection, entry["query"], limit=60)
        files = core.aggregate_hits_by_file(hits, top_chunks_per_file=2, query=entry["query"])
        returned = [(item.get("rel_path") or item.get("file")) for item in files[:8]]
        rows.append({**entry, "collection": collection, "returned": returned})
    return rows


def render_results(rows: list[dict], metrics: dict[str, float]) -> str:
    lines = ["# Retrieval benchmark", "", "| Metric | Value |", "|---|---:|"]
    for name in ("Recall@1", "Recall@3", "Recall@5", "MRR"):
        lines.append(f"| {name} | {metrics[name]:.3f} |")
    lines += ["", "## Misses"]
    misses = 0
    for row in rows:
        if not set(row["expected"]) & set(row["returned"][:5]):
            misses += 1
            lines.append(f"- {row['query']} — expected {row['expected']}; got {row['returned'][:5]}")
    if not misses:
        lines.append("- None")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", default=str(Path(__file__).with_name("queries.yaml")))
    parser.add_argument("--output", default=str(Path(__file__).with_name("RESULTS.md")))
    args = parser.parse_args()
    entries = yaml.safe_load(Path(args.queries).read_text())
    rows = asyncio.run(run_queries(entries))
    metrics = compute_metrics(rows)
    report = render_results(rows, metrics)
    Path(args.output).write_text(report)
    print(report, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
