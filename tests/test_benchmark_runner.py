from benchmarks.run_benchmark import compute_metrics


def test_metrics_compute_recall_and_mrr():
    rows = [
        {"expected": ["a.py"], "returned": ["a.py", "x.py"]},
        {"expected": ["b.py"], "returned": ["x.py", "b.py"]},
        {"expected": ["c.py"], "returned": ["x.py", "y.py"]},
    ]

    metrics = compute_metrics(rows)

    assert metrics["Recall@1"] == 1 / 3
    assert metrics["Recall@3"] == 2 / 3
    assert metrics["Recall@5"] == 2 / 3
    assert metrics["MRR"] == (1 + 1 / 2 + 0) / 3
