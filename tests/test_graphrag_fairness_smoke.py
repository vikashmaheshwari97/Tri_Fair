import pandas as pd

from src.fairness.graphrag import compute_graphrag_fairness


def test_graphrag_fairness_smoke():
    y_true = ["English", "English", "French", "French"]
    y_pred = ["English", "wrong", "French", "wrong"]
    meta = pd.DataFrame(
        {
            "protected_group": ["A", "A", "B", "B"],
            "retrieval_hit": [1, 1, 1, 0],
        }
    )
    result = compute_graphrag_fairness(
        y_true,
        y_pred,
        meta,
        min_group_count=2,
        group_column="protected_group",
        retrieval_metric_col="retrieval_hit",
    )
    assert result.ready
    assert result.loss >= 0.0
    assert "answer_unfairness" in result.diagnostics
