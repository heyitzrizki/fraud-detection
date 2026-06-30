import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score


def simulate_thresholds(
    y_true,
    y_scores,
    transaction_amounts,
    thresholds=None,
    investigation_cost_per_alert: float = 5.0,
) -> pd.DataFrame:
    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 1.00, 0.05), 2)

    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    transaction_amounts = np.asarray(transaction_amounts)

    rows = []
    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)

        true_positive_mask = (y_true == 1) & (y_pred == 1)
        false_positive_mask = (y_true == 0) & (y_pred == 1)
        true_negative_mask = (y_true == 0) & (y_pred == 0)
        false_negative_mask = (y_true == 1) & (y_pred == 0)

        flagged_transactions = int(y_pred.sum())
        captured_fraud_amount = float(transaction_amounts[true_positive_mask].sum())
        investigation_cost = float(flagged_transactions * investigation_cost_per_alert)
        missed_fraud_loss = float(transaction_amounts[false_negative_mask].sum())

        rows.append(
            {
                "threshold": float(threshold),
                "true_positives": int(true_positive_mask.sum()),
                "false_positives": int(false_positive_mask.sum()),
                "true_negatives": int(true_negative_mask.sum()),
                "false_negatives": int(false_negative_mask.sum()),
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "flagged_transactions": flagged_transactions,
                "estimated_fraud_amount_captured": captured_fraud_amount,
                "investigation_cost": investigation_cost,
                "estimated_missed_fraud_loss": missed_fraud_loss,
                "estimated_net_benefit": captured_fraud_amount - investigation_cost,
            }
        )

    return pd.DataFrame(rows)


def select_best_threshold(threshold_results: pd.DataFrame) -> float:
    best_index = threshold_results["estimated_net_benefit"].idxmax()
    return float(threshold_results.loc[best_index, "threshold"])
