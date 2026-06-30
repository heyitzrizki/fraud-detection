import pandas as pd


def assign_risk_level(risk_score: float, recommended_threshold: float) -> str:
    if risk_score >= recommended_threshold:
        return "High"
    if risk_score >= 0.50:
        return "Medium"
    return "Low"


def assign_recommended_action(risk_level: str) -> str:
    if risk_level == "High":
        return "Review immediately"
    if risk_level == "Medium":
        return "Monitor or review if workload allows"
    return "No immediate action"


def generate_analyst_note(
    risk_score: float,
    top_reasons: list[str],
    recommended_action: str,
) -> str:
    reasons_text = ", ".join(top_reasons)
    return (
        f"This transaction received a fraud risk score of {risk_score:.2f}. "
        f"The main risk drivers are {reasons_text}. "
        f"Recommended action: {recommended_action}."
    )


def build_alert_queue(
    validation_data: pd.DataFrame,
    risk_scores,
    recommended_threshold: float,
    reason_lookup: dict[int, list[str]],
    default_reasons: list[str] | None = None,
) -> pd.DataFrame:
    queue = validation_data.copy()
    queue["risk_score"] = risk_scores
    queue["risk_level"] = queue["risk_score"].apply(
        lambda score: assign_risk_level(score, recommended_threshold)
    )
    queue["predicted_fraud"] = (queue["risk_score"] >= recommended_threshold).astype(int)
    queue["recommended_action"] = queue["risk_level"].apply(assign_recommended_action)

    default_reasons = default_reasons or ["Model risk pattern"]
    top_reasons = queue["TransactionID"].apply(
        lambda transaction_id: reason_lookup.get(int(transaction_id), default_reasons)
    )
    queue["top_reason_1"] = top_reasons.apply(lambda reasons: reasons[0] if len(reasons) > 0 else "")
    queue["top_reason_2"] = top_reasons.apply(lambda reasons: reasons[1] if len(reasons) > 1 else "")
    queue["top_reason_3"] = top_reasons.apply(lambda reasons: reasons[2] if len(reasons) > 2 else "")
    queue["analyst_note"] = queue.apply(
        lambda row: generate_analyst_note(
            row["risk_score"],
            [row["top_reason_1"], row["top_reason_2"], row["top_reason_3"]],
            row["recommended_action"],
        ),
        axis=1,
    )

    selected_columns = [
        "TransactionID",
        "risk_score",
        "risk_level",
        "TransactionAmt",
        "ProductCD",
        "predicted_fraud",
        "isFraud",
        "top_reason_1",
        "top_reason_2",
        "top_reason_3",
        "recommended_action",
        "analyst_note",
    ]
    available_columns = [column for column in selected_columns if column in queue.columns]

    return (
        queue.loc[:, available_columns]
        .rename(columns={"isFraud": "actual_isFraud"})
        .sort_values("risk_score", ascending=False)
        .reset_index(drop=True)
    )
