from pathlib import Path
import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


OUTPUT_DIR = Path("outputs")
RAW_TRANSACTION_PATH = Path("data/raw/train_transaction.csv")

METRICS_PATH = OUTPUT_DIR / "metrics_summary.json"
THRESHOLD_PATH = OUTPUT_DIR / "threshold_results.csv"
ALERT_QUEUE_PATH = OUTPUT_DIR / "fraud_alert_queue.csv"


def load_metrics() -> dict:
    with METRICS_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data
def load_outputs():
    metrics = load_metrics()
    threshold_results = pd.read_csv(THRESHOLD_PATH)
    alert_queue = pd.read_csv(ALERT_QUEUE_PATH)
    return metrics, threshold_results, alert_queue


@st.cache_data
def load_pattern_data():
    if not RAW_TRANSACTION_PATH.exists():
        return None

    columns = ["isFraud", "ProductCD", "TransactionAmt", "TransactionDT"]
    return pd.read_csv(RAW_TRANSACTION_PATH, usecols=columns)


def format_currency(value: float) -> str:
    return f"${value:,.0f}"


def show_missing_outputs_message():
    st.title("RiskLens AI")
    st.warning(
        "Model outputs were not found. Run notebooks/01_risklens_ai_mvp.ipynb first, "
        "then restart this dashboard."
    )


def main():
    st.set_page_config(
        page_title="RiskLens AI",
        layout="wide",
    )

    required_files = [METRICS_PATH, THRESHOLD_PATH, ALERT_QUEUE_PATH]
    if any(not path.exists() for path in required_files):
        show_missing_outputs_message()
        st.stop()

    metrics, threshold_results, alert_queue = load_outputs()
    pattern_data = load_pattern_data()

    recommended_threshold = float(metrics["recommended_threshold"])
    selected_threshold_row = threshold_results.iloc[
        (threshold_results["threshold"] - recommended_threshold).abs().argsort()[:1]
    ].iloc[0]

    st.title("RiskLens AI")
    st.caption("Explainable fraud and risk intelligence MVP")

    st.header("Executive Overview")
    overview_columns = st.columns(4)
    overview_columns[0].metric("Validation Transactions", f"{metrics['total_validation_transactions']:,}")
    overview_columns[1].metric("Validation Fraud Rate", f"{metrics['validation_fraud_rate']:.2%}")
    overview_columns[2].metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
    overview_columns[3].metric("PR-AUC", f"{metrics['pr_auc']:.3f}")

    overview_columns = st.columns(4)
    overview_columns[0].metric("Recommended Threshold", f"{recommended_threshold:.2f}")
    overview_columns[1].metric("Alerts", f"{int(selected_threshold_row['flagged_transactions']):,}")
    overview_columns[2].metric(
        "Fraud Amount Captured",
        format_currency(selected_threshold_row["estimated_fraud_amount_captured"]),
    )
    overview_columns[3].metric(
        "Estimated Net Benefit",
        format_currency(selected_threshold_row["estimated_net_benefit"]),
    )

    st.header("Fraud Pattern Analysis")
    if pattern_data is None:
        st.info("Raw transaction data was not found, so fraud pattern charts are unavailable.")
    else:
        product_summary = (
            pattern_data.groupby("ProductCD")
            .agg(
                transactions=("isFraud", "size"),
                fraud_count=("isFraud", "sum"),
                fraud_rate=("isFraud", "mean"),
            )
            .reset_index()
            .sort_values("fraud_rate", ascending=False)
        )

        left, right = st.columns(2)
        left.dataframe(product_summary, use_container_width=True)
        right.plotly_chart(
            px.bar(product_summary, x="ProductCD", y="fraud_rate", title="Fraud Rate by ProductCD"),
            use_container_width=True,
        )

        chart_data = pattern_data.sample(min(len(pattern_data), 100000), random_state=42)
        left, right = st.columns(2)
        left.plotly_chart(
            px.histogram(
                chart_data,
                x="TransactionAmt",
                nbins=80,
                title="Transaction Amount Distribution",
            ),
            use_container_width=True,
        )
        right.plotly_chart(
            px.box(
                chart_data,
                x="isFraud",
                y="TransactionAmt",
                points=False,
                title="Transaction Amount by Fraud Label",
            ),
            use_container_width=True,
        )

    st.header("Model Performance")
    performance_columns = st.columns(5)
    performance_columns[0].metric("Precision", f"{metrics['precision']:.3f}")
    performance_columns[1].metric("Recall", f"{metrics['recall']:.3f}")
    performance_columns[2].metric("F1-Score", f"{metrics['f1_score']:.3f}")
    performance_columns[3].metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
    performance_columns[4].metric("PR-AUC", f"{metrics['pr_auc']:.3f}")

    confusion_matrix = metrics["confusion_matrix"]
    fig = go.Figure(
        data=go.Heatmap(
            z=confusion_matrix,
            x=["Predicted Not Fraud", "Predicted Fraud"],
            y=["Actual Not Fraud", "Actual Fraud"],
            colorscale="Blues",
            text=confusion_matrix,
            texttemplate="%{text}",
        )
    )
    fig.update_layout(title="Confusion Matrix at Recommended Threshold")
    st.plotly_chart(fig, use_container_width=True)

    st.header("Threshold Simulator")
    threshold_values = threshold_results["threshold"].round(2).tolist()
    default_index = min(
        range(len(threshold_values)),
        key=lambda index: abs(threshold_values[index] - recommended_threshold),
    )
    selected_threshold = st.select_slider(
        "Alert threshold",
        options=threshold_values,
        value=threshold_values[default_index],
    )
    threshold_row = threshold_results.loc[
        threshold_results["threshold"].round(2) == selected_threshold
    ].iloc[0]

    simulator_columns = st.columns(5)
    simulator_columns[0].metric("Precision", f"{threshold_row['precision']:.3f}")
    simulator_columns[1].metric("Recall", f"{threshold_row['recall']:.3f}")
    simulator_columns[2].metric("Flagged Transactions", f"{int(threshold_row['flagged_transactions']):,}")
    simulator_columns[3].metric("Missed Fraud Loss", format_currency(threshold_row["estimated_missed_fraud_loss"]))
    simulator_columns[4].metric("Net Benefit", format_currency(threshold_row["estimated_net_benefit"]))

    left, right = st.columns(2)
    left.plotly_chart(
        px.line(
            threshold_results,
            x="threshold",
            y="estimated_net_benefit",
            markers=True,
            title="Threshold vs Net Benefit",
        ),
        use_container_width=True,
    )
    right.plotly_chart(
        px.line(
            threshold_results,
            x="threshold",
            y="flagged_transactions",
            markers=True,
            title="Threshold vs Alert Workload",
        ),
        use_container_width=True,
    )

    st.header("Fraud Alert Queue")
    filter_columns = st.columns(3)
    risk_level_options = sorted(alert_queue["risk_level"].dropna().unique())
    default_risk_levels = ["High"] if "High" in set(risk_level_options) else risk_level_options
    risk_levels = filter_columns[0].multiselect(
        "Risk level",
        options=risk_level_options,
        default=default_risk_levels,
    )
    product_options = sorted(alert_queue["ProductCD"].dropna().unique())
    selected_products = filter_columns[1].multiselect(
        "ProductCD",
        options=product_options,
        default=product_options,
    )
    minimum_risk_score = filter_columns[2].slider(
        "Minimum risk score",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.01,
    )

    filtered_alerts = alert_queue[
        alert_queue["risk_level"].isin(risk_levels)
        & alert_queue["ProductCD"].isin(selected_products)
        & (alert_queue["risk_score"] >= minimum_risk_score)
    ].copy()

    table_columns = [
        "TransactionID",
        "risk_score",
        "risk_level",
        "TransactionAmt",
        "ProductCD",
        "top_reason_1",
        "top_reason_2",
        "top_reason_3",
        "recommended_action",
        "analyst_note",
    ]
    st.dataframe(filtered_alerts[table_columns], use_container_width=True, height=420)

    st.header("Case Explanation")
    case_source = filtered_alerts if len(filtered_alerts) else alert_queue.head(100)
    selected_transaction_id = st.selectbox(
        "TransactionID",
        options=case_source["TransactionID"].tolist(),
    )
    selected_case = alert_queue.loc[
        alert_queue["TransactionID"] == selected_transaction_id
    ].iloc[0]

    case_columns = st.columns(4)
    case_columns[0].metric("Risk Score", f"{selected_case['risk_score']:.3f}")
    case_columns[1].metric("Risk Level", selected_case["risk_level"])
    case_columns[2].metric("Transaction Amount", format_currency(selected_case["TransactionAmt"]))
    case_columns[3].metric("Action", selected_case["recommended_action"])

    st.write("Top reasons")
    st.table(
        pd.DataFrame(
            {
                "rank": [1, 2, 3],
                "reason": [
                    selected_case["top_reason_1"],
                    selected_case["top_reason_2"],
                    selected_case["top_reason_3"],
                ],
            }
        )
    )
    st.info(selected_case["analyst_note"])


if __name__ == "__main__":
    main()
