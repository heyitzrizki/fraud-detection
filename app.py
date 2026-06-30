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


def get_threshold_row(threshold_results: pd.DataFrame, threshold: float) -> pd.Series:
    return threshold_results.iloc[
        (threshold_results["threshold"] - threshold).abs().argsort()[:1]
    ].iloc[0]


def render_risk_business_impact(
    metrics: dict,
    threshold_results: pd.DataFrame,
    alert_queue: pd.DataFrame,
    pattern_data: pd.DataFrame | None,
) -> None:
    recommended_threshold = float(metrics["recommended_threshold"])
    selected_threshold_row = get_threshold_row(threshold_results, recommended_threshold)

    st.header("Risk & Business Impact")
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

    st.subheader("Threshold Simulator")
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

    st.subheader("Model Performance")
    performance_columns = st.columns(5)
    performance_columns[0].metric("Precision", f"{metrics['precision']:.3f}")
    performance_columns[1].metric("Recall", f"{metrics['recall']:.3f}")
    performance_columns[2].metric("F1-Score", f"{metrics['f1_score']:.3f}")
    performance_columns[3].metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
    performance_columns[4].metric("PR-AUC", f"{metrics['pr_auc']:.3f}")

    left, right = st.columns(2)
    confusion_matrix = metrics["confusion_matrix"]
    confusion_fig = go.Figure(
        data=go.Heatmap(
            z=confusion_matrix,
            x=["Predicted Not Fraud", "Predicted Fraud"],
            y=["Actual Not Fraud", "Actual Fraud"],
            colorscale="Blues",
            text=confusion_matrix,
            texttemplate="%{text}",
        )
    )
    confusion_fig.update_layout(title="Confusion Matrix at Recommended Threshold")
    left.plotly_chart(confusion_fig, use_container_width=True)

    high_risk_alerts = alert_queue.loc[alert_queue["risk_level"] == "High"].copy()
    reason_columns = ["top_reason_1", "top_reason_2", "top_reason_3"]
    reason_summary = (
        high_risk_alerts[reason_columns]
        .melt(value_name="risk_driver")["risk_driver"]
        .value_counts()
        .head(10)
        .reset_index()
    )
    reason_summary.columns = ["risk_driver", "count"]
    right.plotly_chart(
        px.bar(
            reason_summary,
            x="count",
            y="risk_driver",
            orientation="h",
            title="Top XAI Drivers in High-Risk Alerts",
        ),
        use_container_width=True,
    )

    st.subheader("Fraud Pattern Analysis")
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
        left.plotly_chart(
            px.bar(product_summary, x="ProductCD", y="fraud_rate", title="Fraud Rate by ProductCD"),
            use_container_width=True,
        )

        chart_data = pattern_data.sample(min(len(pattern_data), 100000), random_state=42)
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


def filter_alert_queue(alert_queue: pd.DataFrame) -> pd.DataFrame:
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

    return filtered_alerts


def render_alert_queue(alert_queue: pd.DataFrame) -> None:
    st.header("Alert Queue")
    filtered_alerts = filter_alert_queue(alert_queue)

    st.subheader("Alert Workload")
    workload_columns = st.columns(4)
    workload_columns[0].metric("Filtered Alerts", f"{len(filtered_alerts):,}")
    workload_columns[1].metric("High Risk", f"{(filtered_alerts['risk_level'] == 'High').sum():,}")
    workload_columns[2].metric("Average Risk Score", f"{filtered_alerts['risk_score'].mean():.3f}" if len(filtered_alerts) else "0.000")
    workload_columns[3].metric("Total Amount", format_currency(filtered_alerts["TransactionAmt"].sum()))

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

    st.subheader("Case Explanation")
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

    st.title("RiskLens AI")
    st.caption("Explainable fraud and risk intelligence MVP")

    page = st.sidebar.radio(
        "Page",
        ["Risk & Business Impact", "Alert Queue"],
    )

    if page == "Risk & Business Impact":
        render_risk_business_impact(metrics, threshold_results, alert_queue, pattern_data)
    else:
        render_alert_queue(alert_queue)


if __name__ == "__main__":
    main()
