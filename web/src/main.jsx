import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import Papa from "papaparse";
import {
  AlertTriangle,
  ArrowDown,
  ArrowUp,
  BarChart3,
  BookOpen,
  CheckCircle2,
  CircleDollarSign,
  Filter,
  Gauge,
  GitBranch,
  Info,
  LineChart as LineChartIcon,
  ListChecks,
  Search,
  ShieldAlert,
  SlidersHorizontal,
  TableProperties,
} from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import "./styles.css";

const DATA_FILES = {
  metrics: "metrics_summary.json",
  thresholds: "threshold_results.csv",
  alerts: "fraud_alert_queue.csv",
};

const moneyFormatter = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  maximumFractionDigits: 0,
});

const numberFormatter = new Intl.NumberFormat("en-US");
const percentFormatter = new Intl.NumberFormat("en-US", {
  style: "percent",
  maximumFractionDigits: 1,
});

const tabItems = [
  { id: "decisioning", label: "Decisioning", icon: Gauge },
  { id: "alerts", label: "Alerts", icon: TableProperties },
  { id: "system", label: "System", icon: BookOpen },
];

function parseNumber(value) {
  if (value === null || value === undefined || value === "") return value;
  const parsed = Number(value);
  return Number.isNaN(parsed) ? value : parsed;
}

async function loadCsv(fileName) {
  const response = await fetch(`${import.meta.env.BASE_URL}${fileName}`);
  if (!response.ok) {
    throw new Error(`Could not load ${fileName}`);
  }

  const text = await response.text();
  return new Promise((resolve, reject) => {
    Papa.parse(text, {
      header: true,
      dynamicTyping: true,
      skipEmptyLines: true,
      complete: (result) => resolve(result.data),
      error: reject,
    });
  });
}

async function loadJson(fileName) {
  const response = await fetch(`${import.meta.env.BASE_URL}${fileName}`);
  if (!response.ok) {
    throw new Error(`Could not load ${fileName}`);
  }
  return response.json();
}

function deriveRiskLevel(score, recommendedThreshold) {
  if (score >= 0.75) return "High";
  if (score >= recommendedThreshold) return "Medium";
  return "Low";
}

function deriveRecommendedAction(riskLevel) {
  if (riskLevel === "High") return "Review immediately";
  if (riskLevel === "Medium") return "Review in alert queue";
  return "No immediate action";
}

function buildAnalystNote(alert) {
  const reasons = [alert.top_reason_1, alert.top_reason_2, alert.top_reason_3]
    .filter(Boolean)
    .join(", ");
  return `Risk score ${formatScore(alert.risk_score)}. Main risk drivers: ${reasons}. Recommended action: ${alert.recommended_action}.`;
}

function normalizeAlert(alert, recommendedThreshold) {
  const riskScore = parseNumber(alert.risk_score);
  const riskLevel = deriveRiskLevel(riskScore, recommendedThreshold);
  const recommendedAction = deriveRecommendedAction(riskLevel);
  const normalized = {
    ...alert,
    risk_score: riskScore,
    TransactionAmt: parseNumber(alert.TransactionAmt),
    predicted_fraud: Number(riskScore >= recommendedThreshold),
    risk_level: riskLevel,
    recommended_action: recommendedAction,
  };
  normalized.analyst_note = buildAnalystNote(normalized);
  return normalized;
}

function formatMoney(value) {
  return moneyFormatter.format(Number(value || 0));
}

function formatNumber(value) {
  return numberFormatter.format(Number(value || 0));
}

function formatPercent(value) {
  return percentFormatter.format(Number(value || 0));
}

function formatScore(value) {
  return Number(value || 0).toFixed(3);
}

function pickThresholdRow(thresholds, selectedThreshold) {
  return thresholds.reduce((best, row) => {
    if (!best) return row;
    return Math.abs(row.threshold - selectedThreshold) < Math.abs(best.threshold - selectedThreshold)
      ? row
      : best;
  }, null);
}

function App() {
  const [activeTab, setActiveTab] = useState(() => {
    const hashTab = window.location.hash.replace("#", "");
    return tabItems.some((item) => item.id === hashTab) ? hashTab : "decisioning";
  });
  const [dataState, setDataState] = useState({
    status: "loading",
    metrics: null,
    thresholds: [],
    alerts: [],
    error: null,
  });

  useEffect(() => {
    let isMounted = true;

    async function loadData() {
      try {
        const [metrics, thresholds, rawAlerts] = await Promise.all([
          loadJson(DATA_FILES.metrics),
          loadCsv(DATA_FILES.thresholds),
          loadCsv(DATA_FILES.alerts),
        ]);
        const recommendedThreshold = Number(metrics.recommended_threshold);
        const alerts = rawAlerts.map((alert) => normalizeAlert(alert, recommendedThreshold));

        if (isMounted) {
          setDataState({
            status: "ready",
            metrics,
            thresholds,
            alerts,
            error: null,
          });
        }
      } catch (error) {
        if (isMounted) {
          setDataState((current) => ({ ...current, status: "error", error }));
        }
      }
    }

    loadData();
    return () => {
      isMounted = false;
    };
  }, []);

  useEffect(() => {
    function syncTabFromHash() {
      const hashTab = window.location.hash.replace("#", "");
      if (tabItems.some((item) => item.id === hashTab)) {
        setActiveTab(hashTab);
      }
    }

    window.addEventListener("hashchange", syncTabFromHash);
    return () => window.removeEventListener("hashchange", syncTabFromHash);
  }, []);

  if (dataState.status === "loading") {
    return <LoadingState />;
  }

  if (dataState.status === "error") {
    return <MissingDataState error={dataState.error} />;
  }

  const ActiveIcon = tabItems.find((item) => item.id === activeTab)?.icon || Gauge;

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand-block">
          <div className="brand-mark">
            <ShieldAlert size={24} />
          </div>
          <div>
            <p className="eyebrow">RiskLens AI</p>
            <h1>Fraud Risk Intelligence</h1>
          </div>
        </div>

        <nav className="nav-list" aria-label="Primary">
          {tabItems.map((item) => {
            const Icon = item.icon;
            return (
              <button
                key={item.id}
                type="button"
                className={`nav-button ${activeTab === item.id ? "active" : ""}`}
                onClick={() => {
                  setActiveTab(item.id);
                  window.location.hash = item.id;
                }}
                title={item.label}
              >
                <Icon size={18} />
                <span>{item.label}</span>
              </button>
            );
          })}
        </nav>

        <div className="sidebar-summary">
          <p>Recommended threshold</p>
          <strong>{Number(dataState.metrics.recommended_threshold).toFixed(2)}</strong>
          <span>{formatNumber(dataState.metrics.recommended_alerts)} alerts at current policy</span>
        </div>
      </aside>

      <main className="main-panel">
        <header className="topbar">
          <div>
            <p className="eyebrow">IEEE-CIS validation workflow</p>
            <h2>
              <ActiveIcon size={22} />
              {tabItems.find((item) => item.id === activeTab)?.label}
            </h2>
          </div>
          <StatusPill label="Model outputs loaded" tone="good" />
        </header>

        {activeTab === "decisioning" && (
          <DecisioningView
            metrics={dataState.metrics}
            thresholds={dataState.thresholds}
            alerts={dataState.alerts}
          />
        )}
        {activeTab === "alerts" && (
          <AlertsView metrics={dataState.metrics} alerts={dataState.alerts} />
        )}
        {activeTab === "system" && (
          <SystemView metrics={dataState.metrics} thresholds={dataState.thresholds} alerts={dataState.alerts} />
        )}
      </main>
    </div>
  );
}

function LoadingState() {
  return (
    <div className="center-state">
      <ShieldAlert size={36} />
      <h1>Loading RiskLens AI</h1>
      <p>Reading model metrics, threshold simulation, and alert queue.</p>
    </div>
  );
}

function MissingDataState({ error }) {
  return (
    <div className="center-state">
      <AlertTriangle size={36} />
      <h1>Model outputs not found</h1>
      <p>Run the notebook to refresh `outputs/`, then restart the React app.</p>
      <code>{error?.message}</code>
    </div>
  );
}

function StatusPill({ label, tone = "neutral" }) {
  return (
    <span className={`status-pill ${tone}`}>
      <CheckCircle2 size={15} />
      {label}
    </span>
  );
}

function MetricCard({ icon: Icon, label, value, detail, trend }) {
  const TrendIcon = trend === "up" ? ArrowUp : trend === "down" ? ArrowDown : null;
  return (
    <article className="metric-card">
      <div className="metric-icon">
        <Icon size={20} />
      </div>
      <div>
        <p>{label}</p>
        <strong>{value}</strong>
        {detail && (
          <span>
            {TrendIcon && <TrendIcon size={13} />}
            {detail}
          </span>
        )}
      </div>
    </article>
  );
}

function DecisioningView({ metrics, thresholds, alerts }) {
  const [selectedThreshold, setSelectedThreshold] = useState(Number(metrics.recommended_threshold));
  const selectedRow = useMemo(
    () => pickThresholdRow(thresholds, Number(selectedThreshold)),
    [thresholds, selectedThreshold],
  );
  const topDrivers = useMemo(() => summarizeDrivers(alerts.filter((alert) => alert.risk_level === "High")), [alerts]);
  const thresholdDomain = thresholds.map((row) => row.threshold);

  return (
    <div className="view-stack">
      <section className="metric-grid">
        <MetricCard
          icon={Gauge}
          label="ROC-AUC"
          value={Number(metrics.roc_auc).toFixed(3)}
          detail={`Baseline ${Number(metrics.baseline_roc_auc).toFixed(3)}`}
          trend="up"
        />
        <MetricCard
          icon={LineChartIcon}
          label="PR-AUC"
          value={Number(metrics.pr_auc).toFixed(3)}
          detail={`Baseline ${Number(metrics.baseline_pr_auc).toFixed(3)}`}
          trend="up"
        />
        <MetricCard
          icon={SlidersHorizontal}
          label="Recommended Threshold"
          value={Number(metrics.recommended_threshold).toFixed(2)}
          detail={`${formatNumber(metrics.recommended_alerts)} alerts`}
        />
        <MetricCard
          icon={CircleDollarSign}
          label="Estimated Net Benefit"
          value={formatMoney(metrics.estimated_net_benefit)}
          detail={`${formatPercent(metrics.validation_fraud_rate)} validation fraud rate`}
        />
      </section>

      <section className="tool-band">
        <div className="section-heading">
          <div>
            <p className="eyebrow">Policy simulator</p>
            <h3>Threshold trade-off</h3>
          </div>
          <StatusPill label={`Selected ${Number(selectedThreshold).toFixed(2)}`} />
        </div>

        <div className="threshold-control">
          <input
            type="range"
            min={Math.min(...thresholdDomain)}
            max={Math.max(...thresholdDomain)}
            step="0.05"
            value={selectedThreshold}
            onChange={(event) => setSelectedThreshold(Number(event.target.value))}
            aria-label="Alert threshold"
          />
          <div className="threshold-values">
            <span>{Math.min(...thresholdDomain).toFixed(2)}</span>
            <strong>{Number(selectedThreshold).toFixed(2)}</strong>
            <span>{Math.max(...thresholdDomain).toFixed(2)}</span>
          </div>
        </div>

        {selectedRow && (
          <div className="sim-grid">
            <MetricCard icon={ShieldAlert} label="Precision" value={Number(selectedRow.precision).toFixed(3)} />
            <MetricCard icon={ListChecks} label="Recall" value={Number(selectedRow.recall).toFixed(3)} />
            <MetricCard
              icon={TableProperties}
              label="Flagged Transactions"
              value={formatNumber(selectedRow.flagged_transactions)}
            />
            <MetricCard
              icon={CircleDollarSign}
              label="Missed Fraud Loss"
              value={formatMoney(selectedRow.estimated_missed_fraud_loss)}
            />
          </div>
        )}
      </section>

      <section className="chart-grid">
        <ChartPanel eyebrow="Optimization" title="Net benefit by threshold">
          <ResponsiveContainer width="100%" height={260}>
            <LineChart data={thresholds} margin={{ top: 10, right: 18, left: 6, bottom: 8 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="threshold" />
              <YAxis tickFormatter={(value) => `$${Math.round(value / 1000)}k`} />
              <Tooltip formatter={(value) => formatMoney(value)} />
              <Line type="monotone" dataKey="estimated_net_benefit" stroke="#0f766e" strokeWidth={3} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </ChartPanel>

        <ChartPanel eyebrow="Operations" title="Alert workload by threshold">
          <ResponsiveContainer width="100%" height={260}>
            <LineChart data={thresholds} margin={{ top: 10, right: 18, left: 6, bottom: 8 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="threshold" />
              <YAxis tickFormatter={(value) => `${Math.round(value / 1000)}k`} />
              <Tooltip formatter={(value) => formatNumber(value)} />
              <Line type="monotone" dataKey="flagged_transactions" stroke="#b45309" strokeWidth={3} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </ChartPanel>
      </section>

      <section className="chart-grid">
        <ConfusionMatrix matrix={metrics.confusion_matrix} />
        <DriverPanel drivers={topDrivers} />
      </section>
    </div>
  );
}

function ChartPanel({ eyebrow, title, children }) {
  return (
    <article className="chart-panel">
      <div className="section-heading compact">
        <div>
          <p className="eyebrow">{eyebrow}</p>
          <h3>{title}</h3>
        </div>
      </div>
      {children}
    </article>
  );
}

function ConfusionMatrix({ matrix }) {
  const cells = [
    { label: "True negative", value: matrix[0][0], tone: "calm" },
    { label: "False positive", value: matrix[0][1], tone: "warn" },
    { label: "False negative", value: matrix[1][0], tone: "danger" },
    { label: "True positive", value: matrix[1][1], tone: "good" },
  ];

  return (
    <article className="chart-panel">
      <div className="section-heading compact">
        <div>
          <p className="eyebrow">Validation</p>
          <h3>Confusion matrix</h3>
        </div>
      </div>
      <div className="matrix-grid">
        {cells.map((cell) => (
          <div key={cell.label} className={`matrix-cell ${cell.tone}`}>
            <span>{cell.label}</span>
            <strong>{formatNumber(cell.value)}</strong>
          </div>
        ))}
      </div>
    </article>
  );
}

function DriverPanel({ drivers }) {
  return (
    <article className="chart-panel">
      <div className="section-heading compact">
        <div>
          <p className="eyebrow">Explainability</p>
          <h3>Top high-risk drivers</h3>
        </div>
      </div>
      <ResponsiveContainer width="100%" height={260}>
        <BarChart data={drivers} layout="vertical" margin={{ top: 8, right: 18, left: 12, bottom: 8 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis type="number" />
          <YAxis type="category" dataKey="driver" width={80} />
          <Tooltip formatter={(value) => formatNumber(value)} />
          <Bar dataKey="count" radius={[0, 6, 6, 0]}>
            {drivers.map((entry, index) => (
              <Cell key={entry.driver} fill={index % 2 === 0 ? "#0f766e" : "#2563eb"} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </article>
  );
}

function AlertsView({ metrics, alerts }) {
  const [riskFilter, setRiskFilter] = useState("High");
  const [productFilter, setProductFilter] = useState("All");
  const [minimumScore, setMinimumScore] = useState(0);
  const [query, setQuery] = useState("");
  const [selectedId, setSelectedId] = useState(null);

  const productOptions = useMemo(
    () => ["All", ...Array.from(new Set(alerts.map((alert) => alert.ProductCD).filter(Boolean))).sort()],
    [alerts],
  );

  const filteredAlerts = useMemo(() => {
    return alerts
      .filter((alert) => riskFilter === "All" || alert.risk_level === riskFilter)
      .filter((alert) => productFilter === "All" || alert.ProductCD === productFilter)
      .filter((alert) => alert.risk_score >= minimumScore)
      .filter((alert) => {
        if (!query.trim()) return true;
        const needle = query.trim().toLowerCase();
        return String(alert.TransactionID).includes(needle) || String(alert.top_reason_1).toLowerCase().includes(needle);
      })
      .sort((a, b) => b.risk_score - a.risk_score);
  }, [alerts, riskFilter, productFilter, minimumScore, query]);

  const selectedAlert = useMemo(() => {
    if (selectedId) {
      const found = alerts.find((alert) => String(alert.TransactionID) === String(selectedId));
      if (found) return found;
    }
    return filteredAlerts[0] || alerts[0];
  }, [alerts, filteredAlerts, selectedId]);

  const visibleAlerts = filteredAlerts.slice(0, 80);
  const highRiskCount = filteredAlerts.filter((alert) => alert.risk_level === "High").length;
  const totalAmount = filteredAlerts.reduce((sum, alert) => sum + Number(alert.TransactionAmt || 0), 0);
  const averageRisk = filteredAlerts.length
    ? filteredAlerts.reduce((sum, alert) => sum + Number(alert.risk_score || 0), 0) / filteredAlerts.length
    : 0;

  return (
    <div className="view-stack">
      <section className="metric-grid">
        <MetricCard icon={Filter} label="Filtered Alerts" value={formatNumber(filteredAlerts.length)} />
        <MetricCard icon={ShieldAlert} label="High Risk" value={formatNumber(highRiskCount)} />
        <MetricCard icon={Gauge} label="Average Risk Score" value={formatScore(averageRisk)} />
        <MetricCard icon={CircleDollarSign} label="Filtered Amount" value={formatMoney(totalAmount)} />
      </section>

      <section className="filter-band">
        <label>
          <span>Risk level</span>
          <select value={riskFilter} onChange={(event) => setRiskFilter(event.target.value)}>
            <option>All</option>
            <option>High</option>
            <option>Medium</option>
            <option>Low</option>
          </select>
        </label>
        <label>
          <span>ProductCD</span>
          <select value={productFilter} onChange={(event) => setProductFilter(event.target.value)}>
            {productOptions.map((product) => (
              <option key={product}>{product}</option>
            ))}
          </select>
        </label>
        <label>
          <span>Minimum score: {formatScore(minimumScore)}</span>
          <input
            type="range"
            min="0"
            max="1"
            step="0.01"
            value={minimumScore}
            onChange={(event) => setMinimumScore(Number(event.target.value))}
          />
        </label>
        <label className="search-field">
          <span>Search</span>
          <div>
            <Search size={16} />
            <input
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Transaction ID or driver"
            />
          </div>
        </label>
      </section>

      <section className="alert-layout">
        <article className="table-panel">
          <div className="section-heading compact">
            <div>
              <p className="eyebrow">Review queue</p>
              <h3>Prioritized transactions</h3>
            </div>
            <span className="muted-text">Showing {formatNumber(visibleAlerts.length)} rows</span>
          </div>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Transaction</th>
                  <th>Risk</th>
                  <th>Level</th>
                  <th>Amount</th>
                  <th>Product</th>
                  <th>Drivers</th>
                  <th>Actual</th>
                </tr>
              </thead>
              <tbody>
                {visibleAlerts.map((alert) => (
                  <tr
                    key={alert.TransactionID}
                    className={String(selectedAlert?.TransactionID) === String(alert.TransactionID) ? "selected" : ""}
                    onClick={() => setSelectedId(alert.TransactionID)}
                  >
                    <td>{alert.TransactionID}</td>
                    <td>{formatScore(alert.risk_score)}</td>
                    <td>
                      <RiskBadge level={alert.risk_level} />
                    </td>
                    <td>{formatMoney(alert.TransactionAmt)}</td>
                    <td>{alert.ProductCD}</td>
                    <td>{[alert.top_reason_1, alert.top_reason_2, alert.top_reason_3].filter(Boolean).join(", ")}</td>
                    <td>{Number(alert.actual_isFraud) === 1 ? "Fraud" : "Legit"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </article>

        <CasePanel alert={selectedAlert} metrics={metrics} />
      </section>
    </div>
  );
}

function RiskBadge({ level }) {
  return <span className={`risk-badge ${String(level).toLowerCase()}`}>{level}</span>;
}

function CasePanel({ alert, metrics }) {
  if (!alert) return null;

  const reasons = [alert.top_reason_1, alert.top_reason_2, alert.top_reason_3].filter(Boolean);

  return (
    <article className="case-panel">
      <div className="section-heading compact">
        <div>
          <p className="eyebrow">Case explanation</p>
          <h3>{alert.TransactionID}</h3>
        </div>
        <RiskBadge level={alert.risk_level} />
      </div>

      <div className="case-metrics">
        <div>
          <span>Risk score</span>
          <strong>{formatScore(alert.risk_score)}</strong>
        </div>
        <div>
          <span>Threshold</span>
          <strong>{Number(metrics.recommended_threshold).toFixed(2)}</strong>
        </div>
        <div>
          <span>Amount</span>
          <strong>{formatMoney(alert.TransactionAmt)}</strong>
        </div>
      </div>

      <div className="reason-list">
        {reasons.map((reason, index) => (
          <div key={`${reason}-${index}`}>
            <span>{index + 1}</span>
            <strong>{reason}</strong>
          </div>
        ))}
      </div>

      <div className="note-box">
        <Info size={17} />
        <p>{alert.analyst_note}</p>
      </div>
    </article>
  );
}

function SystemView({ metrics, thresholds, alerts }) {
  const bestThreshold = pickThresholdRow(thresholds, Number(metrics.recommended_threshold));
  const highCount = alerts.filter((alert) => alert.risk_level === "High").length;
  const mediumCount = alerts.filter((alert) => alert.risk_level === "Medium").length;
  const lowCount = alerts.filter((alert) => alert.risk_level === "Low").length;

  return (
    <div className="view-stack">
      <section className="system-band">
        <div>
          <p className="eyebrow">Model operating model</p>
          <h3>From transaction history to review-ready alerts</h3>
        </div>
        <p>
          RiskLens AI ranks validation transactions with a LightGBM fraud model, chooses a threshold from business
          trade-offs, and turns SHAP drivers into analyst-readable case notes.
        </p>
      </section>

      <section className="pipeline-grid">
        <PipelineStep
          icon={GitBranch}
          title="1. Data"
          body="The MVP uses IEEE-CIS train_transaction.csv. Identity data can exist locally, but the current model does not train on it."
        />
        <PipelineStep
          icon={Gauge}
          title="2. Scoring"
          body="Feature engineering creates time, amount, missingness, card, address, and email-domain signals before LightGBM assigns risk scores."
        />
        <PipelineStep
          icon={SlidersHorizontal}
          title="3. Threshold"
          body={`The selected threshold is ${Number(metrics.recommended_threshold).toFixed(2)}, optimized against captured fraud amount and investigation workload.`}
        />
        <PipelineStep
          icon={ListChecks}
          title="4. Review"
          body="The alert queue prioritizes transactions, displays top drivers, and gives a recommended analyst action."
        />
      </section>

      <section className="metric-grid">
        <MetricCard icon={TableProperties} label="Validation Transactions" value={formatNumber(metrics.total_validation_transactions)} />
        <MetricCard icon={AlertTriangle} label="Validation Fraud Rate" value={formatPercent(metrics.validation_fraud_rate)} />
        <MetricCard icon={ShieldAlert} label="High Risk Cases" value={formatNumber(highCount)} />
        <MetricCard icon={ListChecks} label="Medium Queue Cases" value={formatNumber(mediumCount)} detail={`${formatNumber(lowCount)} low risk`} />
      </section>

      <section className="chart-grid">
        <article className="chart-panel">
          <div className="section-heading compact">
            <div>
              <p className="eyebrow">Policy snapshot</p>
              <h3>Recommended operating point</h3>
            </div>
          </div>
          <dl className="snapshot-list">
            <div>
              <dt>Flagged transactions</dt>
              <dd>{formatNumber(bestThreshold?.flagged_transactions)}</dd>
            </div>
            <div>
              <dt>Captured fraud amount</dt>
              <dd>{formatMoney(bestThreshold?.estimated_fraud_amount_captured)}</dd>
            </div>
            <div>
              <dt>Investigation cost</dt>
              <dd>{formatMoney(bestThreshold?.investigation_cost)}</dd>
            </div>
            <div>
              <dt>Estimated net benefit</dt>
              <dd>{formatMoney(bestThreshold?.estimated_net_benefit)}</dd>
            </div>
          </dl>
        </article>

        <article className="chart-panel">
          <div className="section-heading compact">
            <div>
              <p className="eyebrow">Known limits</p>
              <h3>Current MVP boundaries</h3>
            </div>
          </div>
          <ul className="limit-list">
            <li>Static validation outputs are used by the React app.</li>
            <li>Live transaction inference still needs an API layer.</li>
            <li>Identity features and drift monitoring are not active yet.</li>
            <li>Cost assumptions are simplified for MVP decisioning.</li>
          </ul>
        </article>
      </section>
    </div>
  );
}

function PipelineStep({ icon: Icon, title, body }) {
  return (
    <article className="pipeline-step">
      <div>
        <Icon size={20} />
      </div>
      <h4>{title}</h4>
      <p>{body}</p>
    </article>
  );
}

function summarizeDrivers(alerts) {
  const counts = new Map();
  alerts.forEach((alert) => {
    [alert.top_reason_1, alert.top_reason_2, alert.top_reason_3].filter(Boolean).forEach((reason) => {
      counts.set(reason, (counts.get(reason) || 0) + 1);
    });
  });

  return Array.from(counts.entries())
    .map(([driver, count]) => ({ driver, count }))
    .sort((a, b) => b.count - a.count)
    .slice(0, 10);
}

createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
