
import re
import base64
import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# -----------------------------
# Configuration
# -----------------------------
st.set_page_config(
    page_title="UPI Sentinel | Fraud & Merchant Intelligence",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

TX_PREFIX = "transactions.b64."
MERCHANT_PREFIX = "merchants.b64."
KYC_PREFIX = "transactions_kyc_risk.b64."
CB_PREFIX = "chargebacks.b64."


# -----------------------------
# Styling
# -----------------------------
st.markdown(
    """
<style>
    .stApp { background: #f6f8fb; }
    [data-testid="stSidebar"] { background: #111827; }
    [data-testid="stSidebar"] * { color: #e5e7eb !important; }
    .hero { padding: 22px 26px; border-radius: 18px; background: linear-gradient(135deg, #111827 0%, #1f2937 55%, #334155 100%); color: white; margin-bottom: 18px; }
    .hero h1 { margin: 0; font-size: 2rem; letter-spacing: -0.03em; }
    .hero p { margin: 7px 0 0 0; color: #cbd5e1; font-size: 0.98rem; }
    .section-title { font-size: 1.25rem; font-weight: 700; color: #111827; margin: 14px 0 8px 0; }
    .small-note { color: #64748b; font-size: 0.82rem; }
    .risk-high { color: #b91c1c; font-weight: 700; }
    .risk-medium { color: #a16207; font-weight: 700; }
    .risk-low { color: #15803d; font-weight: 700; }
    .insight { padding: 14px 16px; border-left: 4px solid #334155; background: white; border-radius: 10px; margin: 8px 0; }
    div[data-testid="stMetric"] { background: white; border: 1px solid #e5e7eb; padding: 12px 14px; border-radius: 12px; }
    .stTabs [data-baseweb="tab-list"] { gap: 6px; }
    .stTabs [data-baseweb="tab"] { border-radius: 9px; padding: 8px 14px; }
</style>
""",
    unsafe_allow_html=True,
)


# -----------------------------
# Helpers
# -----------------------------
def norm_id(value):
    if pd.isna(value):
        return ""
    return re.sub(r"[^A-Z0-9]", "", str(value).upper().strip())


def clean_category(value):
    if pd.isna(value):
        return "Unknown"
    s = str(value).strip().lower().replace("_", " ").replace("-", " ")
    s = re.sub(r"\s+", " ", s)
    if any(x in s for x in ["grocery", "kirana"]): return "Grocery"
    if any(x in s for x in ["department", "dept store"]): return "Department Store"
    if any(x in s for x in ["cloth", "apparel", "garment", "fashion"]): return "Apparel"
    if any(x in s for x in ["food", "restaurant", "eating"]): return "Food Services"
    if any(x in s for x in ["hotel", "hospitality"]): return "Hotel & Lodging"
    if any(x in s for x in ["telecom", "phone", "mobile"]): return "Telecom"
    if any(x in s for x in ["transport", "travel", "bus", "taxi"]): return "Transport"
    if any(x in s for x in ["pharmacy", "medical", "chemist"]): return "Healthcare"
    if any(x in s for x in ["book", "stationery"]): return "Books & Stationery"
    return str(value).strip().title()


MCC_CATEGORY = {
    "4131": "Transport", "4814": "Telecom", "5311": "Department Store",
    "5411": "Grocery", "5699": "Apparel", "5812": "Food Services",
    "5912": "Healthcare", "5942": "Books & Stationery", "5999": "Other Retail",
    "7011": "Hotel & Lodging",
}


def money(v):
    if pd.isna(v): return "₹0"
    v = float(v); av = abs(v)
    if av >= 1e7: return f"₹{v/1e7:.2f} Cr"
    if av >= 1e5: return f"₹{v/1e5:.2f} L"
    return f"₹{v:,.0f}"


def pct(v): return f"{v:.2f}%"


@st.cache_data(show_spinner=False)
def load_data():
    def read_bundled_csv(prefix):
        parts = sorted(DATA.glob(prefix + "*"))
        if not parts:
            raise FileNotFoundError(f"Missing bundled dataset: {prefix}*")
        payload = "".join(p.read_text().strip() for p in parts)
        raw = base64.b64decode(payload)
        return pd.read_csv(io.BytesIO(raw), compression="gzip")

    tx = read_bundled_csv(TX_PREFIX)
    mer = read_bundled_csv(MERCHANT_PREFIX)
    kyc = read_bundled_csv(KYC_PREFIX)
    cb = read_bundled_csv(CB_PREFIX)

    tx["date"] = pd.to_datetime(tx["timestamp_clean"], errors="coerce")
    tx["amount_clean"] = pd.to_numeric(tx["amount_clean"], errors="coerce")
    tx["txn_key"] = tx["txn_id"].map(norm_id)
    tx["merchant_key"] = tx["merchant_id"].map(norm_id)
    tx["user_key"] = tx["user_id"].map(norm_id)
    tx["mcc_key"] = pd.to_numeric(tx["mcc_clean"], errors="coerce").round().astype("Int64").astype(str).replace("<NA>", "")
    tx["category"] = tx["mcc_key"].map(MCC_CATEGORY).fillna("Unknown")

    mer["merchant_key"] = mer["merchant_id"].map(norm_id)
    mer["category_clean"] = mer["merchant_category"].map(clean_category)
    merchant_lookup = mer.drop_duplicates("merchant_key")[["merchant_key", "merchant_name", "merchant_category", "category_clean", "business_type", "city", "state", "merchant_status", "declared_avg_ticket_size"]]
    tx = tx.merge(merchant_lookup, on="merchant_key", how="left")

    cb["txn_key"] = cb["txn_id_clean"].map(norm_id)
    cb["disputed_amount_clean"] = pd.to_numeric(cb["disputed_amount_clean"], errors="coerce")
    tx["has_chargeback"] = tx["txn_key"].isin(set(cb["txn_key"].dropna()))
    cb_summary = cb.groupby("txn_key", dropna=False).agg(chargeback_count=("complaint_id", "count"), total_disputed_amount=("disputed_amount_clean", "sum"), max_severity=("severity", lambda s: ", ".join(sorted(set(s.dropna().astype(str)))))).reset_index()
    tx = tx.merge(cb_summary, on="txn_key", how="left")
    tx["chargeback_count"] = tx["chargeback_count"].fillna(0).astype(int)
    tx["total_disputed_amount"] = tx["total_disputed_amount"].fillna(0)

    kyc["txn_key"] = kyc["txn_id"].map(norm_id)
    for c in ["high_risk_record", "identity_conflict_flag"]:
        if c in kyc.columns: kyc[c] = kyc[c].fillna(False).astype(bool)
    risk_cols = ["txn_key", "high_risk_record", "identity_conflict_flag", "risk_segments", "kyc_statuses", "kyc_record_count", "pan_count", "aadhaar_count"]
    risk = kyc[[c for c in risk_cols if c in kyc.columns]].drop_duplicates("txn_key")
    tx = tx.merge(risk, on="txn_key", how="left")
    tx["high_risk_record"] = tx.get("high_risk_record", False).fillna(False).astype(bool)
    tx["identity_conflict_flag"] = tx.get("identity_conflict_flag", False).fillna(False).astype(bool)
    tx["risk_segments"] = tx.get("risk_segments", "").fillna("Unknown")
    tx["category"] = tx["category"].replace("", np.nan).fillna(tx["category_clean"]).fillna("Unknown")
    return tx, mer, cb


def merchant_risk_table(df):
    g = df.groupby(["merchant_key", "merchant_name", "category"], dropna=False).agg(transactions=("txn_id", "count"), successful=("status_clean", lambda s: (s == "SUCCESS").sum()), failed=("status_clean", lambda s: (s == "FAILED").sum()), transaction_value=("amount_clean", "sum"), chargebacks=("has_chargeback", "sum"), identity_conflicts=("identity_conflict_flag", "sum"), high_risk_kyc=("high_risk_record", "sum")).reset_index()
    g["chargeback_rate"] = np.where(g["transactions"] > 0, g["chargebacks"] / g["transactions"] * 100, 0)
    g["failure_rate"] = np.where(g["transactions"] > 0, g["failed"] / g["transactions"] * 100, 0)
    g["identity_rate"] = np.where(g["transactions"] > 0, g["identity_conflicts"] / g["transactions"] * 100, 0)
    g["high_risk_rate"] = np.where(g["transactions"] > 0, g["high_risk_kyc"] / g["transactions"] * 100, 0)
    def minmax(s):
        if s.max() == s.min(): return pd.Series(0.0, index=s.index)
        return (s - s.min()) / (s.max() - s.min()) * 100
    g["risk_score"] = (0.40 * minmax(g["chargeback_rate"]) + 0.25 * minmax(g["high_risk_rate"]) + 0.20 * minmax(g["identity_rate"]) + 0.15 * minmax(g["failure_rate"])).round(1)
    g["risk_level"] = pd.cut(g["risk_score"], bins=[-0.1, 30, 60, 80, 100.1], labels=["Low", "Medium", "High", "Critical"])
    return g.sort_values(["risk_score", "transactions"], ascending=[False, False])


try:
    tx, merchants, chargebacks = load_data()
except Exception as e:
    st.error(f"Dashboard could not load the data: {e}")
    st.stop()

st.sidebar.markdown("## 💳 UPI Sentinel")
st.sidebar.caption("Fraud & Merchant Intelligence")
min_date = tx["date"].min().date(); max_date = tx["date"].max().date()
date_range = st.sidebar.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date = end_date = date_range
categories = st.sidebar.multiselect("Merchant category", sorted(tx["category"].dropna().unique()), default=sorted(tx["category"].dropna().unique()))
statuses = st.sidebar.multiselect("Transaction status", sorted(tx["status_clean"].dropna().unique()), default=sorted(tx["status_clean"].dropna().unique()))
filtered = tx[(tx["date"].dt.date >= start_date) & (tx["date"].dt.date <= end_date) & tx["category"].isin(categories) & tx["status_clean"].isin(statuses)].copy()

st.markdown('<div class="hero"><h1>UPI Sentinel</h1><p>Fraud & Merchant Intelligence Platform • TransOrg AgentIQ Datathon</p></div>', unsafe_allow_html=True)
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Executive Overview", "Fraud Intelligence", "Merchant Risk", "Chargebacks & KYC", "Ask UPI Sentinel"])

with tab1:
    st.markdown('<div class="section-title">Executive Overview</div>', unsafe_allow_html=True)
    total = len(filtered); success = int((filtered["status_clean"] == "SUCCESS").sum()); failed = int((filtered["status_clean"] == "FAILED").sum()); pending = int((filtered["status_clean"] == "PENDING").sum())
    value = filtered.loc[filtered["status_clean"] == "SUCCESS", "amount_clean"].sum(); cb_count = int(filtered["has_chargeback"].sum()); identity = int(filtered["identity_conflict_flag"].sum()); highrisk = int(filtered["high_risk_record"].sum())
    cols = st.columns(8)
    for c, label, val in zip(cols, ["Transactions", "Successful", "Success rate", "Successful value", "Chargeback-linked", "Failed", "Identity conflicts", "High-risk KYC"], [f"{total:,}", f"{success:,}", pct(success/total*100 if total else 0), money(value), f"{cb_count:,}", f"{failed:,}", f"{identity:,}", f"{highrisk:,}"]): c.metric(label, val)
    st.markdown("**Transaction value trend**")
    trend = filtered[filtered["status_clean"] == "SUCCESS"].assign(month=filtered.loc[filtered["status_clean"] == "SUCCESS", "date"].dt.to_period("M").astype(str)).groupby("month", as_index=False).agg(transaction_value=("amount_clean", "sum"))
    fig = px.area(trend, x="month", y="transaction_value", labels={"month": "Month", "transaction_value": "Successful transaction value"})
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=15, b=10))
    st.plotly_chart(fig, use_container_width=True)
    left, right = st.columns(2)
    with left:
        status_df = filtered["status_clean"].value_counts().rename_axis("status").reset_index(name="count")
        fig = px.pie(status_df, names="status", values="count", hole=.5, title="Transaction status mix")
        fig.update_layout(height=340)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        cat = filtered.groupby("category", as_index=False).agg(transactions=("txn_id", "count"), value=("amount_clean", "sum")).sort_values("transactions", ascending=True)
        fig = px.bar(cat, x="transactions", y="category", orientation="h", title="Transaction volume by category")
        fig.update_layout(height=340)
        st.plotly_chart(fig, use_container_width=True)
    st.markdown('<div class="insight"><b>Interpretation:</b> Use the filters to isolate a period, category, or status before moving into merchant-level investigation.</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="section-title">Fraud & Transaction Intelligence</div>', unsafe_allow_html=True)
    cat = filtered.groupby("category", as_index=False).agg(transactions=("txn_id", "count"), chargebacks=("has_chargeback", "sum"), failed=("status_clean", lambda s: (s == "FAILED").sum()))
    cat["chargeback_rate"] = np.where(cat["transactions"] > 0, cat["chargebacks"] / cat["transactions"] * 100, 0)
    left, right = st.columns(2)
    with left:
        fig = px.bar(cat.sort_values("transactions"), x="transactions", y="category", orientation="h", title="Transaction volume")
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        fig = px.bar(cat.sort_values("chargeback_rate"), x="chargeback_rate", y="category", orientation="h", text="chargeback_rate", title="Chargeback rate by category")
        fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
        fig.update_layout(height=380, margin=dict(l=10, r=35, t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)
    daily = filtered.assign(day=filtered["date"].dt.date).groupby("day", as_index=False).agg(failed=("status_clean", lambda s: (s == "FAILED").sum()), chargebacks=("has_chargeback", "sum"))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily["day"], y=daily["failed"], mode="lines", name="Failed"))
    fig.add_trace(go.Scatter(x=daily["day"], y=daily["chargebacks"], mode="lines", name="Chargeback-linked"))
    fig.update_layout(title="Daily failure and chargeback activity", height=360, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)
    r = merchant_risk_table(filtered).head(10).sort_values("risk_score", ascending=True)
    fig = px.bar(r, x="risk_score", y="merchant_name", orientation="h", text="risk_score", title="Top merchant risk signals")
    fig.update_traces(textposition="outside")
    fig.update_layout(height=390, margin=dict(l=10, r=35, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.markdown('<div class="section-title">Merchant Risk & Investigation Queue</div>', unsafe_allow_html=True)
    min_tx = st.number_input("Minimum merchant transactions", min_value=1, max_value=500, value=10, help="Reduces noise from merchants with very small samples.")
    risk_df = merchant_risk_table(filtered)
    risk_df = risk_df[risk_df["transactions"] >= min_tx].copy()
    if len(risk_df):
        fig = px.scatter(risk_df, x="chargeback_rate", y="high_risk_rate", size="transactions", color="risk_level", hover_name="merchant_name", hover_data=["risk_score", "identity_rate", "failure_rate"], title="Merchant risk matrix", labels={"chargeback_rate":"Chargeback rate (%)", "high_risk_rate":"High-risk KYC rate (%)"})
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        top = risk_df.head(15).sort_values("risk_score", ascending=True)
        fig = px.bar(top, x="risk_score", y="merchant_name", color="risk_level", orientation="h", text="risk_score", title="Highest-priority merchants")
        fig.update_traces(textposition="outside")
        fig.update_layout(height=520, margin=dict(l=10, r=35, t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)
        queue = risk_df.head(30)[["merchant_name", "category", "transactions", "transaction_value", "chargebacks", "chargeback_rate", "identity_conflicts", "high_risk_kyc", "risk_score", "risk_level"]].copy()
        queue["transaction_value"] = queue["transaction_value"].round(2)
        st.dataframe(queue, use_container_width=True, hide_index=True)
        st.download_button("Download investigation queue CSV", queue.to_csv(index=False).encode("utf-8"), "merchant_investigation_queue.csv", "text/csv")
    else:
        st.warning("No merchants meet the selected minimum transaction threshold.")
    st.caption("Risk score = 40% chargeback rate + 25% high-risk KYC rate + 20% identity-conflict rate + 15% failure rate, min-max normalized across the displayed merchant population. It is a prioritization convention, not a confirmed-fraud classifier.")

with tab4:
    st.markdown('<div class="section-title">Chargebacks & KYC Signals</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Chargeback complaints", f"{len(chargebacks):,}")
    c2.metric("Linked transactions", f"{int(filtered['has_chargeback'].sum()):,}")
    c3.metric("Disputed amount", money(filtered.loc[filtered["has_chargeback"], "total_disputed_amount"].sum()))
    left, right = st.columns(2)
    with left:
        reason = chargebacks["reason_code_clean"].fillna("UNKNOWN").astype(str).value_counts().rename_axis("reason").reset_index(name="count").head(10).sort_values("count", ascending=True)
        fig = px.bar(reason, x="count", y="reason", orientation="h", title="Chargeback reasons")
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        sev = chargebacks["severity"].fillna("UNKNOWN").astype(str).str.upper().replace({"H":"HIGH","M":"MEDIUM","L":"LOW"}).value_counts().rename_axis("severity").reset_index(name="count")
        fig = px.pie(sev, names="severity", values="count", hole=.52, title="Chargeback severity")
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)
    risk_series = filtered["risk_segments"].fillna("Unknown").astype(str).str.split(",").explode().str.strip().str.upper().replace("", "UNKNOWN").value_counts().rename_axis("risk_segment").reset_index(name="count")
    fig = px.bar(risk_series, x="risk_segment", y="count", title="KYC risk-signal distribution")
    fig.update_layout(height=330)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("**Required business question — current quarter**")
    latest_q = filtered["date"].dt.to_period("Q").max(); qdf = filtered[filtered["date"].dt.to_period("Q") == latest_q].copy()
    min_q_tx = st.number_input("Minimum transactions for category ranking", min_value=1, max_value=100, value=10, help="Avoids ranking tiny categories where one chargeback can create a misleading 100% ratio.")
    qcat = qdf.groupby("category", as_index=False).agg(transactions=("txn_id","count"), chargebacks=("has_chargeback","sum")); qcat["chargeback_ratio"] = np.where(qcat["transactions"] > 0, qcat["chargebacks"] / qcat["transactions"] * 100, 0); qcat_rank = qcat[qcat["transactions"] >= min_q_tx].sort_values("chargeback_ratio", ascending=False)
    if len(qcat_rank):
        winner = qcat_rank.iloc[0]
        st.success(f"**{winner['category']}** has the highest chargeback-to-transaction ratio in {latest_q}: **{winner['chargeback_ratio']:.2f}%** ({int(winner['chargebacks'])} chargeback-linked / {int(winner['transactions'])} transactions).")
        fig = px.bar(qcat_rank.sort_values("chargeback_ratio"), x="chargeback_ratio", y="category", orientation="h", text="chargeback_ratio", title="Current-quarter chargeback ratio")
        fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
        fig.update_layout(height=350, margin=dict(l=10, r=35, t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else: st.warning("No category meets the selected minimum transaction threshold.")
    st.caption("Chargeback linkage is based on normalized transaction ID. The dataset does not provide a universal confirmed-fraud label.")

with tab5:
    st.markdown('<div class="section-title">🤖 Ask UPI Sentinel</div>', unsafe_allow_html=True)
    st.write("Ask a natural-language question. This local text-to-chart layer handles common business questions without paid API credits.")
    examples = ["Show transaction trend by month", "Compare chargeback rate by merchant category", "Show failed transactions by category", "Which merchant category has the highest chargeback ratio this quarter?", "Show identity conflicts by merchant category"]
    st.caption("Try: " + " · ".join(examples))
    question = st.text_input("Business question", placeholder="e.g. Which merchant category has the highest chargeback ratio this quarter?")
    if st.button("Analyze", type="primary") and question.strip():
        q = question.lower().strip()
        if "highest" in q and "chargeback" in q and "ratio" in q:
            latest_q = filtered["date"].dt.to_period("Q").max(); qdf = filtered[filtered["date"].dt.to_period("Q") == latest_q]; qcat = qdf.groupby("category", as_index=False).agg(transactions=("txn_id","count"), chargebacks=("has_chargeback","sum")); qcat["ratio"] = np.where(qcat["transactions"] > 0, qcat["chargebacks"] / qcat["transactions"] * 100, 0); qcat = qcat[qcat["transactions"] >= 10].sort_values("ratio", ascending=False)
            if len(qcat):
                w = qcat.iloc[0]; st.success(f"{w['category']} is highest in {latest_q} at {w['ratio']:.2f}% ({int(w['chargebacks'])}/{int(w['transactions'])}).")
                fig = px.bar(qcat.sort_values("ratio"), x="ratio", y="category", orientation="h", text="ratio", title="Chargeback ratio by category"); fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside"); st.plotly_chart(fig, use_container_width=True)
            else: st.warning("Not enough category volume to produce a stable ranking.")
        elif ("trend" in q or "over time" in q) and ("transaction" in q or "volume" in q):
            trend = filtered.assign(month=filtered["date"].dt.to_period("M").astype(str)).groupby("month", as_index=False).agg(transactions=("txn_id","count")); fig = px.line(trend, x="month", y="transactions", markers=True, title="Transaction trend by month"); st.plotly_chart(fig, use_container_width=True); st.info("Chart selected: **Line** — appropriate for a time trend.")
        elif "chargeback" in q and "category" in q:
            x = filtered.groupby("category", as_index=False).agg(transactions=("txn_id","count"), chargebacks=("has_chargeback","sum")); x["ratio"] = np.where(x["transactions"] > 0, x["chargebacks"] / x["transactions"] * 100, 0); fig = px.bar(x.sort_values("ratio"), x="ratio", y="category", orientation="h", text="ratio", title="Chargeback rate by category"); fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside"); st.plotly_chart(fig, use_container_width=True); st.info("Chart selected: **Bar** — appropriate for category comparison.")
        elif "failed" in q and "category" in q:
            x = filtered.assign(failed=filtered["status_clean"].eq("FAILED")).groupby("category", as_index=False).agg(failed=("failed","sum")); fig = px.bar(x.sort_values("failed"), x="failed", y="category", orientation="h", title="Failed transactions by category"); st.plotly_chart(fig, use_container_width=True); st.info("Chart selected: **Bar** — appropriate for category comparison.")
        elif "identity" in q and "category" in q:
            x = filtered.groupby("category", as_index=False).agg(identity_conflicts=("identity_conflict_flag","sum")); fig = px.bar(x.sort_values("identity_conflicts"), x="identity_conflicts", y="category", orientation="h", title="Identity conflicts by category"); st.plotly_chart(fig, use_container_width=True); st.info("Chart selected: **Bar** — appropriate for category comparison.")
        else: st.warning("I don't have a safe local mapping for that question yet. Start with one of the example questions above.")
    st.markdown("---")
    st.caption("Agent-ready design: the same interface can later connect to an LLM while keeping data execution constrained to approved metrics and columns.")

st.markdown("---")
st.caption("UPI Sentinel • Built for TransOrg AgentIQ Datathon • Risk scores are analytical prioritization signals, not confirmed fraud labels.")
