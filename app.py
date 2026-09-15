from pathlib import Path
import re
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="UPI Sentinel | Fraud & Merchant Intelligence", page_icon="💳", layout="wide")
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

MCC_CATEGORY = {
    "4131":"Transport", "4814":"Telecom", "5311":"Department Store", "5411":"Grocery",
    "5699":"Apparel", "5812":"Food Services", "5912":"Healthcare",
    "5942":"Books & Stationery", "5999":"Other Retail", "7011":"Hotel & Lodging"
}

def norm_id(v):
    if pd.isna(v): return ""
    return re.sub(r"[^A-Z0-9]", "", str(v).upper().strip())

def category(v):
    if pd.isna(v): return "Unknown"
    s = str(v).strip().lower().replace("_", " ").replace("-", " ")
    if any(x in s for x in ["grocery", "kirana"]): return "Grocery"
    if any(x in s for x in ["department", "dept store"]): return "Department Store"
    if any(x in s for x in ["cloth", "apparel", "garment", "fashion"]): return "Apparel"
    if any(x in s for x in ["food", "restaurant", "eating"]): return "Food Services"
    if any(x in s for x in ["hotel", "hospitality"]): return "Hotel & Lodging"
    if any(x in s for x in ["telecom", "phone", "mobile"]): return "Telecom"
    if any(x in s for x in ["transport", "travel", "bus", "taxi"]): return "Transport"
    if any(x in s for x in ["pharmacy", "medical", "chemist"]): return "Healthcare"
    if any(x in s for x in ["book", "stationery"]): return "Books & Stationery"
    return str(v).strip().title()

def money(v):
    v = float(v or 0)
    if abs(v) >= 1e7: return f"₹{v/1e7:.2f} Cr"
    if abs(v) >= 1e5: return f"₹{v/1e5:.2f} L"
    return f"₹{v:,.0f}"

def pct(v): return f"{v:.2f}%"

@st.cache_data(show_spinner=False)
def load_data():
    required = {
        "transactions_standardized.csv": "transactions",
        "merchants_clean.csv": "merchants",
        "transactions_kyc_merged.csv": "KYC/transaction risk",
        "chargebacks_final_analysis.csv": "chargebacks",
    }
    missing = [f for f in required if not (DATA / f).exists()]
    if missing:
        raise FileNotFoundError("Missing data/ files: " + ", ".join(missing) + ". Upload the four CSV files from the project ZIP to the repository's data/ folder.")

    tx = pd.read_csv(DATA / "transactions_standardized.csv")
    mer = pd.read_csv(DATA / "merchants_clean.csv")
    kyc = pd.read_csv(DATA / "transactions_kyc_merged.csv")
    cb = pd.read_csv(DATA / "chargebacks_final_analysis.csv")

    tx["date"] = pd.to_datetime(tx["timestamp_clean"], errors="coerce")
    tx["amount_clean"] = pd.to_numeric(tx["amount_clean"], errors="coerce").fillna(0)
    tx["txn_key"] = tx["txn_id"].map(norm_id)
    tx["merchant_key"] = tx["merchant_id"].map(norm_id)
    tx["mcc_key"] = pd.to_numeric(tx["mcc_clean"], errors="coerce").round().astype("Int64").astype(str).replace("<NA>", "")
    tx["category"] = tx["mcc_key"].map(MCC_CATEGORY).fillna("Unknown")

    mer["merchant_key"] = mer["merchant_id"].map(norm_id)
    mer["category_clean"] = mer["merchant_category"].map(category)
    cols = [c for c in ["merchant_key","merchant_name","merchant_category","category_clean","business_type","city","state","merchant_status"] if c in mer.columns]
    tx = tx.merge(mer.drop_duplicates("merchant_key")[cols], on="merchant_key", how="left")

    cb["txn_key"] = cb["txn_id_clean"].map(norm_id)
    cb["disputed_amount_clean"] = pd.to_numeric(cb["disputed_amount_clean"], errors="coerce").fillna(0)
    tx["has_chargeback"] = tx["txn_key"].isin(set(cb["txn_key"].dropna()))
    cb_sum = cb.groupby("txn_key", dropna=False).agg(chargeback_count=("complaint_id","count"), disputed_amount=("disputed_amount_clean","sum")).reset_index()
    tx = tx.merge(cb_sum, on="txn_key", how="left")
    tx["chargeback_count"] = tx["chargeback_count"].fillna(0).astype(int)
    tx["disputed_amount"] = tx["disputed_amount"].fillna(0)

    kyc["txn_key"] = kyc["txn_id"].map(norm_id)
    for c in ["high_risk_record", "identity_conflict_flag"]:
        if c in kyc: kyc[c] = kyc[c].fillna(False).astype(bool)
    rcols = [c for c in ["txn_key","high_risk_record","identity_conflict_flag","risk_segments"] if c in kyc]
    tx = tx.merge(kyc[rcols].drop_duplicates("txn_key"), on="txn_key", how="left")
    tx["high_risk_record"] = tx.get("high_risk_record", False).fillna(False).astype(bool)
    tx["identity_conflict_flag"] = tx.get("identity_conflict_flag", False).fillna(False).astype(bool)
    tx["risk_segments"] = tx.get("risk_segments", "Unknown").fillna("Unknown")
    tx["category"] = tx["category"].replace("Unknown", np.nan).fillna(tx["category_clean"]).fillna("Unknown")
    return tx, mer, cb

def risk_table(df):
    g = df.groupby(["merchant_key","merchant_name","category"], dropna=False).agg(
        transactions=("txn_id","count"), successful=("status_clean",lambda s:(s=="SUCCESS").sum()),
        failed=("status_clean",lambda s:(s=="FAILED").sum()), value=("amount_clean","sum"),
        chargebacks=("has_chargeback","sum"), identity_conflicts=("identity_conflict_flag","sum"),
        high_risk=("high_risk_record","sum")).reset_index()
    g["chargeback_rate"] = g["chargebacks"] / g["transactions"].clip(lower=1) * 100
    g["failure_rate"] = g["failed"] / g["transactions"].clip(lower=1) * 100
    g["identity_rate"] = g["identity_conflicts"] / g["transactions"].clip(lower=1) * 100
    g["high_risk_rate"] = g["high_risk"] / g["transactions"].clip(lower=1) * 100
    def mm(s):
        return pd.Series(0.0,index=s.index) if s.max()==s.min() else (s-s.min())/(s.max()-s.min())*100
    g["risk_score"] = (0.40*mm(g["chargeback_rate"])+0.25*mm(g["high_risk_rate"])+0.20*mm(g["identity_rate"])+0.15*mm(g["failure_rate"])).round(1)
    g["risk_level"] = pd.cut(g["risk_score"],[-.1,30,60,80,100.1],labels=["Low","Medium","High","Critical"])
    return g.sort_values(["risk_score","transactions"],ascending=[False,False])

try:
    tx, merchants, chargebacks = load_data()
except Exception as e:
    st.error(f"Dashboard could not load the data: {e}")
    st.stop()

st.markdown("""
<style>
.stApp{background:#f6f8fb}.hero{padding:22px 26px;border-radius:18px;background:linear-gradient(135deg,#111827,#334155);color:white;margin-bottom:18px}.hero h1{margin:0}.hero p{color:#cbd5e1}.insight{padding:14px 16px;border-left:4px solid #334155;background:white;border-radius:10px;margin:8px 0}div[data-testid="stMetric"]{background:white;border:1px solid #e5e7eb;padding:12px;border-radius:12px}
</style>""", unsafe_allow_html=True)

st.sidebar.markdown("## 💳 UPI Sentinel")
st.sidebar.caption("Fraud & Merchant Intelligence • Team Project")
min_d,max_d=tx.date.min().date(),tx.date.max().date()
rng=st.sidebar.date_input("Date range",(min_d,max_d),min_value=min_d,max_value=max_d)
start,end=(rng if isinstance(rng,tuple) and len(rng)==2 else (rng,rng))
cats=st.sidebar.multiselect("Merchant category",sorted(tx.category.dropna().unique()),default=sorted(tx.category.dropna().unique()))
statuses=st.sidebar.multiselect("Transaction status",sorted(tx.status_clean.dropna().unique()),default=sorted(tx.status_clean.dropna().unique()))
f=tx[(tx.date.dt.date>=start)&(tx.date.dt.date<=end)&tx.category.isin(cats)&tx.status_clean.isin(statuses)].copy()

st.markdown('<div class="hero"><h1>UPI Sentinel</h1><p>Fraud & Merchant Intelligence Platform • TransOrg AgentIQ Datathon</p></div>',unsafe_allow_html=True)
t1,t2,t3,t4,t5=st.tabs(["Executive Overview","Fraud Intelligence","Merchant Risk","Chargebacks & KYC","Ask UPI Sentinel"])

with t1:
    total=len(f); succ=int((f.status_clean=="SUCCESS").sum()); fail=int((f.status_clean=="FAILED").sum()); pend=int((f.status_clean=="PENDING").sum()); val=f.loc[f.status_clean=="SUCCESS","amount_clean"].sum(); cbn=int(f.has_chargeback.sum()); ident=int(f.identity_conflict_flag.sum()); high=int(f.high_risk_record.sum())
    for c,l,v in zip(st.columns(8),["Transactions","Successful","Success rate","Successful value","Chargeback-linked","Failed","Identity conflicts","High-risk KYC"],[f"{total:,}",f"{succ:,}",pct(succ/total*100 if total else 0),money(val),f"{cbn:,}",f"{fail:,}",f"{ident:,}",f"{high:,}"]): c.metric(l,v)
    s=f[f.status_clean=="SUCCESS"].assign(month=f[f.status_clean=="SUCCESS"].date.dt.to_period("M").astype(str)).groupby("month",as_index=False).amount_clean.sum()
    st.plotly_chart(px.area(s,x="month",y="amount_clean",labels={"amount_clean":"Successful transaction value"},title="Successful transaction value trend"),use_container_width=True)
    a,b=st.columns(2)
    with a: st.plotly_chart(px.pie(f.status_clean.value_counts().rename_axis("status").reset_index(name="count"),names="status",values="count",hole=.5,title="Transaction status mix"),use_container_width=True)
    with b: st.plotly_chart(px.bar(f.category.value_counts().rename_axis("category").reset_index(name="transactions").sort_values("transactions"),x="transactions",y="category",orientation="h",title="Transaction volume by category"),use_container_width=True)

with t2:
    c=f.groupby("category",as_index=False).agg(transactions=("txn_id","count"),chargebacks=("has_chargeback","sum"),failed=("status_clean",lambda s:(s=="FAILED").sum()),identity_conflicts=("identity_conflict_flag","sum")); c["chargeback_rate"]=c.chargebacks/c.transactions.clip(lower=1)*100
    a,b=st.columns(2)
    with a: st.plotly_chart(px.bar(c.sort_values("transactions"),x="transactions",y="category",orientation="h",title="Transaction volume"),use_container_width=True)
    with b: st.plotly_chart(px.bar(c.sort_values("chargeback_rate"),x="chargeback_rate",y="category",orientation="h",title="Chargeback rate by category",labels={"chargeback_rate":"Chargeback rate (%)"}),use_container_width=True)
    d=f.assign(day=f.date.dt.date).groupby("day",as_index=False).agg(failed=("status_clean",lambda s:(s=="FAILED").sum()),chargebacks=("has_chargeback","sum"))
    st.plotly_chart(px.line(d,x="day",y=["failed","chargebacks"],title="Daily failed and chargeback-linked activity"),use_container_width=True)

with t3:
    st.caption("Risk score is an analytical prioritization signal, not a confirmed-fraud classifier.")
    threshold=st.slider("Minimum transactions per merchant",1,100,10)
    r=risk_table(f); rr=r[r.transactions>=threshold].copy()
    a,b=st.columns(2)
    with a: st.plotly_chart(px.scatter(rr,x="chargeback_rate",y="high_risk_rate",size="transactions",hover_name="merchant_name",color="risk_level",title="Merchant risk matrix",labels={"chargeback_rate":"Chargeback rate (%)","high_risk_rate":"High-risk KYC rate (%)"}),use_container_width=True)
    with b: st.plotly_chart(px.bar(rr.head(15).sort_values("risk_score"),x="risk_score",y="merchant_name",orientation="h",title="Top merchant risk scores"),use_container_width=True)
    cols=["merchant_name","category","transactions","value","chargebacks","chargeback_rate","identity_conflicts","high_risk","risk_score","risk_level"]
    st.dataframe(rr[cols].head(50),use_container_width=True,hide_index=True)
    st.download_button("Download investigation queue",rr[cols].to_csv(index=False).encode(),"upi_sentinel_investigation_queue.csv","text/csv")

with t4:
    a,b=st.columns(2)
    with a: st.plotly_chart(px.bar(chargebacks.reason.value_counts().rename_axis("reason").reset_index(name="complaints").head(10),x="complaints",y="reason",orientation="h",title="Top chargeback reasons"),use_container_width=True)
    with b:
        seg=f.risk_segments.value_counts().rename_axis("risk_segment").reset_index(name="transactions")
        st.plotly_chart(px.bar(seg,x="transactions",y="risk_segment",orientation="h",title="KYC risk segments"),use_container_width=True)
    q=tx[(tx.date.dt.quarter==tx.date.max().quarter)&(tx.date.dt.year==tx.date.max().year)].groupby("category",as_index=False).agg(transactions=("txn_id","count"),chargebacks=("has_chargeback","sum")); q=q[q.transactions>=10]; q["chargeback_ratio"]=q.chargebacks/q.transactions*100
    if len(q):
        top=q.sort_values("chargeback_ratio",ascending=False).iloc[0]
        st.success(f"Latest-quarter highest chargeback ratio (minimum 10 transactions): **{top.category} — {top.chargeback_ratio:.2f}%** ({int(top.chargebacks)} linked / {int(top.transactions)} transactions)")
        st.dataframe(q.sort_values("chargeback_ratio",ascending=False),use_container_width=True,hide_index=True)
    else: st.info("No category meets the minimum transaction threshold in the latest quarter.")

with t5:
    st.markdown("### Ask UPI Sentinel")
    question=st.text_input("Ask a business question",placeholder="Which merchant category has the highest chargeback ratio this quarter?")
    q=question.lower().strip()
    if question:
        if "highest" in q and "chargeback" in q and "category" in q:
            latest=tx[(tx.date.dt.quarter==tx.date.max().quarter)&(tx.date.dt.year==tx.date.max().year)].groupby("category",as_index=False).agg(transactions=("txn_id","count"),chargebacks=("has_chargeback","sum")); latest=latest[latest.transactions>=10]; latest["chargeback_ratio"]=latest.chargebacks/latest.transactions*100; latest=latest.sort_values("chargeback_ratio",ascending=False)
            if len(latest):
                top=latest.iloc[0]; st.plotly_chart(px.bar(latest,x="chargeback_ratio",y="category",orientation="h",title="Latest-quarter chargeback ratio",labels={"chargeback_ratio":"Chargeback ratio (%)"}),use_container_width=True); st.info(f"**{top.category}** has the highest chargeback ratio at **{top.chargeback_ratio:.2f}%** among categories with at least 10 transactions.")
        elif "trend" in q and ("transaction" in q or "value" in q):
            d=f.assign(month=f.date.dt.to_period("M").astype(str)).groupby("month",as_index=False).amount_clean.sum(); st.plotly_chart(px.line(d,x="month",y="amount_clean",markers=True,title="Monthly transaction value"),use_container_width=True)
        elif "failed" in q and "category" in q:
            d=f[f.status_clean=="FAILED"].category.value_counts().rename_axis("category").reset_index(name="failed"); st.plotly_chart(px.bar(d,x="failed",y="category",orientation="h",title="Failed transactions by category"),use_container_width=True)
        elif "identity" in q and "category" in q:
            d=f.groupby("category",as_index=False).identity_conflict_flag.sum().rename(columns={"identity_conflict_flag":"identity_conflicts"}); st.plotly_chart(px.bar(d.sort_values("identity_conflicts"),x="identity_conflicts",y="category",orientation="h",title="Identity conflicts by category"),use_container_width=True)
        else: st.warning("Try a question about chargeback ratio, transaction trend, failed transactions by category, or identity conflicts by category.")

st.divider()
st.caption("UPI Sentinel is a team-built analytical dashboard. Risk indicators are prioritization signals and should be validated by investigators before action.")
