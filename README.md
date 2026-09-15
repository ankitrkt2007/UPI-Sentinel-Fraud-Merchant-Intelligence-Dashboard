# 💳 UPI Sentinel — Fraud & Merchant Intelligence Dashboard

> **TransOrg AgentIQ Datathon | FinTech & BFSI Track**

UPI Sentinel is a **team-built Streamlit + Plotly analytics platform** developed for the TransOrg AgentIQ Datathon challenge **“UPI Fraud Ring & Merchant Analytics.”**

Our team built the dashboard to turn cleaned UPI transaction, merchant, KYC and chargeback data into an interactive investigation and decision-support experience.

## 👥 Team Project

This repository contains the work developed collaboratively by our team. The dashboard, analytical logic, data preparation outputs and presentation-ready insights are intended to be submitted as a **team solution**, not an individual project.

**Team:** Add the names / GitHub profiles of all team members here.

## 🎯 Problem We Address

The solution focuses on identifying transaction and merchant patterns that may indicate elevated financial risk, including:

- Chargeback concentration
- Failed transaction patterns
- KYC identity conflicts
- High-risk KYC signals
- Merchant-level risk prioritization
- Category-level chargeback ratios

The dashboard deliberately does **not** claim that every flagged transaction is confirmed fraud. These are analytical signals that help investigators prioritize review.

## 📊 Dashboard Modules

### 1. Executive Overview
- Total transactions
- Successful transactions
- Success rate
- Successful transaction value
- Chargeback-linked transactions
- Failed transactions
- Identity conflicts
- High-risk KYC records
- Transaction value trend
- Transaction status mix

### 2. Fraud Intelligence
- Transaction volume by merchant category
- Chargeback rate by category
- Daily failed and chargeback-linked activity
- Interactive filtering by date, category and transaction status

### 3. Merchant Risk
- Merchant risk matrix
- Risk-score ranking
- Minimum transaction threshold
- Investigation queue
- CSV export for analyst follow-up

The merchant risk score is an **analytical prioritization score** based on chargeback rate, high-risk KYC rate, identity-conflict rate and transaction failure rate. It is not an official fraud classifier.

### 4. Chargebacks & KYC
- Chargeback reason analysis
- KYC risk-segment analysis
- Latest-quarter chargeback-ratio ranking
- Minimum transaction threshold to avoid misleading rankings from very small samples

### 5. Ask UPI Sentinel
The dashboard includes a deterministic natural-language text-to-chart layer for common business questions, providing a foundation for the competition's proposed agentic analytics layer.

Example:

> “Which merchant category has the highest chargeback ratio this quarter?”

## 🗂️ Project Structure

```text
UPI-Sentinel-Fraud-Merchant-Intelligence-Dashboard/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── data/
    ├── transactions_standardized.csv
    ├── merchants_clean.csv
    ├── transactions_kyc_merged.csv
    └── chargebacks_final_analysis.csv
```

**Important:** the four CSV files must be present in `data/` for the deployed dashboard to load.

## 🛠️ Technology Stack

- **Python**
- **Pandas / NumPy** — data processing and analytics
- **Plotly** — interactive visualizations
- **Streamlit** — dashboard and application layer
- **GitHub** — version control and submission repository
- **Streamlit Community Cloud** — deployment

## 🚀 Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 🌐 Deployment

The dashboard is deployed using Streamlit Community Cloud.

**Live Dashboard:** https://upi-sentinel-dashboard.streamlit.app

**GitHub Repository:** https://github.com/ankitrkt2007/UPI-Sentinel-Fraud-Merchant-Intelligence-Dashboard

## 🔍 Analytical Notes

- Transaction and merchant identifiers are normalized before joins.
- Merchant categories are normalized using MCC mappings where available.
- Chargeback-linked transactions are identified by normalized transaction identifiers.
- KYC identity conflicts and high-risk KYC signals are treated as risk indicators rather than confirmed fraud labels.
- Merchant risk scores are intended for investigation prioritization.
- Category rankings use a minimum transaction threshold where appropriate to reduce small-sample distortion.

## 💡 Datathon Story

Our team designed the dashboard around the following investigation flow:

1. **Measure** the overall transaction ecosystem.
2. **Locate** failure and chargeback concentrations.
3. **Compare** merchant categories.
4. **Prioritize** individual merchants for investigation.
5. **Connect** transaction anomalies with KYC risk signals.
6. **Answer** business questions through natural-language analytics.
7. **Extend** the workflow toward an agentic graph-based investigation layer.

## ⚠️ Disclaimer

UPI Sentinel is a decision-support and analytical system created for the datathon. Risk indicators and scores should be validated by appropriate investigators before any operational, financial or compliance decision is made.

---

**Built collaboratively by our team for the TransOrg AgentIQ Datathon 2026.**
