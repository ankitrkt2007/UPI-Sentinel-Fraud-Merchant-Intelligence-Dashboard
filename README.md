# UPI Sentinel — Fraud & Merchant Intelligence Dashboard

A Streamlit + Plotly dashboard for the TransOrg AgentIQ Datathon track:

**FinTech — UPI Fraud Ring & Merchant Analytics**

## What it includes

- Executive KPI overview
- Transaction value and status trends
- Merchant-category analytics
- Chargeback-rate analysis
- Merchant risk prioritization
- Chargeback reason/severity analysis
- KYC identity-conflict and high-risk signals
- Required current-quarter chargeback-ratio question
- Local natural-language text-to-chart layer for common questions
- CSV export of the merchant investigation queue
- Responsive Streamlit UI

## Folder structure

```text
upi_sentinel_dashboard/
├── app.py
├── requirements.txt
├── README.md
└── data/
    ├── transactions.b64.01 ... transactions.b64.07
    ├── merchants.b64.01 ... merchants.b64.03
    ├── transactions_kyc_risk.b64.01 ... transactions_kyc_risk.b64.02
    └── chargebacks.b64.01
```

The data files are gzip-compressed CSV payloads split into Base64 chunks so they remain practical for GitHub. The app reconstructs them automatically at startup.

## Run locally

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

## Deploy

The project can be deployed as a Streamlit app. Select:

- Main file: `app.py`
- Python version: a supported current Python version
- Dependencies: `requirements.txt`

## Important analytical note

The transaction dataset does not provide a universal confirmed-fraud label. Therefore, the dashboard distinguishes **chargeback-linked transactions, KYC identity conflicts, high-risk KYC signals and failed transactions** from confirmed fraud.

The merchant risk score is an analytical prioritization score created for the dashboard. It is not an official fraud classifier.

Merchant category labels are intentionally messy in the supplied master data, so the dashboard uses MCC-based category normalization where an MCC mapping is available.

## Suggested presentation story

1. How large and healthy is the UPI transaction ecosystem?
2. Where is transaction failure and chargeback activity concentrated?
3. Which merchant categories need attention?
4. Which individual merchants should be investigated first?
5. What KYC/identity anomalies are visible?
6. Which category has the highest chargeback-to-transaction ratio?
7. How can the text-to-chart layer help analysts ask follow-up questions?

## Datathon compliance

The dashboard is designed around the competition's requirements for an interactive dashboard, core business metrics, storytelling, and an eventual graph-first AI agent.
