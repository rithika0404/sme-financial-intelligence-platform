import streamlit as st
import pandas as pd
import numpy as np

import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors


# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="SME Financial Intelligence Platform",
    layout="wide"
)


# ==========================================
# PREMIUM UI FIX (BACKGROUND + DARK MODE SAFE)
# ==========================================
st.markdown("""
<style>

/* Full App Background */
[data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at top, #f8fbff, #ffffff) !important;
}

/* Sidebar Background */
[data-testid="stSidebar"] {
    background: #f1f5f9 !important;
}

/* Hero Banner */
.hero {
    padding: 35px;
    border-radius: 25px;
    background: linear-gradient(90deg, #2563eb, #06b6d4);
    color: white;
    margin-bottom: 25px;
    box-shadow: 0px 8px 25px rgba(0,0,0,0.18);
}

.hero h1 {
    font-size: 44px;
    font-weight: 900;
}

/* KPI Cards */
.card {
    background: rgba(255,255,255,0.97) !important;
    padding: 22px;
    border-radius: 18px;
    box-shadow: 0px 6px 18px rgba(0,0,0,0.08);
    text-align: center;
    color: #0f172a !important;
}

/* Recommendation Cards */
.reco-card {
    background: rgba(255,255,255,0.97) !important;
    padding: 18px;
    border-radius: 16px;
    margin-bottom: 12px;
    border-left: 6px solid #2563eb;
    box-shadow: 0px 4px 14px rgba(0,0,0,0.07);
}

/* Section Titles */
.section-title {
    font-size: 22px;
    font-weight: 900;
    margin: 20px 0 15px;
    color: #0f172a !important;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# HERO HEADER
# ==========================================
st.markdown("""
<div class="hero">
    <h1>💰 SME Financial Intelligence Platform</h1>
    <p>Credit Risk • Forecasting • Compliance • CCC Optimizer • Advisory • Bookkeeping • Investor PDF</p>
</div>
""", unsafe_allow_html=True)


# ==========================================
# LOAD BENCHMARKS
# ==========================================
bench_df = pd.read_csv("industry_benchmarks.csv")


# ==========================================
# SIDEBAR INPUTS
# ==========================================
st.sidebar.markdown("## ⚙️ Business Profile")

company = st.sidebar.text_input("Business Name", "ABC Traders Pvt Ltd")

industry = st.sidebar.selectbox(
    "Industry Segment",
    bench_df["Industry"].unique()
)

st.sidebar.markdown("### Upload SME Financial File")
uploaded_file = st.sidebar.file_uploader(
    "CSV/XLSX File",
    type=["csv", "xlsx"]
)

st.sidebar.markdown("### Upload Transactions (Optional)")
tx_file = st.sidebar.file_uploader(
    "transactions.csv",
    type=["csv"]
)

st.sidebar.markdown("### Demo Mode")
if st.sidebar.button("Use Demo Dataset"):
    uploaded_file = "sample_financials_12months.csv"

st.sidebar.info("Required Columns: Month, Revenue, Expenses, Debt")


# ==========================================
# READ DATA
# ==========================================
if uploaded_file is None:
    st.warning("👈 Upload SME financial file or click Demo Dataset.")
    st.stop()

if isinstance(uploaded_file, str):
    df = pd.read_csv(uploaded_file)
elif uploaded_file.name.endswith(".xlsx"):
    df = pd.read_excel(uploaded_file)
else:
    df = pd.read_csv(uploaded_file)

required_cols = ["Month", "Revenue", "Expenses", "Debt"]

if not all(col in df.columns for col in required_cols):
    st.error("❌ File must contain Month, Revenue, Expenses, Debt")
    st.stop()


# OPTIONAL COLS
has_tax = "Tax" in df.columns
has_receivables = "Receivables" in df.columns
has_payables = "Payables" in df.columns
has_inventory = "Inventory" in df.columns


# ==========================================
# CORE METRICS
# ==========================================
revenue = df["Revenue"].sum()
expenses = df["Expenses"].sum()
debt = df["Debt"].sum()

profit = revenue - expenses
profit_margin = profit / revenue
debt_ratio = debt / revenue

liquidity_ratio = revenue / (expenses + 1)
annual_debt_payment = debt * 0.2
dscr = profit / (annual_debt_payment + 1)


# ==========================================
# HEALTH SCORE (REALISTIC)
# ==========================================
score = 50
score += profit_margin * 120
score -= debt_ratio * 100

if liquidity_ratio >= 1.5:
    score += 15
elif liquidity_ratio >= 1.0:
    score += 5
else:
    score -= 15

if dscr >= 2:
    score += 15
elif dscr >= 1:
    score += 5
else:
    score -= 20

score = max(0, min(100, score))


# CREDIT RATING
if score >= 85:
    credit_rating = "AAA (Excellent)"
elif score >= 70:
    credit_rating = "A (Good)"
elif score >= 55:
    credit_rating = "BBB (Moderate)"
elif score >= 40:
    credit_rating = "B (Risky)"
else:
    credit_rating = "CCC (High Risk)"


# ==========================================
# FORECASTING
# ==========================================
X = np.arange(len(df)).reshape(-1, 1)
y = df["Revenue"].values

model = LinearRegression()
model.fit(X, y)

future_X = np.arange(len(df), len(df) + 3).reshape(-1, 1)
forecast = model.predict(future_X)

future_months = ["Next-1", "Next-2", "Next-3"]
upper = forecast * 1.1
lower = forecast * 0.9


# ==========================================
# CASH CONVERSION CYCLE
# ==========================================
ccc = None
if has_receivables and has_inventory and has_payables:
    avg_receivables = df["Receivables"].mean()
    avg_inventory = df["Inventory"].mean()
    avg_payables = df["Payables"].mean()

    dso = (avg_receivables / revenue) * 365
    dio = (avg_inventory / expenses) * 365
    dpo = (avg_payables / expenses) * 365

    ccc = dso + dio - dpo


# ==========================================
# PRODUCT ADVISORY
# ==========================================
products = [
    ("Working Capital Overdraft", "Short-term liquidity support"),
    ("Invoice Financing", "Unlock receivables cash faster"),
    ("Equipment Finance", "Modernize operations with capex support")
]

if profit_margin > 0.2:
    products.append(("Growth Expansion Credit Line", "Strong profitability → eligible"))

if debt_ratio > 0.4:
    products.append(("Debt Restructuring Loan", "Reduce repayment burden"))


# ==========================================
# TABS UI
# ==========================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Dashboard",
    "📈 Forecast",
    "✅ Compliance",
    "💼 CCC Optimizer",
    "🏦 Advisory",
    "📒 Bookkeeping + PDF"
])


# ==========================================
# TAB 1 DASHBOARD
# ==========================================
with tab1:
    st.markdown("<div class='section-title'>📊 Executive Snapshot</div>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"<div class='card'><h2>₹{revenue:,.0f}</h2><p>Total Revenue</p></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='card'><h2>₹{profit:,.0f}</h2><p>Net Profit</p></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='card'><h2>{score:.0f}/100</h2><p>Health Score</p></div>", unsafe_allow_html=True)
    c4.markdown(f"<div class='card'><h2>{credit_rating.split()[0]}</h2><p>Credit Rating</p></div>", unsafe_allow_html=True)

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={"text": "Financial Health Score"},
        gauge={"axis": {"range": [0, 100]}}
    ))
    st.plotly_chart(fig, use_container_width=True)


# ==========================================
# TAB 2 FORECAST
# ==========================================
with tab2:
    st.markdown("<div class='section-title'>📈 Revenue Forecast</div>", unsafe_allow_html=True)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["Month"], y=df["Revenue"],
        mode="lines+markers",
        name="Historical Revenue"
    ))

    fig.add_trace(go.Scatter(
        x=future_months, y=forecast,
        mode="lines+markers",
        name="Forecast",
        line=dict(dash="dot")
    ))

    fig.add_trace(go.Scatter(
        x=future_months, y=upper,
        mode="lines",
        opacity=0.2,
        name="Upper Bound"
    ))

    fig.add_trace(go.Scatter(
        x=future_months, y=lower,
        mode="lines",
        fill="tonexty",
        opacity=0.2,
        name="Lower Bound"
    ))

    st.plotly_chart(fig, use_container_width=True)


# ==========================================
# TAB 3 COMPLIANCE
# ==========================================
with tab3:
    st.markdown("<div class='section-title'>✅ Compliance & GST Check</div>", unsafe_allow_html=True)

    if has_tax:
        st.success("✅ GST/Tax metadata available. Compliance-ready.")
    else:
        st.warning("⚠ Tax/GST column missing.")


# ==========================================
# TAB 4 CCC OPTIMIZER
# ==========================================
with tab4:
    st.markdown("<div class='section-title'>💼 Cash Conversion Cycle Optimizer</div>", unsafe_allow_html=True)

    if ccc:
        st.metric("Cash Conversion Cycle", f"{ccc:.1f} days")
        st.success("Lower CCC = Faster cash recovery.")
    else:
        st.warning("CCC requires Receivables + Inventory + Payables columns.")


# ==========================================
# TAB 5 ADVISORY
# ==========================================
with tab5:
    st.markdown("<div class='section-title'>🏦 Bank/NBFC Recommendations</div>", unsafe_allow_html=True)

    for p, reason in products:
        st.markdown(
            f"<div class='reco-card'>💡 <b>{p}</b><br>🔎 {reason}</div>",
            unsafe_allow_html=True
        )


# ==========================================
# TAB 6 BOOKKEEPING + PDF
# ==========================================
with tab6:
    st.markdown("<div class='section-title'>📒 Bookkeeping Assistant</div>", unsafe_allow_html=True)

    if tx_file:
        tx_df = pd.read_csv(tx_file)
        st.dataframe(tx_df)

        st.markdown("### Expense Distribution")
        st.bar_chart(tx_df.groupby("Category")["Amount"].sum())

    else:
        st.info("Upload transactions.csv for automated bookkeeping.")

    st.markdown("<div class='section-title'>📄 Investor One-Page PDF</div>", unsafe_allow_html=True)

    def generate_pdf():
        filename = "Final_Investor_Report.pdf"

        doc = SimpleDocTemplate(filename)
        styles = getSampleStyleSheet()
        story = []

        story.append(Paragraph("SME Investor Financial Report", styles["Title"]))
        story.append(Spacer(1, 10))

        story.append(Paragraph(f"Company: {company}", styles["Normal"]))
        story.append(Paragraph(f"Industry: {industry}", styles["Normal"]))
        story.append(Paragraph(f"Credit Rating: {credit_rating}", styles["Normal"]))
        story.append(Spacer(1, 12))

        data = [
            ["Metric", "Value"],
            ["Revenue", f"₹{revenue:,.0f}"],
            ["Profit", f"₹{profit:,.0f}"],
            ["Profit Margin", f"{profit_margin*100:.1f}%"],
            ["Debt Ratio", f"{debt_ratio*100:.1f}%"],
            ["Liquidity Ratio", f"{liquidity_ratio:.2f}"],
            ["Health Score", f"{score:.0f}/100"]
        ]

        table = Table(data)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2563eb")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ]))

        story.append(table)
        story.append(Spacer(1, 10))

        story.append(Paragraph("Recommended Products:", styles["Heading2"]))
        for p, reason in products[:3]:
            story.append(Paragraph(f"• {p} — {reason}", styles["Normal"]))

        doc.build(story)
        return filename

    if st.button("Generate Final Investor PDF"):
        pdf_file = generate_pdf()
        with open(pdf_file, "rb") as f:
            st.download_button("⬇️ Download Investor Report", f, file_name="Final_Investor_Report.pdf")

        st.success("✅ Investor Report Generated Successfully!")
