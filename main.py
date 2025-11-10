# 🧊 ICEBERG Query API
# Version: 2.6 (Chart-Ready for ICEBERG GPT)
# ---------------------------------------------------
# Unified multi-table query engine with formatted tables
# and automatic chart generation for GPT visualization.
# ---------------------------------------------------

from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
import psycopg2
import pandas as pd
import os, re
from io import StringIO, BytesIO
from dotenv import load_dotenv
import base64
import matplotlib.pyplot as plt

# ==============================================================
#  SETUP
# ==============================================================

load_dotenv()
app = FastAPI(title="ICEBERG Query API", version="2.6")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"status": "ok", "message": "ICEBERG API is live"}

# ==============================================================
#  DB HELPER
# ==============================================================

def run_sql(sql: str):
    try:
        conn = psycopg2.connect(os.getenv("DATABASE_URL"))
        df = pd.read_sql(sql, conn)
        conn.close()
        return df
    except Exception as e:
        return pd.DataFrame([{"error": str(e), "sql": sql}])

# ==============================================================
#  MAPPINGS
# ==============================================================

metric_map = {
    "noi": ("fact_noi_enterprise", "iceberg_metric_code", "raw_value", "TOTAL_NOI", "total_noi"),
    "net operating income": ("fact_noi_enterprise", "iceberg_metric_code", "raw_value", "TOTAL_NOI", "total_noi"),
    "enterprise value": ("fact_capitalization_enterprise", "iceberg_metric_code", "raw_value", "ENTERPRISE_VALUE", "enterprise_value"),
    "total debt": ("fact_capitalization_enterprise", "iceberg_metric_code", "raw_value", "TOTAL_DEBT", "total_debt"),
    "cap rate": ("derived_enterprise", "derived_metric_code", "derived_value", "IMPLIED_CAP_RATE", "implied_cap_rate"),
    "implied cap rate": ("derived_enterprise", "derived_metric_code", "derived_value", "IMPLIED_CAP_RATE", "implied_cap_rate"),
}

reit_tickers = ["PLD","EGP","FR","DLR","COLD","LXP","ILPT","REXR","STAG","TRNO","LINE"]

# ==============================================================
#  PARSER
# ==============================================================

def interpret_components(question: str):
    q = question.lower().strip()

    detected = []
    for key, (table, mcol, vcol, code, alias) in metric_map.items():
        if key in q:
            detected.append((table, mcol, vcol, code, alias))

    tickers = [t for t in reit_tickers if t.lower() in q]
    m = re.search(r"(q[1-4])\s?(\d{4})", q)
    fq, fy = None, None
    if m:
        fq = int(m.group(1).replace("q", ""))
        fy = int(m.group(2))
    return detected, tickers, fq, fy

# ==============================================================
#  QUERY ORCHESTRATOR
# ==============================================================

def orchestrate_query(question: str):
    metrics, tickers, fq, fy = interpret_components(question)
    if not metrics:
        return pd.DataFrame([{"error": "No recognized metrics"}]), []

    grouped = {}
    for table, mcol, vcol, code, alias in metrics:
        if table not in grouped:
            grouped[table] = {"mcol": mcol, "vcol": vcol, "codes": set(), "alias_map": {}}
        grouped[table]["codes"].add(code)
        grouped[table]["alias_map"][code] = alias

    sqls, dfs = [], []
    for table, cfg in grouped.items():
        mcol, vcol, codes, alias_map = cfg["mcol"], cfg["vcol"], cfg["codes"], cfg["alias_map"]
        code_list = ", ".join(f"'{c}'" for c in codes)
        sql = f"""
        SELECT reit_ticker, {mcol} AS metric_code, {vcol} AS value,
               fiscal_year, fiscal_quarter
        FROM iceberg.{table}
        WHERE {mcol} IN ({code_list})
        """
        if tickers:
            tlist = ", ".join(f"'{t}'" for t in tickers)
            sql += f" AND reit_ticker IN ({tlist})"
        if fq and fy:
            sql += f" AND fiscal_quarter={fq} AND fiscal_year={fy}"
        sql += ";"
        sqls.append(sql)
        df = run_sql(sql)
        if "error" not in df.columns and not df.empty:
            for code, alias in alias_map.items():
                df.loc[df["metric_code"] == code, "metric_code"] = alias
            dfs.append(df)

    if not dfs:
        return pd.DataFrame([{"error": "No data"}]), sqls

    merged = pd.concat(dfs)
    pivot = merged.pivot_table(
        index=["reit_ticker", "fiscal_year", "fiscal_quarter"],
        columns="metric_code",
        values="value",
        aggfunc="first"
    ).reset_index()

    pivot = pivot.fillna("")
    return pivot, sqls

# ==============================================================
#  CHART GENERATOR
# ==============================================================

def make_chart(df: pd.DataFrame) -> str:
    """Create bar chart for key metrics and return base64 string."""
    if df.empty or "reit_ticker" not in df.columns:
        return ""

    # Safely convert numeric columns (ignore blanks)
    df_chart = df.copy()
    for col in ["total_noi", "implied_cap_rate"]:
        if col in df_chart.columns:
            df_chart[col] = pd.to_numeric(df_chart[col], errors="coerce")

    fig, ax = plt.subplots(figsize=(8, 4))
    labels = df_chart["reit_ticker"]

    if "total_noi" in df_chart.columns and df_chart["total_noi"].notna().any():
        ax.bar(labels, df_chart["total_noi"], label="NOI ($)", alpha=0.7)
        ax.set_ylabel("NOI ($)")

    if "implied_cap_rate" in df_chart.columns and df_chart["implied_cap_rate"].notna().any():
        ax2 = ax.twinx()
        ax2.plot(labels, df_chart["implied_cap_rate"], color="red", marker="o", label="Cap Rate (%)")
        ax2.set_ylabel("Cap Rate (%)")

    ax.set_title("Industrial REITs – NOI and Cap Rate")
    plt.xticks(rotation=45)
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

# ==============================================================
#  FORMATTERS
# ==============================================================

def df_to_markdown(df: pd.DataFrame) -> str:
    buf = StringIO()
    subset = df.head(15)
    buf.write("| " + " | ".join(subset.columns) + " |\n")
    buf.write("|" + " --- |" * len(subset.columns) + "\n")
    for _, r in subset.iterrows():
        row = [str(v) for v in r]
        buf.write("| " + " | ".join(row) + " |\n")
    return buf.getvalue()

def build_summary(df: pd.DataFrame) -> str:
    if df.empty or "reit_ticker" not in df.columns:
        return "No data available."
    fy = int(df.iloc[0].get("fiscal_year", 0))
    fq = int(df.iloc[0].get("fiscal_quarter", 0))
    df_fmt = df.copy()

    # format values
    if "total_noi" in df_fmt.columns:
        df_fmt["total_noi"] = df_fmt["total_noi"].apply(
            lambda x: f"${float(x):,.0f}" if str(x).replace('.', '', 1).isdigit() else x
        )
    if "implied_cap_rate" in df_fmt.columns:
        df_fmt["implied_cap_rate"] = df_fmt["implied_cap_rate"].apply(
            lambda x: f"{float(x):.2f} %" if str(x).replace('.', '', 1).isdigit() else x
        )

    cols = ["reit_ticker"]
    for c in ["total_noi", "implied_cap_rate", "enterprise_value", "total_debt"]:
        if c in df_fmt.columns:
            cols.append(c)
    for c in ["fiscal_year", "fiscal_quarter"]:
        if c in df_fmt.columns:
            cols.append(c)
    df_fmt = df_fmt[cols]

    return f"Industrial REIT metrics in Q{fq} {fy}:\n\n" + df_to_markdown(df_fmt)

# ==============================================================
#  ENDPOINT
# ==============================================================

@app.post("/query")
def query(payload: dict = Body(...)):
    question = payload.get("question", "")
    df, sqls = orchestrate_query(question)
    if not df.empty:
        df = df.fillna("")
    summary = build_summary(df) if "error" not in df.columns else None
    chart_b64 = make_chart(df)
    return {
        "summary": f"Ran query for: {question}",
        "sql": sqls,
        "summary_text": summary,
        "chart_base64": chart_b64,
        "result": df.to_dict(orient="records"),
    }
