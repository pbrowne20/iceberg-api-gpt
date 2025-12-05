from typing import Optional, List
from pydantic import BaseModel
from fastapi import FastAPI, Query
import psycopg2
import os

app = FastAPI()

# Connection string
DATABASE_URL = os.getenv("DATABASE_URL")


# ---------------------------
# Pydantic Models
# ---------------------------

class MarketOverviewRequest(BaseModel):
    reit_ticker: Optional[str] = None
    market: Optional[str] = None
    state: Optional[str] = None
    property_type: Optional[str] = None
    limit: int = 50


class ReitMetadataRequest(BaseModel):
    ticker: Optional[str] = None
    sector: Optional[str] = None
    hq_state: Optional[str] = None
    limit: int = 50


# ---------------------------
# Database Helper
# ---------------------------

def run_sql(query: str, params: List):
    """Executes SQL with parameters and returns all rows."""
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    cur.execute(query, params)
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows


# ---------------------------
# /reit_metadata ENDPOINT
# ---------------------------

@app.get("/reit_metadata")
def reit_metadata(
    ticker: Optional[str] = None,
    sector: Optional[str] = None,
    hq_state: Optional[str] = None,
    limit: int = 50
):

    sql = """
        SELECT 
            reit_id,
            reit_name,
            reit_ticker,
            sector,
            subsector,
            hq_city,
            hq_state
        FROM iceberg.dim_reit
        WHERE 1=1
    """

    params = []

    if ticker:
        sql += " AND LOWER(reit_ticker) = LOWER(%s)"
        params.append(ticker)

    if sector:
        sql += " AND LOWER(sector) = LOWER(%s)"
        params.append(sector)

    if hq_state:
        sql += " AND LOWER(hq_state) = LOWER(%s)"
        params.append(hq_state)

    sql += " LIMIT %s"
    params.append(limit)

    rows = run_sql(sql, params)

    return {"count": len(rows), "rows": rows}

# ---------------------------
# /reit_overview ENDPOINT  (NEW)
# ---------------------------

@app.get("/reit_overview")
def reit_overview(
    reit_ticker: Optional[str] = None,
    metric: Optional[str] = None,
    limit: int = 50
):

    sql = """
        SELECT 
            dr.reit_ticker,
            fr.iceberg_metric_code,
            fr.metric_value,
            fr.unit,
            dt.reporting_period
        FROM iceberg.fact_reit fr
        JOIN iceberg.dim_reit dr ON dr.reit_id = fr.reit_id
        JOIN iceberg.dim_time dt ON dt.time_id = fr.time_id
        WHERE 1=1
    """

    params = []

    # Filter by REIT
    if reit_ticker:
        sql += " AND LOWER(dr.reit_ticker) = LOWER(%s)"
        params.append(reit_ticker)

    # Filter by metric (e.g., 'noi', 'total_debt')
    if metric:
        sql += " AND LOWER(fr.iceberg_metric_code) = LOWER(%s)"
        params.append(metric)

    # Ordering / limit
    sql += """
        ORDER BY 
            dr.reit_ticker,
            fr.iceberg_metric_code,
            dt.reporting_period DESC
        LIMIT %s
    """
    params.append(limit)

    rows = run_sql(sql, params)

    return {"count": len(rows), "rows": rows}



# ---------------------------
# /market_overview ENDPOINT
# ---------------------------

@app.get("/market_overview")
def market_overview(
    reit_ticker: Optional[str] = None,
    market: Optional[str] = None,
    state: Optional[str] = None,
    property_type: Optional[str] = None,
    limit: int = 50
):

    sql = """
        SELECT 
            dr.reit_ticker,
            dm.market_name,
            dm.state_code,
            brm.submarket_type,
            fm.property_type_code,
            fm.iceberg_metric_code,
            fm.metric_value,
            fm.unit,
            dt.reporting_period
        FROM iceberg.fact_market fm
        JOIN iceberg.dim_reit dr ON dr.reit_id = fm.reit_id
        JOIN iceberg.dim_market dm ON dm.market_id = fm.market_id
        JOIN iceberg.bridge_reit_market brm 
              ON brm.reit_id = fm.reit_id 
             AND brm.market_id = fm.market_id
        JOIN iceberg.dim_time dt ON dt.time_id = fm.time_id
        WHERE 1=1
    """

    params = []

    # ---------------------------
    # REQUIRED: ticker filter
    # ---------------------------
    if reit_ticker:
        sql += " AND LOWER(dr.reit_ticker) = LOWER(%s)"
        params.append(reit_ticker)

    # ---------------------------
    # OPTIONAL: market filter
    # ---------------------------
    if market:
        sql += " AND LOWER(dm.market_name) = LOWER(%s)"
        params.append(market)

    # ---------------------------
    # OPTIONAL: state filter (NEW)
    # ---------------------------
    if state:
        # normalize like GPT: uppercase 2-letter
        sql += " AND UPPER(dm.state_code) = UPPER(%s)"
        params.append(state)

    # ---------------------------
    # OPTIONAL: property_type filter
    # ---------------------------
    if property_type:
        sql += " AND LOWER(fm.property_type_code) = LOWER(%s)"
        params.append(property_type)

    # ---------------------------
    # SORTING / LIMIT
    # ---------------------------
    sql += """
        ORDER BY 
            dm.market_name,
            fm.property_type_code,
            dt.reporting_period DESC
        LIMIT %s
    """
    params.append(limit)

    # SQL EXECUTION
    rows = run_sql(sql, params)

    return {"count": len(rows), "rows": rows}
