# Streamlit Player Analytics Dashboard
# ----------------------------------
# What this app does
# - Connects to your Google Sheet (the one that uses the schema from your sample)
# - Cleans & normalizes types (dates, numbers, booleans)
# - Lets you filter by date, status, country, and quick search
# - Shows KPIs, trends, distributions, and a players table
# - Exposes CSV download of filtered data
#
# How to run
#   1) Save this file as `streamlit_app.py`.
#   2) In a terminal, run: `pip install streamlit pandas numpy altair gspread oauth2client` (gspread is optional if using CSV export).
#   3) Run: `streamlit run streamlit_app.py`.
#
# Connecting to Google Sheets (choose ONE of these)
#   A) EASIEST (CSV Export):
#      - Open your Google Sheet → Share → "Anyone with the link" (Viewer)
#      - Copy the spreadsheet URL (e.g., https://docs.google.com/spreadsheets/d/<SHEET_ID>/edit#gid=0)
#      - Paste that URL in the sidebar field here and optionally the worksheet name.
#      - This app will convert it to a CSV export URL and read it with pandas.
#
#   B) SERVICE ACCOUNT (for private sheets):
#      - Create a Google Service Account and download the JSON key.
#      - In Streamlit Cloud, add the JSON under `st.secrets["gcp_service_account"]`.
#      - Share your sheet with the service account email.
#      - Enter the Sheet ID and Worksheet name in the sidebar, toggle "Use gspread", and the app will load via Google API.
#
# Column expectations (minimum):
#   _id, username, email, status, demo_status, profile_status, createdAt, updatedAt,
#   country, dob, name, phone, state,
#   feed_spent_total, feed_won_total, feed_won_to_spent_ratio, wallet_balance,
#   contests_count_total, lineups_count_total, contests_participated,
#   activated_user, active_user,
#   referral_code, referral_count, signupSource,
#   usd_wallet_balance, usd_spent_total, usd_won_total, usd_deposit_total, usd_won_to_spent_ratio,
#   usd_withdraw_gross_total, usd_withdraw_refund_total, usd_withdraw_net_total, usd_net_total,
#   usd_spent_14d, usd_won_14d, usd_deposit_14d, usd_won_to_spent_ratio_14d,
#   usd_withdraw_gross_14d, usd_withdraw_refund_14d, usd_withdraw_net_14d, usd_net_14d
#
# Notes:
# - The app is resilient if some columns are missing; it will compute what it can and warn about the rest.
# - The "contests_participated" free-text list is not parsed into structured entries; we do count how many contest mentions per row.

import re
from datetime import datetime, date
from typing import List, Optional

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# -----------------------------
# Helpers: Data Loading
# -----------------------------
CSV_EXPORT_TMPL = "https://docs.google.com/spreadsheets/d/{sid}/gviz/tq?tqx=out:csv"


def _extract_sheet_id(url: str) -> Optional[str]:
    m = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", url)
    return m.group(1) if m else None


def _build_csv_url(sheet_url_or_id: str, worksheet_name: Optional[str] = None) -> str:
    """Accepts a full URL or just the Sheet ID. Optionally add `sheet=<worksheet>`."""
    sid = sheet_url_or_id
    if sheet_url_or_id.startswith("http"):
        sid = _extract_sheet_id(sheet_url_or_id) or sheet_url_or_id
    base = CSV_EXPORT_TMPL.format(sid=sid)
    if worksheet_name:
        # When a sheet name is provided, Google accepts `&sheet=<name>` in the export URL
        base += f"&sheet={worksheet_name}"
    return base


@st.cache_data(show_spinner=True)
def load_from_csv_export(sheet_url_or_id: str, worksheet_name: Optional[str] = None) -> pd.DataFrame:
    csv_url = _build_csv_url(sheet_url_or_id, worksheet_name)
    df = pd.read_csv(csv_url)
    return df


@st.cache_data(show_spinner=True)
def load_from_gspread(sheet_id: str, worksheet_name: str) -> pd.DataFrame:
    """Requires st.secrets["gcp_service_account"] to be present. Only use if needed."""
    try:
        import gspread
        from oauth2client.service_account import ServiceAccountCredentials
    except Exception as e:
        raise RuntimeError("gspread and oauth2client must be installed to use this mode.") from e

    scopes = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
        "https://www.googleapis.com/auth/drive.file",
    ]
    creds_dict = st.secrets.get("gcp_service_account", None)
    if not creds_dict:
        raise RuntimeError("Missing st.secrets['gcp_service_account'] for gspread mode.")

    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scopes)
    client = gspread.authorize(creds)
    sh = client.open_by_key(sheet_id)
    ws = sh.worksheet(worksheet_name)
    data = ws.get_all_records()
    return pd.DataFrame(data)


# -----------------------------
# Helpers: Cleaning & Types
# -----------------------------
NUMERIC_COLS = [
    "feed_spent_total", "feed_won_total", "feed_won_to_spent_ratio", "wallet_balance",
    "contests_count_total", "lineups_count_total",
    "usd_wallet_balance", "usd_spent_total", "usd_won_total", "usd_deposit_total",
    "usd_won_to_spent_ratio", "usd_withdraw_gross_total", "usd_withdraw_refund_total",
    "usd_withdraw_net_total", "usd_net_total", "usd_spent_14d", "usd_won_14d", "usd_deposit_14d",
    "usd_won_to_spent_ratio_14d", "usd_withdraw_gross_14d", "usd_withdraw_refund_14d",
    "usd_withdraw_net_14d", "usd_net_14d"
]

BOOL_COLS = ["activated_user", "active_user"]
DATE_COLS = ["createdAt", "updatedAt", "dob"]
TEXT_SEARCH_COLS = ["_id", "username", "email", "name", "referral_code", "phone", "state", "country", "signupSource"]


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    for col in DATE_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def to_bools(df: pd.DataFrame) -> pd.DataFrame:
    for col in BOOL_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower().map({
                "true": True, "yes": True, "1": True,
                "false": False, "no": False, "0": False
            }).fillna(False)
    return df


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    # contest list count (rough) from the free text column if present
    if "contests_participated" in df.columns:
        df["contests_list_count"] = (
            df["contests_participated"].fillna("").astype(str).apply(
                lambda s: 0 if s.strip()=="" else len([x for x in re.split(r",\s*", s) if x])
            )
        )

    # Compute ratios if missing
    if "usd_won_total" in df.columns and "usd_spent_total" in df.columns:
        df["computed_usd_won_to_spent_ratio"] = np.where(
            (df["usd_spent_total"].fillna(0) > 0),
            df["usd_won_total"].fillna(0) / df["usd_spent_total"].replace(0, np.nan),
            np.nan
        )

    # Banned flag heuristic
    def is_banned(row):
        s = str(row.get("status", "")).lower()
        p = str(row.get("profile_status", "")).lower()
        return ("banned" in s) or ("banned" in p)

    df["is_banned"] = df.apply(is_banned, axis=1)

    # Status simplified
    df["status_simple"] = df.get("status", "").astype(str).str.title().replace(
        {"On-Boarded": "Onboarded", "On-Boarding": "Onboarding"}
    )

    # Days since created (for freshness filters)
    if "createdAt" in df.columns:
        df["days_since_created"] = (pd.Timestamp.utcnow() - df["createdAt"]).dt.days

    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame()
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    df = parse_dates(df)
    df = to_numeric(df)
    df = to_bools(df)
    df = enrich(df)
    return df


# -----------------------------
# UI: Sidebar Controls
# -----------------------------
st.set_page_config(page_title="Player Analytics Dashboard", page_icon="📊", layout="wide")

st.sidebar.header("🔗 Data Source")
mode = st.sidebar.radio("Load method", ["CSV Export (public)", "gspread (private)"])

sheet_url_or_id = st.sidebar.text_input("Google Sheet URL or ID", help="Paste the full Google Sheet URL or just the Sheet ID.")
worksheet_name = st.sidebar.text_input("Worksheet name (optional)")

use_cache_bust = st.sidebar.checkbox("Force refresh (bust cache)", value=False)

load_btn = st.sidebar.button("Load Data", type="primary")

if "_cache_bust" not in st.session_state:
    st.session_state._cache_bust = 0
if use_cache_bust:
    st.session_state._cache_bust += 1

# DataFrame placeholder
df: pd.DataFrame = pd.DataFrame()
load_error: Optional[str] = None

if load_btn and sheet_url_or_id:
    try:
        if mode.startswith("CSV"):
            df = load_from_csv_export(
                sheet_url_or_id + f"?cache_bust={st.session_state._cache_bust}",
                worksheet_name or None
            )
        else:
            sid = _extract_sheet_id(sheet_url_or_id) or sheet_url_or_id
            if not worksheet_name:
                raise RuntimeError("Worksheet name is required for gspread mode.")
            df = load_from_gspread(sid, worksheet_name)
        df = clean(df)
    except Exception as e:
        load_error = str(e)

st.title("📊 Player Analytics Dashboard")

tab_main, tab_beta, tab_racing = st.tabs(["All Users", "Beta Testing", "Racing Dudes"])

with st.expander("How to connect to your sheet", expanded=not load_btn):
    st.markdown(
        """
        **Option A — CSV Export (no code / easiest)**  
        1. Share your Sheet with "Anyone with the link" (Viewer).  
        2. Paste the full URL above and click **Load Data**.  
        3. If you have multiple tabs, provide the *Worksheet name* too.  

        **Option B — Private via gspread**  
        1. Add your Service Account JSON to `st.secrets["gcp_service_account"]`.  
        2. Share the Sheet with the service account email.  
        3. Enter Sheet ID + Worksheet name, select **gspread (private)**, then **Load Data**.
        """
    )

if load_error:
    st.error(f"Failed to load data: {load_error}")

if df.empty and load_btn:
    st.warning("No data loaded. Double-check the URL/ID and the worksheet name or sharing settings.")
with tab_main:
    if not df.empty:
        # -----------------------------
        # Sidebar filters
        # -----------------------------
        st.sidebar.header("🔎 Filters")
    
        # Date range filter (createdAt) — robust to mixed types/empty strings/NaN
        if "createdAt" in df.columns:
            created_series = pd.to_datetime(df["createdAt"], errors="coerce")
            if created_series.notna().any():
                min_date = created_series.min()
                max_date = created_series.max()
                # Coerce to date for widget (avoid min/max on mixed dtypes)
                start_date_default = min_date.date()
                end_date_default = max_date.date()
                start_date, end_date = st.sidebar.date_input(
                    "Created date range",
                    value=(start_date_default, end_date_default),
                    min_value=start_date_default,
                    max_value=end_date_default,
                )
            else:
                start_date, end_date = None, None
        else:
            start_date, end_date = None, None
    
        # Status filter
        statuses = sorted(df["status_simple"].dropna().unique().tolist()) if "status_simple" in df.columns else []
        status_sel = st.sidebar.multiselect("Status", statuses, default=statuses)
    
        # Country filter
        countries = sorted(df["country"].dropna().unique().tolist()) if "country" in df.columns else []
        country_sel = st.sidebar.multiselect("Country", countries, default=countries)
    
        # Quick search
        quick_q = st.sidebar.text_input("Quick search (username, email, name, referral)")
    
        # Minimum contests
        min_contests = st.sidebar.number_input("Min contests count (total)", min_value=0, value=0, step=1)
    
        # Apply filters
        fdf = df.copy()
    
        # Date filter
        if start_date and end_date and "createdAt" in fdf.columns:
            created_parsed = pd.to_datetime(fdf["createdAt"], errors="coerce")
            left = pd.to_datetime(start_date)
            right = pd.to_datetime(end_date) + pd.Timedelta(days=1)  # inclusive end
            mask_date = created_parsed.between(left, right, inclusive="left").fillna(False)
            fdf = fdf[mask_date]
    
        # Status filter
        if status_sel and "status_simple" in fdf.columns:
            fdf = fdf[fdf["status_simple"].isin(status_sel)]
    
        # Country filter
        if country_sel and "country" in fdf.columns:
            fdf = fdf[fdf["country"].isin(country_sel)]
    
        # Min contests
        if "contests_count_total" in fdf.columns:
            fdf = fdf[fdf["contests_count_total"].fillna(0) >= min_contests]
    
        # Quick search across text fields
        if quick_q:
            qq = quick_q.lower().strip()
            text_cols = [c for c in TEXT_SEARCH_COLS if c in fdf.columns]
            if text_cols:
                fdf = fdf[fdf[text_cols].astype(str).apply(lambda row: any(qq in str(v).lower() for v in row), axis=1)]
    
        # ------- Hard filters (expanded email exclusions + US-only) -------
        if "email" in fdf.columns:
            # Remove any email containing these keywords (case-insensitive)
            exclude_keywords = ["test", "prod", "yopmail", "rawleigh"]
            pattern = "|".join(exclude_keywords)
            fdf = fdf[~fdf["email"].astype(str).str.contains(pattern, case=False, na=False)]
    
        # Only include users where country == "United States" (case-insensitive)
        if "country" in fdf.columns:
            fdf = fdf[fdf["country"].astype(str).str.strip().str.lower() == "united states"]
    
    
        # -----------------------------
        # Key Metrics
        # -----------------------------
        st.subheader("Key Metrics")
        
        # Total players and verification summary
        total_players = len(fdf)
        
        # Normalize profile_status safely
        if "profile_status" in fdf.columns:
            ps_norm = (
                fdf["profile_status"]
                .astype(str)
                .str.strip()
                .str.lower()
            )
        else:
            ps_norm = pd.Series([], dtype="object")
        
        # Buckets
        unverified_set = {"unverified", "unverified np", "unverified p"}  # Only pure unverified
        kyc_set = {"grade-i", "grade-ii", "grade-iii"}
        
        unverified_count = int(ps_norm.isin(unverified_set).sum()) if not ps_norm.empty else 0
        kyc_verified_count = int(ps_norm.isin(kyc_set).sum()) if not ps_norm.empty else 0
        
        # Robust banned count
        banned_count = 0
        if not ps_norm.empty:
            banned_count = int(ps_norm.str.contains(r"\bbanned\b", na=False).sum())
        if banned_count == 0 and "is_banned" in fdf.columns:
            banned_count = int(fdf["is_banned"].fillna(False).astype(bool).sum())
        
        # --- TOTAL USD METRICS ---
        usd_spent = float(fdf.get("usd_spent_total", 0).fillna(0).sum())
        usd_won = float(fdf.get("usd_won_total", 0).fillna(0).sum())
        usd_deposit = float(fdf.get("usd_deposit_total", 0).fillna(0).sum())
        usd_net = float(fdf.get("usd_net_total", 0).fillna(0).sum())
        usd_wallet = float(fdf.get("usd_wallet_balance", 0).fillna(0).sum())
        usd_withdraw = float(fdf.get("usd_withdraw_net_total", 0).fillna(0).sum())
        
        # --- LAST 14 DAYS METRICS ---
        cutoff_14d = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=14)
        recent_df = fdf.copy()
        if "createdAt" in recent_df.columns:
            created_parsed = pd.to_datetime(recent_df["createdAt"], errors="coerce")
            recent_df = recent_df[created_parsed >= cutoff_14d]
        
        usd_spent_14d = float(recent_df.get("usd_spent_total", 0).fillna(0).sum())
        usd_won_14d = float(recent_df.get("usd_won_total", 0).fillna(0).sum())
        usd_deposit_14d = float(recent_df.get("usd_deposit_total", 0).fillna(0).sum())
        usd_net_14d = float(recent_df.get("usd_net_total", 0).fillna(0).sum())
        usd_withdraw_14d = float(recent_df.get("usd_withdraw_net_total", 0).fillna(0).sum())
        
        # --- ROW 1: General User Metrics ---
        kpi_row1 = st.columns(4)
        kpi_row1[0].metric("Players", f"{total_players:,}")
        kpi_row1[1].metric("Unverified", f"{unverified_count:,}")
        kpi_row1[2].metric("KYC Verified", f"{kyc_verified_count:,}")
        kpi_row1[3].metric("Banned", f"{banned_count:,}")
        
        # --- ROW 2: USD Totals (All-Time) ---
        st.markdown("#### 💵 All-Time USD Metrics")
        usd_row = st.columns(6)
        usd_row[0].metric("USD Deposits ($)", f"${usd_deposit:,.0f}")
        usd_row[1].metric("USD Withdrawals ($)", f"${usd_withdraw:,.0f}")
        usd_row[2].metric("USD Spent ($)", f"${usd_spent:,.0f}")
        usd_row[3].metric("USD Won ($)", f"${usd_won:,.0f}")
        usd_row[4].metric("USD Net ($)", f"${usd_net:,.0f}")
        usd_row[5].metric("USD in Wallets ($)", f"${usd_wallet:,.0f}")
        
        # ROI Metric under USD rows
        if usd_spent > 0:
            roi = usd_won / usd_spent
            st.metric("Pooled ROI (won/spent)", f"{roi:,.2f}×")
        else:
            st.metric("Pooled ROI (won/spent)", "–")
        
        st.markdown("---")
        # --- ROW 3: USD (Last 14 Days) ---
        st.markdown("#### 📅 Last 14 Days USD Metrics")
        usd_14d_row = st.columns(5)
        usd_14d_row[0].metric("USD Deposits (14d)", f"${usd_deposit_14d:,.0f}")
        usd_14d_row[1].metric("USD Withdrawals (14d)", f"${usd_withdraw_14d:,.0f}")
        usd_14d_row[2].metric("USD Spent (14d)", f"${usd_spent_14d:,.0f}")
        usd_14d_row[3].metric("USD Won (14d)", f"${usd_won_14d:,.0f}")
        usd_14d_row[4].metric("USD Net (14d)", f"${usd_net_14d:,.0f}")
        
    
    
        # -----------------------------
        # Charts
        # -----------------------------
        chart_cols = st.columns(2)
    
        # Players by Status
        if "status_simple" in fdf.columns:
            status_counts = (
                fdf.groupby("status_simple").size().reset_index(name="count")
            )
            ch1 = alt.Chart(status_counts).mark_bar().encode(
                x=alt.X("status_simple:N", title="Status"),
                y=alt.Y("count:Q", title="Players"),
                tooltip=["status_simple", "count"]
            ).properties(height=320, title="Players by Status")
            chart_cols[0].altair_chart(ch1, use_container_width=True)
    
        # New Players by Month & Last 14 Days
        if "createdAt" in fdf.columns:
            created_parsed = pd.to_datetime(fdf["createdAt"], errors="coerce")
            if created_parsed.notna().any():
                # ---- Monthly new players ----
                monthly = (
                    pd.DataFrame({"month": created_parsed.dt.to_period("M").dt.to_timestamp()})
                    .dropna()
                    .groupby("month").size().reset_index(name="new_players")
                    .sort_values("month")
                )
        
                ch_month = alt.Chart(monthly).mark_bar().encode(
                    x=alt.X("month:T", title="Month"),
                    y=alt.Y("new_players:Q", title="New players"),
                    tooltip=[
                        alt.Tooltip("month:T", title="Month"),
                        alt.Tooltip("new_players:Q", title="New players")
                    ]
                ).properties(height=320, title="New Players by Month")
        
                    # ---- New players in last 14 days (tz-safe, line chart) ----
            now_naive = pd.Timestamp.utcnow().tz_localize(None)
            cutoff_date = (now_naive - pd.Timedelta(days=14)).normalize()
            
            recent_mask = created_parsed >= cutoff_date
            recent = created_parsed[recent_mask]
            
            if recent.notna().any():
                recent_by_day = (
                    pd.DataFrame({"day": recent.dt.floor("D")})
                    .groupby("day").size().reset_index(name="new_players")
                    .sort_values("day")
                )
            
                ch_recent = (
                    alt.Chart(recent_by_day)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X("day:T", title="Date"),
                        y=alt.Y("new_players:Q", title="New players"),
                        tooltip=[
                            alt.Tooltip("day:T", title="Date"),
                            alt.Tooltip("new_players:Q", title="New players"),
                        ],
                    )
                    .properties(height=320, title="New Players (Last 14 Days)")
                )
            
                chart_cols = st.columns(2)
                chart_cols[0].altair_chart(ch_month, use_container_width=True)
                chart_cols[1].altair_chart(ch_recent, use_container_width=True)
            else:
                chart_cols = st.columns(2)
                chart_cols[0].altair_chart(ch_month, use_container_width=True)
                chart_cols[1].write("No new players in the last 14 days.")
    
        # State distribution (US-only view)
        if "state" in fdf.columns:
            st.markdown("### Geography (States)")
    
            # Clean up state values a bit
            state_series = (
                fdf["state"]
                .astype(str)
                .str.strip()
                .replace({"": np.nan, "nan": np.nan, "None": np.nan})
            )
    
            top_states = (
                state_series.dropna()
                .to_frame("state")
                .groupby("state").size()
                .reset_index(name="players")
                .sort_values("players", ascending=False)
            )
    
            if not top_states.empty:
                ch_state = alt.Chart(top_states).mark_bar().encode(
                    x=alt.X("players:Q", title="Players"),
                    y=alt.Y("state:N", sort='-x', title="State"),
                    tooltip=["state", "players"]
                ).properties(height=380, title="Players by State")
                st.altair_chart(ch_state, use_container_width=True)
            else:
                st.info("No state data available to display.")
    
        # Finance distributions
        if any(col in fdf.columns for col in ["usd_spent_total", "usd_won_total", "computed_usd_won_to_spent_ratio", "usd_won_to_spent_ratio"]):
            st.markdown("### Finance Distributions")
            fin_cols = st.columns(2)
    
            if "usd_spent_total" in fdf.columns:
                dens = fdf[["usd_spent_total"]].dropna()
                if not dens.empty:
                    ch4 = alt.Chart(dens).mark_bar().encode(
                        x=alt.X("usd_spent_total:Q", bin=alt.Bin(maxbins=40), title="USD Spent (Total)"),
                        y=alt.Y("count():Q", title="Players"),
                        tooltip=["count()"]
                    ).properties(height=320, title="Distribution: USD Spent")
                    fin_cols[0].altair_chart(ch4, use_container_width=True)
    
            ratio_col = "usd_won_to_spent_ratio" if "usd_won_to_spent_ratio" in fdf.columns else "computed_usd_won_to_spent_ratio"
            if ratio_col in fdf.columns:
                densr = fdf[[ratio_col]].dropna()
                if not densr.empty:
                    ch5 = alt.Chart(densr).mark_bar().encode(
                        x=alt.X(f"{ratio_col}:Q", bin=alt.Bin(maxbins=40), title="Won/Spent Ratio (ROI×)"),
                        y=alt.Y("count():Q", title="Players"),
                        tooltip=["count()"]
                    ).properties(height=320, title="Distribution: ROI (Won/Spent)")
                    fin_cols[1].altair_chart(ch5, use_container_width=True)
    
        st.markdown("---")
        # -----------------------------
        # Players Table
        # -----------------------------
        st.markdown("### Players (filtered)")
        
        # Only include these specific columns
        show_cols_default = [
            "username",
            "name",
            "email",
            "contests_count_total",
            "lineups_count_total",
            "contests_participated",
            "referral_code",
            "referral_count",
            "usd_wallet_balance",
            "usd_spent_total",
            "usd_won_total",
            "usd_deposit_total",
            "usd_withdraw_net_total",
        ]
        
        # Keep only columns that actually exist in the dataset
        table_cols = [c for c in show_cols_default if c in fdf.columns]
        
        # Sort only by keys that are present in the *displayed* columns
        preferred_sort_order = ["usd_spent_total", "usd_won_total"]  # drop "createdAt"
        sort_keys = [c for c in preferred_sort_order if c in table_cols]
        
        df_display = fdf[table_cols].copy()
        if sort_keys:
            df_display = df_display.sort_values(
                by=sort_keys,
                ascending=[False] * len(sort_keys)
            )
        
        st.dataframe(
            df_display.reset_index(drop=True),
            use_container_width=True,
            hide_index=True,
        )
        
        # CSV download for the filtered dataset
        csv = df_display.to_csv(index=False).encode("utf-8")
        st.download_button(
        "Download filtered CSV",
        csv,
        file_name="players_filtered.csv",
        mime="text/csv",
        key="download_filtered_players"
    )
    
    
        # -----------------------------
        # Dropped-off at Signup (Unverified ONLY)
        # -----------------------------
        st.markdown("### Users that dropped off at signup")
        
        # Choose data source to avoid accidentally hiding real unverified users
        source_choice = st.radio(
            "Show from:",
            ["All data (ignore page filters)", "Use page filters"],
            horizontal=True,
            help="If unverified users are missing, they may have been filtered out earlier. "
                 "Use 'All data' to ignore page filters for this list (but still exclude junk emails)."
        )
        
        # Pick the frame: df = all cleaned data, fdf = filtered view
        base_frame = df if source_choice.startswith("All data") else fdf
        
        # Always exclude junk emails per your rule
        exclude_keywords = ["test", "prod", "yopmail", "rawleigh"]
        email_pattern = "|".join(exclude_keywords)
        if "email" in base_frame.columns:
            base_frame = base_frame[~base_frame["email"].astype(str).str.contains(email_pattern, case=False, na=False)]
        
        # Prefer 'player_status' then fall back to 'profile_status'
        status_source_col = (
            "player_status" if "player_status" in base_frame.columns
            else ("profile_status" if "profile_status" in base_frame.columns else None)
        )
        
        if status_source_col is None:
            st.info("No player/profile status column found.")
        else:
            ps_norm = (
                base_frame[status_source_col]
                .astype(str)
                .str.strip()
                .str.lower()
            )
        
            # EXACTLY "unverified" (not "unverified np" or "unverified p")
            mask_unverified_exact = ps_norm.eq("unverified")
        
            dropped = (
                base_frame.loc[mask_unverified_exact, [c for c in ["username", "email"] if c in base_frame.columns]]
                .copy()
            )
        
            # Small summary comparing against fully filtered view (optional)
            summary_note = ""
            if source_choice.startswith("All data") and status_source_col is not None:
                # Count how many unverified remain after page filters too (fdf)
                if not fdf.empty and status_source_col in fdf.columns:
                    ps_f = fdf[status_source_col].astype(str).str.strip().str.lower()
                    filtered_count = int(ps_f.eq("unverified").sum())
                    summary_note = f" (Visible with page filters: **{filtered_count:,}**)"
            
            if dropped.empty:
                st.info("No users with status exactly 'unverified' found." + summary_note)
            else:
                st.caption(f"Found **{len(dropped):,}** unverified user(s){summary_note}.")
                dropped["tag"] = "Users that dropped off at signup"
                st.dataframe(dropped, use_container_width=True, hide_index=True)
        
                # CSV download
                st.download_button(
                    "Download dropped-off users CSV",
                    dropped.to_csv(index=False).encode("utf-8"),
                    file_name="dropped_off_users.csv",
                    mime="text/csv",
                )
    
    
        # CSV download for the filtered dataset
        csv = fdf.to_csv(index=False).encode("utf-8")
        st.download_button("Download filtered CSV", csv, file_name="players_filtered.csv", mime="text/csv")
    
        # -----------------------------
        # Data Quality Checks
        # -----------------------------
        st.markdown("---")
        st.markdown("### Data Quality Checks")
        dq_msgs: List[str] = []
    
        # Missing key fields
        for key_col in ["_id", "email", "username"]:
            if key_col in fdf.columns:
                missing = int(fdf[key_col].isna().sum() + (fdf[key_col].astype(str).str.strip()=="").sum())
                if missing:
                    dq_msgs.append(f"• {missing} rows missing `{key_col}`")
    
        # Date issues
        for dcol in ["createdAt", "updatedAt"]:
            if dcol in fdf.columns:
                bad = int(fdf[dcol].isna().sum())
                if bad:
                    dq_msgs.append(f"• {bad} rows with invalid `{dcol}`")
    
        # Ratio sanity (> 20× might be suspicious depending on your domain)
        ratio_col = "usd_won_to_spent_ratio" if "usd_won_to_spent_ratio" in fdf.columns else "computed_usd_won_to_spent_ratio"
        if ratio_col in fdf.columns:
            extreme = int((fdf[ratio_col] > 20).sum())
            if extreme:
                dq_msgs.append(f"• {extreme} rows with very high ROI (>20×) — verify correctness")
    
        if dq_msgs:
            st.warning("\n".join(dq_msgs))
        else:
            st.success("No major data quality warnings detected.")
    
    else:
        st.info("Load your Google Sheet from the sidebar to get started.")


with tab_beta:
    st.subheader("🧪 Beta Testing Metrics")

    # Filter users whose signupSource == 'betatesting'
    beta_df = df[df["signupSource"].astype(str).str.lower() == "betatesting"].copy()

    if beta_df.empty:
        st.info("No users found with signupSource = 'betatesting'.")
    else:
        st.caption(f"Found **{len(beta_df):,}** beta testing user(s).")

        # Reuse your metrics logic for this filtered DataFrame
        fdf = beta_df  # Reuse naming for simplicity

        # --- (Reuse your Key Metrics Section here) ---
        # You can literally copy the same “Key Metrics” block you already have, 
        # but replace all references to the main fdf with this one.
        
        # Example:
        total_players = len(fdf)
        ps_norm = fdf["profile_status"].astype(str).str.strip().str.lower() if "profile_status" in fdf.columns else pd.Series([], dtype="object")

        unverified_set = {"unverified"}
        kyc_set = {"grade-i", "grade-ii", "grade-iii"}

        unverified_count = int(ps_norm.isin(unverified_set).sum())
        kyc_verified_count = int(ps_norm.isin(kyc_set).sum())
        banned_count = int(ps_norm.str.contains("banned", na=False).sum())

        usd_spent = fdf["usd_spent_total"].fillna(0).sum()
        usd_won = fdf["usd_won_total"].fillna(0).sum()
        usd_deposit = fdf["usd_deposit_total"].fillna(0).sum()
        usd_withdraw = fdf["usd_withdraw_net_total"].fillna(0).sum()
        usd_net = fdf["usd_net_total"].fillna(0).sum()
        usd_wallet = fdf["usd_wallet_balance"].fillna(0).sum()

        st.markdown("#### 💵 Beta Users — All-Time USD Metrics")
        cols = st.columns(6)
        cols[0].metric("USD Deposits ($)", f"${usd_deposit:,.0f}")
        cols[1].metric("USD Withdrawals ($)", f"${usd_withdraw:,.0f}")
        cols[2].metric("USD Spent ($)", f"${usd_spent:,.0f}")
        cols[3].metric("USD Won ($)", f"${usd_won:,.0f}")
        cols[4].metric("USD Net ($)", f"${usd_net:,.0f}")
        cols[5].metric("USD in Wallets ($)", f"${usd_wallet:,.0f}")

        # ROI metric
        if usd_spent > 0:
            st.metric("Pooled ROI (won/spent)", f"{usd_won / usd_spent:,.2f}×")
        else:
            st.metric("Pooled ROI (won/spent)", "–")

        st.markdown("---")

        # Charts reuse (Monthly + 14d)
        if "createdAt" in fdf.columns:
            created_parsed = pd.to_datetime(fdf["createdAt"], errors="coerce")
            monthly = (
                pd.DataFrame({"month": created_parsed.dt.to_period("M").dt.to_timestamp()})
                .dropna()
                .groupby("month").size().reset_index(name="new_players")
                .sort_values("month")
            )

            ch_month = alt.Chart(monthly).mark_bar().encode(
                x=alt.X("month:T", title="Month"),
                y=alt.Y("new_players:Q", title="New players"),
            ).properties(height=300, title="New Beta Players by Month")

            now_naive = pd.Timestamp.utcnow().tz_localize(None)
            cutoff_date = (now_naive - pd.Timedelta(days=14)).normalize()
            recent_mask = created_parsed >= cutoff_date
            recent = created_parsed[recent_mask]

            if recent.notna().any():
                recent_by_day = (
                    pd.DataFrame({"day": recent.dt.floor("D")})
                    .groupby("day").size().reset_index(name="new_players")
                    .sort_values("day")
                )

                ch_recent = (
                    alt.Chart(recent_by_day)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X("day:T", title="Date"),
                        y=alt.Y("new_players:Q", title="New players"),
                    )
                    .properties(height=300, title="New Beta Players (Last 14 Days)")
                )

                ch_cols = st.columns(2)
                ch_cols[0].altair_chart(ch_month, use_container_width=True)
                ch_cols[1].altair_chart(ch_recent, use_container_width=True)

with tab_racing:
    st.subheader("🏇 Racing Dudes Metrics")

    # --- Safely check for campaign_tags column ---
    if "campaign_tags" not in df.columns:
        st.warning("No `campaign_tags` column found in the dataset.")
    else:
        # Normalize the campaign_tags column to string for matching
        fdf_racing = df.copy()
        fdf_racing["campaign_tags"] = fdf_racing["campaign_tags"].astype(str).str.lower()

        # Filter for any users that contain 'racing_dudes'
        racing_df = fdf_racing[fdf_racing["campaign_tags"].str.contains("racing_dudes", na=False)].copy()

        # Further filter out junk/test emails
        if "email" in racing_df.columns:
            exclude_keywords = ["test", "yopmail", "ralls", "plutus21"]
            pattern = "|".join(exclude_keywords)
            racing_df = racing_df[~racing_df["email"].astype(str).str.contains(pattern, case=False, na=False)]

        if racing_df.empty:
            st.info("No valid users found with campaign_tags containing 'racing_dudes'.")
        else:
            st.caption(f"Found **{len(racing_df):,}** Racing Dudes user(s).")

            # --- Basic User Stats ---
            total_players = len(racing_df)
            ps_norm = racing_df["profile_status"].astype(str).str.strip().str.lower() if "profile_status" in racing_df.columns else pd.Series([], dtype="object")

            unverified_set = {"unverified"}
            kyc_set = {"grade-i", "grade-ii", "grade-iii"}

            unverified_count = int(ps_norm.isin(unverified_set).sum())
            kyc_verified_count = int(ps_norm.isin(kyc_set).sum())
            banned_count = int(ps_norm.str.contains("banned", na=False).sum())

            # Display simple player verification stats
            kpi_row = st.columns(4)
            kpi_row[0].metric("Players", f"{total_players:,}")
            kpi_row[1].metric("Unverified", f"{unverified_count:,}")
            kpi_row[2].metric("KYC Verified", f"{kyc_verified_count:,}")
            kpi_row[3].metric("Banned", f"{banned_count:,}")

            st.markdown("---")

            # --- Charts for this segment ---
            if "createdAt" in racing_df.columns:
                created_parsed = pd.to_datetime(racing_df["createdAt"], errors="coerce")
                monthly = (
                    pd.DataFrame({"month": created_parsed.dt.to_period("M").dt.to_timestamp()})
                    .dropna()
                    .groupby("month").size().reset_index(name="new_players")
                    .sort_values("month")
                )

                ch_month = alt.Chart(monthly).mark_bar().encode(
                    x=alt.X("month:T", title="Month"),
                    y=alt.Y("new_players:Q", title="New players"),
                ).properties(height=300, title="New Racing Dudes Players by Month")

                now_naive = pd.Timestamp.utcnow().tz_localize(None)
                cutoff_date = (now_naive - pd.Timedelta(days=14)).normalize()
                recent_mask = created_parsed >= cutoff_date
                recent = created_parsed[recent_mask]

                if recent.notna().any():
                    recent_by_day = (
                        pd.DataFrame({"day": recent.dt.floor("D")})
                        .groupby("day").size().reset_index(name="new_players")
                        .sort_values("day")
                    )

                    ch_recent = (
                        alt.Chart(recent_by_day)
                        .mark_line(point=True)
                        .encode(
                            x=alt.X("day:T", title="Date"),
                            y=alt.Y("new_players:Q", title="New players"),
                        )
                        .properties(height=300, title="New Racing Dudes Players (Last 14 Days)")
                    )

                    ch_cols = st.columns(2)
                    ch_cols[0].altair_chart(ch_month, use_container_width=True)
                    ch_cols[1].altair_chart(ch_recent, use_container_width=True)

            st.markdown("---")

            # --- Table for these users ---
            st.markdown("### Racing Dudes Players")
            show_cols = [
                "username", "name", "email", "country", "profile_status",
                "createdAt", "contests_count_total", "usd_wallet_balance"
            ]
            show_cols = [c for c in show_cols if c in racing_df.columns]
            st.dataframe(racing_df[show_cols], use_container_width=True, hide_index=True)

            # CSV export
            csv = racing_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Racing Dudes CSV",
                csv,
                file_name="racing_dudes_players.csv",
                mime="text/csv"
            )
