# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import re
from difflib import get_close_matches
from datetime import datetime
from pathlib import Path
from typing import Optional, List

# ===============================
# App config
# ===============================
st.set_page_config(page_title="Dengue Allocation", page_icon="🏥", layout="wide")
st.title("🏥 Integrated Hospital Dengue Patient Allocation System (DSCC Region)")

# ===============================
# Fixed hospital list
# ===============================
HOSPITALS_UI = [
    "Dhaka Medical College Hospital",
    "SSMC & Mitford Hospital",
    "Bangladesh Shishu Hospital & Institute",
    "Shaheed Suhrawardy Medical College hospital",
    "Bangabandhu Shiekh Mujib Medical University",
    "Police Hospital, Rajarbagh",
    "Mugda Medical College",
    "Bangladesh Medical College Hospital",
    "Holy Family Red Cresent Hospital",
    "BIRDEM Hospital",
    "Ibn Sina Hospital",
    "Square Hospital",
    "Samorita Hospital",
    "Central Hospital Dhanmondi",
    "Lab Aid Hospital",
    "Green Life Medical Hospital",
    "Sirajul Islam Medical College Hospital",
    "Ad-Din Medical College Hospital",
]

# ===============================
# Styles (glassmorphism + chips)
# ===============================
st.markdown("""
<style>
:root{
  --bg:#0b1220; --card:#0f172a; --muted:#94a3b8;
  --ring:#22d3ee; --good:#10b981; --warn:#f59e0b; --bad:#ef4444; --info:#3b82f6;
}
html, body, [data-testid="stAppViewContainer"]{background:linear-gradient(135deg,#0b1220 0%,#0b1220 40%,#111827 100%) !important;}
.card{border-radius:18px;padding:18px 20px;background:rgba(255,255,255,0.06);backdrop-filter:blur(8px);
      border:1px solid rgba(255,255,255,0.08); box-shadow:0 10px 30px rgba(0,0,0,.25);}
.grid{display:grid; gap:14px;}
.grid-4{grid-template-columns:repeat(4,minmax(0,1fr));}
.kpi{font-weight:800;font-size:2rem;line-height:1;margin:0;}
.kpi-label{margin:2px 0 0 0;color:var(--muted);font-size:.9rem;}
.ribbon{display:inline-flex;align-items:center;gap:.6rem;margin-top:8px}
.badge{padding:6px 12px;border-radius:9999px;font-weight:700;color:#fff;display:inline-flex;align-items:center;gap:.4rem}
.badge.red{background:linear-gradient(135deg,#ef4444,#b91c1c)}
.badge.amber{background:linear-gradient(135deg,#f59e0b,#b45309)}
.badge.green{background:linear-gradient(135deg,#10b981,#047857)}
.badge.blue{background:linear-gradient(135deg,#3b82f6,#1d4ed8)}
.pill{padding:8px 14px;border-radius:9999px;background:rgba(255,255,255,.09);border:1px solid rgba(255,255,255,.12);
      font-weight:700;color:#e5e7eb;display:inline-flex;align-items:center}
.arrow{width:38px;height:38px;border-radius:10px;background:rgba(255,255,255,.12);display:grid;place-items:center;
       border:1px solid rgba(255,255,255,.16);margin:0 8px}
.sep{height:1px;background:rgba(255,255,255,.08);margin:12px 0}
.banner{padding:10px 14px;border-radius:12px;display:inline-flex;align-items:center;gap:.6rem;font-weight:700}
.banner.ok{background:rgba(16,185,129,.12); color:#a7f3d0; border:1px solid rgba(16,185,129,.25)}
.banner.warn{background:rgba(245,158,11,.12); color:#fde68a; border:1px solid rgba(245,158,11,.25)}
.small{color:var(--muted);font-size:.9rem}
.route{display:flex;align-items:center;flex-wrap:wrap;gap:6px}
.ticket{display:grid;grid-template-columns:1.2fr .8fr;gap:16px}
.codebox{background:#0b1220;border:1px dashed rgba(255,255,255,.15);border-radius:12px;padding:10px}
</style>
""", unsafe_allow_html=True)

def severity_badge(sev:str)->str:
    color = {"Mild":"green","Moderate":"amber","Severe":"red","Very Severe":"red"}.get(sev,"blue")
    return f'<span class="badge {color}">{sev}</span>'

def resource_badge(res:str)->str:
    color = "red" if res=="ICU" else "blue"
    return f'<span class="badge {color}">{res}</span>'

def availability_badge(txt:str)->str:
    if txt == "Yes": color = "green"
    elif "No vacancy" in txt: color = "amber"
    else: color = "blue"
    return f'<span class="badge {color}">{txt}</span>'

def sev_percent(sev:str)->int:
    return {"Mild":25,"Moderate":50,"Severe":75,"Very Severe":100}.get(sev,50)

# ===============================
# Utilities
# ===============================
def ensure_df(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if isinstance(df, pd.DataFrame):
        df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
        df.columns = [str(c).strip() for c in df.columns]
        return df
    return None

def autodetect(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    if not isinstance(df, pd.DataFrame) or df.empty: return None
    for patt in candidates:
        for c in df.columns:
            if patt.lower() in str(c).lower():
                return c
    return None

def parse_date_series(s: pd.Series) -> pd.Series:
    def _one(x):
        if isinstance(x, (pd.Timestamp, datetime)): return pd.to_datetime(x).date()
        try: return pd.to_datetime(x, errors="coerce").date()
        except: return pd.NaT
    return s.apply(_one)

def build_distance_matrix(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy(); df.columns = [str(c).strip() for c in df.columns]
    if df.shape[1] > 2:
        df = df.set_index(df.columns[0])
        for c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
        return df.combine_first(df.T)
    raise ValueError("Could not interpret Location matrix format.")

STOPWORDS = {"hospital","medical","college","institute","university","center","centre","clinic","and"}
def norm_key(s: str) -> str:
    s = str(s).lower().strip().replace("&"," and ")
    s = re.sub(r"[^a-z0-9\s]"," ", s)
    tokens = [t for t in s.split() if t and t not in STOPWORDS]
    return "".join(tokens)

def build_name_maps(availability, dist_mat, ui_list):
    avail_names = sorted(set(availability.index.get_level_values(0).tolist()))
    dm_names = sorted(set(map(str, dist_mat.index.tolist())) | set(map(str, dist_mat.columns.tolist())))
    ui_names = list(ui_list)

    avail_by_key = {norm_key(a): a for a in avail_names}
    dm_by_key    = {norm_key(d): d for d in dm_names}

    dm_to_av = {}
    for d in dm_names:
        kd = norm_key(d)
        if kd in avail_by_key: dm_to_av[d] = avail_by_key[kd]
        else:
            m = get_close_matches(kd, list(avail_by_key.keys()), n=1, cutoff=0.6)
            dm_to_av[d] = avail_by_key[m[0]] if m else None

    ui_to_dm, ui_to_av = {}, {}
    for u in ui_names:
        ku = norm_key(u)
        if ku in dm_by_key: ui_to_dm[u] = dm_by_key[ku]
        else:
            m = get_close_matches(ku, list(dm_by_key.keys()), n=1, cutoff=0.6)
            ui_to_dm[u] = dm_by_key[m[0]] if m else None

        if ku in avail_by_key: ui_to_av[u] = avail_by_key[ku]
        else:
            m2 = get_close_matches(ku, list(avail_by_key.keys()), n=1, cutoff=0.6)
            ui_to_av[u] = avail_by_key[m2[0]] if m2 else None
    return dm_to_av, ui_to_dm, ui_to_av

# ===============================
# Severity logic
# ===============================
def compute_platelet_score(platelet: int) -> int:
    if platelet >= 150_000: return 0
    if platelet >= 100_000: return 1
    if platelet >= 50_000:  return 2
    if platelet >= 20_000:  return 3
    return 4

def compute_severity_score(age: int, ns1: int, igm: int, igg: int, platelet: int) -> tuple[int, int]:
    p_score = compute_platelet_score(platelet)
    age_weight = 1 if (age < 15 or age > 60) else 0
    secondary = 1 if (igg == 1 and (ns1 == 1 or igm == 1)) else 0
    severity_score = min(4, round(p_score + age_weight + secondary))
    return p_score, severity_score

def verdict_from_score(score: int) -> str:
    if score >= 3: return "Very Severe"
    if score == 2: return "Severe"
    if score == 1: return "Moderate"
    return "Mild"

def required_resource(severity: str) -> str:
    return "ICU" if severity in ("Severe", "Very Severe") else "General Bed"

# ===============================
# Load repo files
# ===============================
pred_file_path = Path("Predicted dataset AIO.xlsx")
loc_file_path  = Path("Location matrix.xlsx")
if not pred_file_path.exists() or not loc_file_path.exists():
    st.error("Required files not found in repo folder: 'Predicted dataset AIO.xlsx' and 'Location matrix.xlsx'")
    st.stop()

df_pred_raw = ensure_df(pd.read_excel(pred_file_path))
df_loc       = ensure_df(pd.read_excel(loc_file_path))

# ===============================
# Sidebar – Time resolution
# ===============================
st.sidebar.header("⏱️ Time Resolution")
granularity = st.sidebar.selectbox("Time granularity", ["Daily","Weekly","Monthly"], index=0)
interp_method = st.sidebar.selectbox("Interpolation (when expanding)", ["linear","ffill"], index=0)

# ===============================
# Build availability with expansion
# ===============================
def build_availability_from_predictions(df_pred_raw: pd.DataFrame,
                                        granularity: str,
                                        interp_method: str) -> pd.DataFrame:
    df = df_pred_raw.copy()
    hospital_col = autodetect(df, ["hospital","hospital name"])
    if not hospital_col: raise ValueError("Couldn't detect hospital column in predictions.")
    df["_Hospital"] = df[hospital_col].astype(str).str.strip()

    date_col  = autodetect(df, ["date"])
    year_col  = autodetect(df, ["year"])
    month_col = autodetect(df, ["month"])
    day_col   = autodetect(df, ["day"])

    if date_col:
        df["_Date"] = pd.to_datetime(df[date_col], errors="coerce")
    elif year_col and month_col:
        if day_col: df["_Date"] = pd.to_datetime(dict(year=df[year_col], month=df[month_col], day=df[day_col]), errors="coerce")
        else: df["_Date"] = pd.to_datetime(df[year_col].astype(int).astype(str) + "-" +
                                           df[month_col].astype(int).astype(str) + "-01", errors="coerce")
    else:
        raise ValueError("Provide either a Date column or (Year & Month) in predictions.")
    df = df.dropna(subset=["_Date"]); df["_Date"] = df["_Date"].dt.normalize()

    pred_normal_avail_col = autodetect(df, ["predicted normal beds available","normal beds available (pred)","beds available predicted","pred beds"])
    pred_icu_avail_col    = autodetect(df, ["predicted icu beds available","icu beds available (pred)","icu available predicted","pred icu"])
    beds_total_col  = autodetect(df, ["beds total","total beds"])
    icu_total_col   = autodetect(df, ["icu beds total","total icu"])
    beds_occ_col    = autodetect(df, ["beds occupied","occupied beds"])
    icu_occ_col     = autodetect(df, ["icu beds occupied","occupied icu"])

    if pred_normal_avail_col and pred_icu_avail_col:
        df["_BedsAvail"] = pd.to_numeric(df[pred_normal_avail_col], errors="coerce")
        df["_ICUAvail"]  = pd.to_numeric(df[pred_icu_avail_col], errors="coerce")
    elif all([beds_total_col, beds_occ_col, icu_total_col, icu_occ_col]):
        df["_BedsAvail"] = pd.to_numeric(df[beds_total_col], errors="coerce") - pd.to_numeric(df[beds_occ_col], errors="coerce")
        df["_ICUAvail"]  = pd.to_numeric(df[icu_total_col],  errors="coerce") - pd.to_numeric(df[icu_occ_col],  errors="coerce")
    else:
        raise ValueError("Could not find predicted availability columns or totals/occupied fallback.")
    df["_BedsAvail"] = df["_BedsAvail"].fillna(0); df["_ICUAvail"] = df["_ICUAvail"].fillna(0)

    if granularity == "Monthly":
        df["_Month"] = df["_Date"].dt.to_period("M").dt.to_timestamp()
        grouped = (df.groupby(["_Hospital","_Month"], as_index=False)[["_BedsAvail","_ICUAvail"]].mean())
        availability = (grouped.set_index(["_Hospital","_Month"]).sort_index())
        availability.index = availability.index.set_names(["_Hospital","_Date"])
        return availability

    # Expand to Daily, then weekly if needed
    df_ts = df.set_index("_Date")
    availability = []
    for hosp, g in df_ts.groupby("_Hospital"):
        g_daily = g[["_BedsAvail","_ICUAvail"]].resample("D").interpolate(method=interp_method)
        g_daily["_Hospital"] = hosp
        availability.append(g_daily)
    availability = pd.concat(availability)
    availability = availability.set_index(["_Hospital", availability.index])
    return availability

availability = build_availability_from_predictions(df_pred_raw, granularity, interp_method)

# ===============================
# Distance matrix
# ===============================
distance_matrix = build_distance_matrix(df_loc)
dm_to_av, ui_to_dm, ui_to_av = build_name_maps(availability, distance_matrix, HOSPITALS_UI)

# ===============================
# Dynamic Monthly Served KPI
# ===============================
def dynamic_monthly_served(df_pred: pd.DataFrame) -> pd.DataFrame:
    df = df_pred.copy()
    hosp_col = autodetect(df, ["hospital","hospital name"])
    date_col = autodetect(df, ["date"])
    total_col = autodetect(df, ["total admitted","total admitted till date","admitted till date"])

    if not (hosp_col and date_col and total_col):
        return pd.DataFrame(columns=["Hospital","Month","Served"])

    df["_Hospital"] = df[hosp_col].astype(str).str.strip()
    df["_Date"] = pd.to_datetime(df[date_col], errors="coerce")
    df["_Month"] = df["_Date"].dt.to_period("M").dt.to_timestamp()
    df["_TotalAdmit"] = pd.to_numeric(df[total_col], errors="coerce").fillna(0)

    monthly_served = []
    for hosp, group in df.groupby("_Hospital"):
        grp_sorted = group.sort_values("_Date")
        grp_sorted["_MonthlyServed"] = grp_sorted["_TotalAdmit"].diff().fillna(grp_sorted["_TotalAdmit"])
        monthly = grp_sorted.groupby("_Month")["_MonthlyServed"].sum().reset_index()
        monthly["Hospital"] = hosp
        monthly_served.append(monthly)
    return pd.concat(monthly_served, ignore_index=True)

dynamic_served_df = dynamic_monthly_served(df_pred_raw)

# ===============================
# Dashboard KPI section
# ===============================
st.header("📊 Dashboard")
dashboard_start_av = st.selectbox("Select Hospital", HOSPITALS_UI, index=0)
dashboard_date     = st.date_input("Select Date", datetime.today())

def month_str_from_date(dt: datetime) -> pd.Timestamp:
    return pd.to_datetime(dt).to_period("M").to_timestamp()

dashboard_month = month_str_from_date(dashboard_date)
hosp_month_data = dynamic_served_df[
    (dynamic_served_df["Hospital"] == dashboard_start_av) &
    (dynamic_served_df["_Month"] == dashboard_month)
]
served_count_dynamic = int(hosp_month_data["_MonthlyServed"].sum()) if not hosp_month_data.empty else 0

# KPI Cards
st.markdown(f"""
<div class="card grid grid-4">
    <div>
        <div class="kpi">{served_count_dynamic}</div>
        <div class="kpi-label">Total Patients Served (this month)</div>
    </div>
    <div>
        <div class="kpi">{availability.loc[(dashboard_start_av,dashboard_month)]['_BedsAvail'] if (dashboard_start_av,dashboard_month) in availability.index else 0}</div>
        <div class="kpi-label">Available General Beds</div>
    </div>
    <div>
        <div class="kpi">{availability.loc[(dashboard_start_av,dashboard_month)]['_ICUAvail'] if (dashboard_start_av,dashboard_month) in availability.index else 0}</div>
        <div class="kpi-label">Available ICU Beds</div>
    </div>
    <div>
        <div class="kpi">--</div>
        <div class="kpi-label">Other Metric</div>
    </div>
</div>
""", unsafe_allow_html=True)
