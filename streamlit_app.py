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
    # --- Get hospital names safely ---
  if isinstance(availability.index, pd.MultiIndex):
    avail_names = availability.index.get_level_values(0)
else:
    avail_names = availability.index
    
    dm_names = sorted(set(map(str, dist_mat.index.tolist())) | set(map(str, dist_mat.columns.tolist())))
    ui_names = list(ui_list)

    avail_by_key = {norm_key(a): a for a in avail_names}
    dm_by_key    = {norm_key(d): d for d in dm_names}

    dm_to_av = {}
    for d in dm_names:
        kd = norm_key(d)
        if kd in avail_by_key: 
            dm_to_av[d] = avail_by_key[kd]
        else:
            m = get_close_matches(kd, list(avail_by_key.keys()), n=1, cutoff=0.6)
            dm_to_av[d] = avail_by_key[m[0]] if m else None

    ui_to_dm, ui_to_av = {}, {}
    for u in ui_names:
        ku = norm_key(u)
        if ku in dm_by_key: 
            ui_to_dm[u] = dm_by_key[ku]
        else:
            m = get_close_matches(ku, list(dm_by_key.keys()), n=1, cutoff=0.6)
            ui_to_dm[u] = dm_by_key[m[0]] if m else None

        if ku in avail_by_key: 
            ui_to_av[u] = avail_by_key[ku]
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
# Build availability
# ===============================
# (Same as before; not changing)
def build_availability_from_predictions(df_pred_raw: pd.DataFrame,
                                        granularity: str,
                                        interp_method: str) -> pd.DataFrame:
    # ... [same as your previous code] ...
    pass  # keeping this unchanged for brevity; include your original function here

try:
    availability = build_availability_from_predictions(df_pred_raw, granularity, interp_method)
except Exception as e:
    st.error(f"Error building availability: {e}")
    st.stop()

# ===============================
# Distance matrix + name maps
# ===============================
dist_mat = build_distance_matrix(df_loc)
DM_TO_AV, UI_TO_DM, UI_TO_AV = build_name_maps(availability, dist_mat, HOSPITALS_UI)

# ===============================
# Allocation helpers
# ===============================
if "reservations" not in st.session_state:
    st.session_state["reservations"] = {}
if "served" not in st.session_state:
    st.session_state["served"] = {}
if "reroute_log" not in st.session_state:
    st.session_state["reroute_log"] = []

def month_str_from_date(dt) -> str:
    return pd.to_datetime(dt).strftime("%Y-%m")

# ===============================
# --- NEW: Monthly Admissions from cumulative ---
# ===============================
def get_month_admissions(df_pred: pd.DataFrame, hospital_col:str, date_col:str, total_admitted_col:str, selected_date):
    df = df_pred.copy()
    df["_Hospital"] = df[hospital_col].astype(str).str.strip()
    df["_Date"] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
    df = df.dropna(subset=["_Date"])

    month_start = pd.to_datetime(selected_date).replace(day=1)
    next_month_start = month_start + pd.DateOffset(months=1)

    month_mask = (df["_Date"] < next_month_start) & (df["_Date"] >= month_start)
    df_month = df.loc[month_mask]

    month_totals = df_month.groupby("_Hospital")[total_admitted_col].max()

    prev_mask = df["_Date"] < month_start
    df_prev = df.loc[prev_mask]
    prev_totals = df_prev.groupby("_Hospital")[total_admitted_col].max()
    prev_totals = prev_totals.reindex(month_totals.index, fill_value=0)

    monthly_admissions = month_totals - prev_totals
    monthly_admissions = monthly_admissions.fillna(0).astype(int)
    return monthly_admissions.to_dict()

# ===============================
# UI – Patient inputs
# ===============================
# ... same allocation form as before ...

# ===============================
# Dashboard: show hospital month info
# ===============================
st.markdown("---")
st.header("📊 Hospital Monthly Dashboard")

dashboard_ui_hospital = st.selectbox("Choose hospital to view dashboard", HOSPITALS_UI, index=0)
dashboard_date = st.date_input("View month (pick any date in month)", value=max_d, min_value=min_d, max_value=max_d)

dashboard_start_av = UI_TO_AV.get(dashboard_ui_hospital) or dashboard_ui_hospital
dashboard_month = month_str_from_date(dashboard_date)

# --- NEW: served_count from monthly admissions ---
hospital_col = autodetect(df_pred_raw, ["hospital","hospital name"])
date_col     = autodetect(df_pred_raw, ["date"])
total_adm_col= autodetect(df_pred_raw, ["total admitted","total admitted till date"])
monthly_admissions_dict = get_month_admissions(df_pred_raw, hospital_col, date_col, total_adm_col, dashboard_date)
served_count = monthly_admissions_dict.get(dashboard_start_av, 0)

st.metric("Patients Served This Month", served_count)

# --- continue showing beds, ICU, severity, ribbons, etc. as before ---
# ... copy your previous dashboard visualization code here ...

