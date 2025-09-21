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
    score = 0
    score += 2 if ns1 else 0
    score += 2 if igm else 0
    score += 1 if igg else 0
    score += compute_platelet_score(platelet)
    return score, min(max(score,0),10)

def severity_from_score(score: int) -> str:
    if score <= 3: return "Mild"
    if score <= 6: return "Moderate"
    if score <= 8: return "Severe"
    return "Very Severe"

# ===============================
# Build availability from predictions
# ===============================
def build_availability_from_predictions(df_pred_raw: pd.DataFrame,
                                        granularity: str="Daily",
                                        interp_method: str="linear") -> pd.DataFrame:
    df = ensure_df(df_pred_raw).copy()
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
        if day_col:
            df["_Date"] = pd.to_datetime(dict(year=df[year_col], month=df[month_col], day=df[day_col]), errors="coerce")
        else:
            df["_Date"] = pd.to_datetime(df[year_col].astype(int).astype(str) + "-" +
                                         df[month_col].astype(int).astype(str) + "-01", errors="coerce")
    else:
        raise ValueError("Provide either a Date column or (Year & Month) in predictions.")
    df = df.dropna(subset=["_Date"]); df["_Date"] = df["_Date"].dt.normalize()

    # detect availability columns
    pred_normal_avail_col = autodetect(df, ["predicted normal beds available","normal beds available (pred)","beds available predicted","pred beds"])
    pred_icu_avail_col    = autodetect(df, ["predicted icu beds available","icu beds available (pred)","icu available predicted","pred icu"])
    beds_total_col  = autodetect(df, ["beds total","total beds"])
    icu_total_col   = autodetect(df, ["icu beds total","total icu"])
    beds_occ_col    = autodetect(df, ["beds occupied","occupied beds"])
    icu_occ_col     = autodetect(df, ["icu beds occupied","occupied icu"])
    admitted_till_date_col = autodetect(df, ["total admitted till date", "admitted till date", "total admitted"])

    if pred_normal_avail_col and pred_icu_avail_col:
        df["_BedsAvail"] = pd.to_numeric(df[pred_normal_avail_col], errors="coerce")
        df["_ICUAvail"]  = pd.to_numeric(df[pred_icu_avail_col], errors="coerce")
    elif all([beds_total_col, beds_occ_col, icu_total_col, icu_occ_col]):
        df["_BedsAvail"] = pd.to_numeric(df[beds_total_col], errors="coerce") - pd.to_numeric(df[beds_occ_col], errors="coerce")
        df["_ICUAvail"]  = pd.to_numeric(df[icu_total_col],  errors="coerce") - pd.to_numeric(df[icu_occ_col],  errors="coerce")
    else:
        raise ValueError("Could not find predicted availability columns or totals/occupied fallback.")

    df["_BedsAvail"] = df["_BedsAvail"].fillna(0); df["_ICUAvail"] = df["_ICUAvail"].fillna(0)

    # attach total admitted till date
    if admitted_till_date_col:
        df["_TotalAdmittedTillDate"] = pd.to_numeric(df[admitted_till_date_col], errors="coerce").fillna(0)
    else:
        df["_TotalAdmittedTillDate"] = np.nan

    # daily pivot
    df_ts = df.set_index("_Date")
    beds_piv = df_ts.pivot_table(index="_Date", columns="_Hospital", values="_BedsAvail", aggfunc="mean")
    icu_piv  = df_ts.pivot_table(index="_Date", columns="_Hospital", values="_ICUAvail",  aggfunc="mean")
    admit_piv= df_ts.pivot_table(index="_Date", columns="_Hospital", values="_TotalAdmittedTillDate", aggfunc="last")
    
    full_idx = pd.date_range(start=beds_piv.index.min(), end=beds_piv.index.max(), freq="D")
    beds_piv = beds_piv.reindex(full_idx)
    icu_piv  = icu_piv.reindex(full_idx)
    admit_piv= admit_piv.reindex(full_idx)

    if interp_method == "linear":
        beds_piv = beds_piv.interpolate(method="time", limit_direction="both")
        icu_piv  = icu_piv.interpolate(method="time", limit_direction="both")
        admit_piv= admit_piv.ffill()
    else:
        beds_piv = beds_piv.ffill().bfill()
        icu_piv  = icu_piv.ffill().bfill()
        admit_piv= admit_piv.ffill()

    beds_long  = beds_piv.stack(dropna=False).rename("_BedsAvail").to_frame()
    icu_long   = icu_piv.stack(dropna=False).rename("_ICUAvail").to_frame()
    admit_long = admit_piv.stack(dropna=False).rename("_TotalAdmittedTillDate").to_frame()

    long_df = beds_long.join(icu_long).join(admit_long).reset_index()
    long_df.columns = ["_Date","_Hospital","_BedsAvail","_ICUAvail","_TotalAdmittedTillDate"]
    availability = long_df.groupby(["_Hospital","_Date"], as_index=False)[["_BedsAvail","_ICUAvail","_TotalAdmittedTillDate"]].mean()
    return availability.set_index(["_Hospital","_Date"]).sort_index()

# ===============================
# Get total served
# ===============================
def get_month_served(hospital_av_name: str, date) -> int:
    key = (hospital_av_name, pd.to_datetime(date).normalize())
    if key in availability.index and "_TotalAdmittedTillDate" in availability.columns:
        val = availability.loc[key, "_TotalAdmittedTillDate"]
        return int(val) if not pd.isna(val) else 0
    return 0

# ===============================
# Load predictions & build availability
# ===============================
uploaded_file = st.file_uploader("Upload Predicted Dataset AIO.xlsx", type=["xlsx"])
if uploaded_file:
    df_pred = pd.read_excel(uploaded_file)
    availability = build_availability_from_predictions(df_pred, granularity="Daily", interp_method="linear")
    st.success("✅ Availability built from predictions.")

    # demo: select hospital & date
    hospital_select = st.selectbox("Choose Hospital", HOSPITALS_UI)
    date_select = st.date_input("Select Date", datetime.today())
    served_count = get_month_served(hospital_select, date_select)

    st.markdown(f"<div class='card'><p class='kpi'>{served_count}</p><p class='kpi-label'>Total Patients Served (this month)</p></div>", unsafe_allow_html=True)
