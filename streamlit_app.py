# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import re
from difflib import get_close_matches
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List
import plotly.express as px

# ===============================
# App config
# ===============================
st.set_page_config(page_title="Dengue Allocation", page_icon="🏥", layout="wide")
st.title("🏥 Integrated Hospital Dengue Patient Allocation System (DSCC Region)")

# ===============================
# Fixed hospital list (UI names)
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
# Styles (catchy, glassy cards)
# ===============================
st.markdown("""
<style>
:root{--muted:#94a3b8;--accent:#06b6d4;--good:#10b981;--warn:#f59e0b;--bad:#ef4444}
html, body, [data-testid="stAppViewContainer"]{background:linear-gradient(135deg,#071126 0%,#071b2a 60%,#081527 100%) !important;color:#e6eef8}
.card{border-radius:14px;padding:14px;background:linear-gradient(180deg,rgba(255,255,255,0.03),rgba(255,255,255,0.02));border:1px solid rgba(255,255,255,0.04);}
.kpi{font-weight:800;font-size:2.0rem;margin:0}
.kpi-label{color:var(--muted);font-size:.9rem}
.small{color:var(--muted);font-size:.9rem}
.badge{padding:6px 10px;border-radius:9999px;font-weight:700;color:#fff}
.badge.green{background:linear-gradient(90deg,#10b981,#059669)}
.badge.red{background:linear-gradient(90deg,#ef4444,#b91c1c)}
.badge.blue{background:linear-gradient(90deg,#06b6d4,#0ea5b7)}
.sep{height:1px;background:rgba(255,255,255,0.03);margin:12px 0}
</style>
""", unsafe_allow_html=True)

# ===============================
# Helper utils
# ===============================
STOPWORDS = {"hospital","medical","college","institute","university","center","centre","clinic","and"}

def norm_key(s: str) -> str:
    s = str(s).lower().strip().replace("&"," and ")
    s = re.sub(r"[^a-z0-9\s]"," ", s)
    tokens = [t for t in s.split() if t and t not in STOPWORDS]
    return "".join(tokens)

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

# ===============================
# Severity & resource logic
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

raw_pred = ensure_df(pd.read_excel(pred_file_path))
df_loc  = ensure_df(pd.read_excel(loc_file_path))

# ===============================
# Build availability and cumulative admitted pivot
# ===============================
def build_availability_and_cum(df: pd.DataFrame, granularity: str, interp_method: str):
    df = df.copy()
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
            df["_Date"] = pd.to_datetime(df[year_col].astype(int).astype(str) + "-" + df[month_col].astype(int).astype(str) + "-01", errors="coerce")
    else:
        raise ValueError("Provide either a Date column or (Year & Month) in predictions.")

    df = df.dropna(subset=["_Date"]); df["_Date"] = df["_Date"].dt.normalize()

    # availability columns
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

    # cumulative admitted column detection (the user's YTD column)
    cum_col = autodetect(df, ["total admitted till date","total admitted","admitted till date","total admitted till"])

    # prepare availability time series (daily/weekly/monthly) similar to earlier
    if granularity == "Monthly":
        df["_Month"] = df["_Date"].dt.to_period("M").dt.to_timestamp()
        grouped = (df.groupby(["_Hospital","_Month"], as_index=False)[["_BedsAvail","_ICUAvail"]].mean())
        availability = (grouped.set_index(["_Hospital","_Month"]).sort_index())
        availability.index = availability.index.set_names(["_Hospital","_Date"])
    else:
        df_ts = df.set_index("_Date")
        beds_piv = df_ts.pivot_table(index="_Date", columns="_Hospital", values="_BedsAvail", aggfunc="mean")
        icu_piv  = df_ts.pivot_table(index="_Date", columns="_Hospital", values="_ICUAvail",  aggfunc="mean")
        full_idx = pd.date_range(start=beds_piv.index.min(), end=beds_piv.index.max(), freq="D")
        beds_piv = beds_piv.reindex(full_idx); icu_piv = icu_piv.reindex(full_idx)
        if interp_method == "linear":
            beds_piv = beds_piv.interpolate(method="time", limit_direction="both")
            icu_piv  = icu_piv.interpolate(method="time", limit_direction="both")
        else:
            beds_piv = beds_piv.ffill().bfill(); icu_piv = icu_piv.ffill().bfill()
        if granularity == "Weekly":
            beds_piv = beds_piv.resample("W-MON").mean()
            icu_piv  = icu_piv.resample("W-MON").mean()
        beds_long = beds_piv.stack(dropna=False).rename("_BedsAvail").to_frame()
        icu_long  = icu_piv.stack(dropna=False).rename("_ICUAvail").to_frame()
        long_df = beds_long.join(icu_long, how="outer").reset_index()
        long_df.columns = ["_Date","_Hospital","_BedsAvail","_ICUAvail"]
        long_df["_BedsAvail"] = long_df["_BedsAvail"].fillna(0).clip(lower=0)
        long_df["_ICUAvail"]  = long_df["_ICUAvail"].fillna(0).clip(lower=0)
        availability = (long_df.groupby(["_Hospital","_Date"], as_index=False)[["_BedsAvail","_ICUAvail"]].mean().set_index(["_Hospital","_Date"]).sort_index())

    # build cumulative pivot if column exists
    cum_pivot = None
    if cum_col:
        tmp = df[["_Hospital","_Date", cum_col]].copy()
        tmp[cum_col] = pd.to_numeric(tmp[cum_col], errors="coerce").fillna(0)
        cum_pivot = tmp.pivot_table(index="_Date", columns="_Hospital", values=cum_col, aggfunc="max").sort_index()
        # reindex to full daily index and forward fill
        full_idx = pd.date_range(start=cum_pivot.index.min(), end=cum_pivot.index.max(), freq="D")
        cum_pivot = cum_pivot.reindex(full_idx).ffill().fillna(0)

    return availability, cum_pivot

# Sidebar controls
st.sidebar.header("⏱️ Time Resolution")
granularity = st.sidebar.selectbox("Time granularity", ["Daily","Weekly","Monthly"], index=0)
interp_method = st.sidebar.selectbox("Interpolation (when expanding)", ["linear","ffill"], index=0)

try:
    availability, cum_pivot = build_availability_and_cum(raw_pred, granularity, interp_method)
except Exception as e:
    st.error(f"Error building availability: {e}")
    st.stop()

# ===============================
# Distance matrix + name maps
# ===============================
def build_distance_matrix(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy(); df.columns = [str(c).strip() for c in df.columns]
    if df.shape[1] > 2:
        df = df.set_index(df.columns[0])
        for c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
        return df.combine_first(df.T)
    raise ValueError("Could not interpret Location matrix format.")

DM = build_distance_matrix(df_loc)

# name maps between DM sheet and availability
avail_names = sorted(set(availability.index.get_level_values(0).tolist()))
dm_names = sorted(set(map(str, DM.index.tolist())) | set(map(str, DM.columns.tolist())))

avail_by_key = {norm_key(a): a for a in avail_names}
dm_by_key    = {norm_key(d): d for d in dm_names}

DM_TO_AV = {}
for d in dm_names:
    kd = norm_key(d)
    if kd in avail_by_key: DM_TO_AV[d] = avail_by_key[kd]
    else:
        m = get_close_matches(kd, list(avail_by_key.keys()), n=1, cutoff=0.6)
        DM_TO_AV[d] = avail_by_key[m[0]] if m else None

UI_TO_DM = {}
UI_TO_AV = {}
for u in HOSPITALS_UI:
    ku = norm_key(u)
    if ku in dm_by_key: UI_TO_DM[u] = dm_by_key[ku]
    else:
        m = get_close_matches(ku, list(dm_by_key.keys()), n=1, cutoff=0.6)
        UI_TO_DM[u] = dm_by_key[m[0]] if m else None
    if ku in avail_by_key: UI_TO_AV[u] = avail_by_key[ku]
    else:
        m2 = get_close_matches(ku, list(avail_by_key.keys()), n=1, cutoff=0.6)
        UI_TO_AV[u] = avail_by_key[m2[0]] if m2 else None

# ===============================
# Allocation helpers and session state
# ===============================
if "reservations" not in st.session_state:
    st.session_state["reservations"] = {}
if "reroute_log" not in st.session_state:
    st.session_state["reroute_log"] = []

def get_remaining(hospital: str, date, bed_type: str) -> int:
    base = 0.0
    key = (hospital, pd.to_datetime(date).normalize())
    if key in availability.index:
        base = float(availability.loc[key, "_ICUAvail" if bed_type == "ICU" else "_BedsAvail"])
    reserved = st.session_state["reservations"].get((hospital, key[1], bed_type), 0)
    return max(0, int(np.floor(base)) - int(reserved))

def reserve_bed(hospital: str, date, bed_type: str, n: int = 1):
    k = (hospital, pd.to_datetime(date).normalize(), bed_type)
    st.session_state["reservations"][k] = st.session_state["reservations"].get(k, 0) + n

def find_reroute_nearest_first(start_ui_name: str, date, bed_key: str):
    start_dm = UI_TO_DM.get(start_ui_name)
    checks = []
    if not start_dm or start_dm not in DM.index:
        return None, None, "Hospital not found in distance matrix", checks
    row = DM.loc[start_dm].astype(float).dropna().sort_values()
    for neighbor_dm, dist in row.items():
        if neighbor_dm == start_dm: continue
        neighbor_av = DM_TO_AV.get(neighbor_dm)
        rem = None
        if neighbor_av and ((neighbor_av, pd.to_datetime(date).normalize()) in availability.index):
            rem = get_remaining(neighbor_av, date, bed_key)
        checks.append({"Neighbor Hospital": neighbor_dm, "Remaining Beds/ICU": rem, "Distance (km)": float(dist)})
        if rem and rem > 0:
            return neighbor_av, float(dist), None, checks
    return None, None, "No hospitals with vacancy found", checks

# ===============================
# New: functions using cumulative admitted (monthly counts)
# ===============================

def month_start(dt):
    dt = pd.to_datetime(dt)
    return pd.to_datetime(dt.strftime('%Y-%m-01'))

def cum_on_date(hospital_av_name: str, date) -> float:
    """Return cumulative admitted (YTD) for hospital at exact date using cum_pivot; if missing, choose last available before date."""
    if cum_pivot is None or hospital_av_name not in cum_pivot.columns:
        return 0.0
    d = pd.to_datetime(date)
    if d in cum_pivot.index:
        return float(cum_pivot.loc[d, hospital_av_name])
    # pick last index <= d
    idx = cum_pivot.index[cum_pivot.index <= d]
    if len(idx) == 0: return 0.0
    return float(cum_pivot.loc[idx[-1], hospital_av_name])

def monthly_count_to_date(hospital_av_name: str, date) -> int:
    """Return admitted count from month start up to and including `date` for this hospital."""
    d = pd.to_datetime(date)
    ms = month_start(d)
    prev_day = ms - timedelta(days=1)
    cum_at_date = cum_on_date(hospital_av_name, d)
    cum_before_month = cum_on_date(hospital_av_name, prev_day)
    return max(0, int(round(cum_at_date - cum_before_month)))

def monthly_total_for_month(hospital_av_name: str, month_dt) -> int:
    """Return total admitted for the whole month (month_dt is any date in month)."""
    # last day of month
    m = pd.to_datetime(month_dt)
    last_day = (m + pd.offsets.MonthEnd(0)).normalize()
    return monthly_count_to_date(hospital_av_name, last_day)

# ===============================
# UI – Patient inputs (form)
# ===============================
all_dates = sorted(list(set([d for _, d in availability.index])))
min_d, max_d = min(all_dates), max(all_dates)

with st.form("allocation_form"):
    st.subheader("Patient Intake")
    c1,c2,c3,c4 = st.columns([1.2,1,1,1])
    with c1:
        hospital_ui = st.selectbox("Hospital Name", HOSPITALS_UI)
        date_input  = st.date_input("Date", value=max_d, min_value=min_d, max_value=max_d)
        weight = st.number_input("Weight (kg)", min_value=1.0, max_value=250.0, value=60.0)
    with c2:
        age = st.number_input("Age (years)", min_value=0, max_value=120, value=25)
        platelet = st.number_input("Platelet Count (/µL)", min_value=0, value=120000, step=1000)
    with c3:
        ns1_val = st.selectbox("NS1", [0,1], index=0, help="0=Negative, 1=Positive")
        igm_val = st.selectbox("IgM", [0,1], index=0, help="0=Negative, 1=Positive")
    with c4:
        igg_val = st.selectbox("IgG", [0,1], index=0, help="0=Negative, 1=Positive")
        st.caption(f"Time: **{granularity}** · Interp: **{interp_method if granularity!='Monthly' else 'N/A'}**")
    submit = st.form_submit_button("🚑 Allocate")

# ===============================
# Allocation on submit
# ===============================
assigned_av = None
rerouted_distance = None
note = ""
debug_checks = []

if submit:
    p_score, s_score = compute_severity_score(age, ns1_val, igm_val, igg_val, platelet)
    severity = verdict_from_score(s_score)
    resource = required_resource(severity)
    bed_key  = "ICU" if resource == "ICU" else "Normal"

    start_av = UI_TO_AV.get(hospital_ui) or hospital_ui
    remaining_here = get_remaining(start_av, date_input, bed_key)

    if remaining_here > 0 and ((start_av, pd.to_datetime(date_input).normalize()) in availability.index):
        assigned_av, rerouted_distance, note = start_av, None, "Assigned at selected hospital"
        available_status = "Yes"
    else:
        available_status = "No vacancy available here"
        assigned_av, rerouted_distance, err, debug_checks = find_reroute_nearest_first(hospital_ui, date_input, bed_key)
        note = f"Rerouted to {assigned_av}" if assigned_av else err

    if assigned_av:
        reserve_bed(assigned_av, date_input, bed_key, 1)
        # log reroute if different
        if assigned_av != (start_av or hospital_ui):
            st.session_state["reroute_log"].append({
                "date": pd.to_datetime(date_input).date().isoformat(),
                "original_ui": hospital_ui,
                "assigned_av": assigned_av,
                "month": pd.to_datetime(date_input).strftime('%Y-%m')
            })

    # ---------- Allocation Ticket UI ----------
    st.subheader("Allocation Result")
    st.markdown('<div class="card" style="padding:18px">', unsafe_allow_html=True)
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        st.markdown(f"<div class='kpi'>{s_score}</div><div class='kpi-label'>Severity Score</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='small'>{verdict_from_score(s_score)}</div>", unsafe_allow_html=True)
    with col_b:
        st.markdown(f"<div class='kpi'>{resource}</div><div class='kpi-label'>Resource Needed</div>", unsafe_allow_html=True)
    with col_c:
        st.markdown(f"<div class='kpi'>{pd.to_datetime(date_input).date()}</div><div class='kpi-label'>Date</div>", unsafe_allow_html=True)
    with col_d:
        dist_txt = f"{float(rerouted_distance):.1f} km" if rerouted_distance is not None else "—"
        st.markdown(f"<div class='kpi'>{dist_txt}</div><div class='kpi-label'>Travel Distance</div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ticket body
    st.markdown('<div class="card" style="margin-top:12px">', unsafe_allow_html=True)
    left, right = st.columns([1.3,0.9])
    with left:
        if available_status == "Yes":
            st.success("✅ Bed available at selected hospital")
        else:
            st.warning("⚠️ No vacancy available here — finding nearest option…")
        st.markdown('<div class="sep"></div>', unsafe_allow_html=True)
        st.write(f"**Hospital Tried:** {hospital_ui}")
        st.write(f"**Assigned Hospital:** {assigned_av if assigned_av else '—'}")
        st.caption(f"Note: {note}")
    with right:
        summary = {
            "Severity": severity,
            "Resource": resource,
            "Severity Score": s_score,
            "Hospital Tried": hospital_ui,
            "Available at Current Hospital": available_status,
            "Assigned Hospital": assigned_av,
            "Distance (km)": float(rerouted_distance) if rerouted_distance is not None else None
        }
        st.json(summary)
    st.markdown('</div>', unsafe_allow_html=True)

    # debug
    with st.expander("🧪 Debug: Nearest Hospitals Checked"):
        if debug_checks:
            st.dataframe(pd.DataFrame(debug_checks), use_container_width=True)
        else:
            st.write("No neighbor checks — assigned at selected hospital.")

# ===============================
# Dashboard — interactive & monthly (uses cumulative column)
# ===============================
st.markdown("---")
st.header("📊 Interactive Hospital Dashboard (Monthly view)")

dash_col1, dash_col2 = st.columns([1,1])
with dash_col1:
    dashboard_ui_hospital = st.selectbox("Choose hospital to view dashboard", HOSPITALS_UI, index=0)
with dash_col2:
    dashboard_date = st.date_input("View month (pick any date in month)", value=max_d, min_value=min_d, max_value=max_d)

dashboard_av = UI_TO_AV.get(dashboard_ui_hospital) or dashboard_ui_hospital
month_key = pd.to_datetime(dashboard_date).strftime('%Y-%m')

# compute monthly served using cumulative pivot (to-date in-month)
served_to_date = monthly_count_to_date(dashboard_av, dashboard_date) if 'cum_pivot' in locals() else 0
served_full_month = monthly_total_for_month(dashboard_av, dashboard_date) if 'cum_pivot' in locals() else 0

# availability on selected day
key = (dashboard_av, pd.to_datetime(dashboard_date).normalize())
if key in availability.index:
    av_row = availability.loc[key]
    beds_avail = int(np.floor(float(av_row['_BedsAvail'])))
    icu_avail  = int(np.floor(float(av_row['_ICUAvail'])))
else:
    beds_avail = None; icu_avail = None

# top KPI row
kp1, kp2, kp3, kp4 = st.columns([1,1,1,1])
with kp1:
    st.markdown(f"<div class='card'><div class='kpi'>{served_to_date}</div><div class='kpi-label'>Patients this month (to selected date)</div></div>", unsafe_allow_html=True)
with kp2:
    st.markdown(f"<div class='card'><div class='kpi'>{served_full_month}</div><div class='kpi-label'>Patients this month (full month)</div></div>", unsafe_allow_html=True)
with kp3:
    st.markdown(f"<div class='card'><div class='kpi'>{beds_avail if beds_avail is not None else '—'}</div><div class='kpi-label'>Normal beds available (selected day)</div></div>", unsafe_allow_html=True)
with kp4:
    st.markdown(f"<div class='card'><div class='kpi'>{icu_avail if icu_avail is not None else '—'}</div><div class='kpi-label'>ICU beds available (selected day)</div></div>", unsafe_allow_html=True)

st.markdown("\n")

# Time series visualizations if cumulative pivot exists
if cum_pivot is not None:
    # prepare a timeseries for this hospital
    hos = dashboard_av
    if hos in cum_pivot.columns:
        ts = cum_pivot[hos].reset_index().rename(columns={'index':'Date', hos:'Cumulative'})
        # daily admissions (diff)
        ts['Daily'] = ts['Cumulative'].diff().fillna(ts['Cumulative']).clip(lower=0)
        ts['Month'] = ts['Date'].dt.to_period('M').dt.to_timestamp()

        # filter to 6 months window for nicer display
        end = pd.to_datetime(dashboard_date)
        start = end - pd.DateOffset(months=5)
        ts_win = ts[(ts['Date']>=start)&(ts['Date']<=end)]

        fig1 = px.line(ts_win, x='Date', y='Cumulative', title=f'Cumulative admissions — {dashboard_ui_hospital}', labels={'Cumulative':'Cumulative admitted'})
        fig2 = px.bar(ts_win, x='Date', y='Daily', title=f'Daily admissions (derived) — {dashboard_ui_hospital}', labels={'Daily':'Admissions'})

        st.plotly_chart(fig1, use_container_width=True)
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No cumulative admitted data available for this hospital in predictions.")
else:
    st.info("Dataset does not contain a cumulative 'Total Admitted Till Date' column — monthly counts cannot be computed automatically.")

# Reroute dashboards (show aggregated reroutes in selected month)
st.markdown('---')
st.subheader('Reroute Summary')
if st.session_state['reroute_log']:
    df_r = pd.DataFrame(st.session_state['reroute_log'])
    df_month = df_r[df_r['month']==month_key]
    if not df_month.empty:
        agg = df_month.groupby('assigned_av').size().reset_index(name='ReroutedCount').sort_values('ReroutedCount', ascending=False)
        st.dataframe(agg, use_container_width=True)
        # show small cards for top rerouted hospitals
        for _, row in agg.iterrows():
            hosp = row['assigned_av']
            count = int(row['ReroutedCount'])
            month_total = monthly_total_for_month(hosp, dashboard_date) if cum_pivot is not None else 'n/a'
            st.markdown(f"**{hosp}** — rerouted here: {count} times this month — total admissions this month: {month_total}")
    else:
        st.info('No reroutes recorded for the selected month yet.')
else:
    st.info('No reroute events logged yet.')

# Monthly leaderboard across hospitals (based on cumulative pivot)
st.markdown('---')
st.subheader('Monthly Leaderboard — Patients (selected month)')
if cum_pivot is not None:
    # compute monthly totals for each hospital for the selected month
    month_end = (pd.to_datetime(dashboard_date) + pd.offsets.MonthEnd(0)).normalize()
    month_start_dt = month_start(dashboard_date)
    results = []
    for col in cum_pivot.columns:
        cum_end = cum_on_date(col, month_end)
        cum_before = cum_on_date(col, month_start_dt - timedelta(days=1))
        results.append({'Hospital': col, 'PatientsThisMonth': max(0, int(round(cum_end - cum_before)))})
    df_rank = pd.DataFrame(results).sort_values('PatientsThisMonth', ascending=False).reset_index(drop=True)
    if not df_rank.empty:
        fig = px.bar(df_rank.head(12), x='PatientsThisMonth', y='Hospital', orientation='h', title='Top hospitals this month')
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df_rank, use_container_width=True)
    else:
        st.write('No admissions recorded in dataset for this month.')
else:
    st.info('No cumulative admitted column found in dataset; cannot compute leaderboard.')

# Optional: raw logs and reservation debug
with st.expander('🔁 Full Reroute Log'):
    if st.session_state['reroute_log']:
        st.dataframe(pd.DataFrame(st.session_state['reroute_log']), use_container_width=True)
    else:
        st.write('No reroute events logged yet.')

with st.expander('🗂️ Raw Reservations (debug)'):
    if st.session_state['reservations']:
        rows = []
        for (h, date, bed_type), cnt in st.session_state['reservations'].items():
            rows.append({'Hospital':h, 'Date': date, 'Bed Type': bed_type, 'Reserved': cnt})
        st.dataframe(pd.DataFrame(rows).sort_values(['Date','Hospital']), use_container_width=True)
    else:
        st.write('No reservations yet.')

# Footer note
st.markdown('\n---\n*Note: Monthly "Patients this month" is derived from the dataset column that contains cumulative "Total Admitted Till Date" values. If your predictions file does not contain that column, monthly counts cannot be generated automatically.*')
