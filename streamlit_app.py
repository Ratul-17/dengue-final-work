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
st.title("🏥 Integrated Hospital Dengue Patient Allocation System")

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
# Excel-based severity (NS1, IgM, IgG as 0/1)
# ===============================
def excel_based_severity(platelet, age, NS1, IgM, IgG):
    if platelet > 120000: H = 0
    elif platelet > 80000: H = 0.5
    elif platelet > 60000: H = 1
    elif platelet > 30000: H = 2
    elif platelet >= 10000: H = 3
    else: H = 4
    age_factor = 0.5 if (age < 15 or age > 60) else 0
    I = min(4, NS1 + IgM * 0.5 + IgG * 0.5 + age_factor + H)
    if I <= 1: verdict = "Mild"
    elif I < 2.5: verdict = "Moderate"
    elif I < 4: verdict = "Severe"
    else: verdict = "Very Severe"
    return verdict, I

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
    availability = (long_df.groupby(["_Hospital","_Date"], as_index=False)[["_BedsAvail","_ICUAvail"]]
                    .mean().set_index(["_Hospital","_Date"]).sort_index())
    return availability

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
    if not start_dm or start_dm not in dist_mat.index:
        return None, None, "Hospital not found in distance matrix", checks
    row = dist_mat.loc[start_dm].astype(float).dropna().sort_values()
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
# UI – Patient inputs
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
if submit:
    severity, score = excel_based_severity(platelet, age, ns1_val, igm_val, igg_val)
    resource = required_resource(severity)
    bed_key  = "ICU" if resource == "ICU" else "Normal"
    start_av = UI_TO_AV.get(hospital_ui) or hospital_ui
    remaining_here = get_remaining(start_av, date_input, bed_key)

    debug_checks = []
    if remaining_here > 0 and ((start_av, pd.to_datetime(date_input).normalize()) in availability.index):
        assigned_av, rerouted_distance, note = start_av, None, "Assigned at selected hospital"
        available_status = "Yes"
    else:
        available_status = "No vacancy available here"
        assigned_av, rerouted_distance, err, debug_checks = find_reroute_nearest_first(hospital_ui, date_input, bed_key)
        note = f"Rerouted to {assigned_av}" if assigned_av else err

    if assigned_av:
        reserve_bed(assigned_av, date_input, bed_key, 1)

    # ---------- Allocation Ticket UI ----------
    st.subheader("Allocation Result")
    st.markdown('<div class="grid grid-4">', unsafe_allow_html=True)
    # KPI cards
    st.markdown(f'''
      <div class="card">
        <div class="kpi">{round(float(score),2)}</div>
        <div class="kpi-label">Severity Score</div>
        <div class="ribbon">{severity_badge(severity)}</div>
      </div>
    ''', unsafe_allow_html=True)
    st.markdown(f'''
      <div class="card">
        <div class="kpi">{resource}</div>
        <div class="kpi-label">Resource Needed</div>
        <div class="ribbon">{resource_badge(resource)}</div>
      </div>
    ''', unsafe_allow_html=True)
    st.markdown(f'''
      <div class="card">
        <div class="kpi">{pd.to_datetime(date_input).date()}</div>
        <div class="kpi-label">Date</div>
        <div class="ribbon"><span class="badge blue">{granularity}</span></div>
      </div>
    ''', unsafe_allow_html=True)
    dist_txt = f"{float(rerouted_distance):.1f} km" if rerouted_distance is not None else "—"
    st.markdown(f'''
      <div class="card">
        <div class="kpi">{dist_txt}</div>
        <div class="kpi-label">Travel Distance</div>
        <div class="ribbon"><span class="badge blue">{interp_method if granularity!='Monthly' else 'N/A'}</span></div>
      </div>
    ''', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Ticket
    st.markdown('<div class="card ticket">', unsafe_allow_html=True)
    left, right = st.columns([1.2,.8], gap="medium")
    with left:
        # availability banner
        if available_status == "Yes":
            st.markdown('<div class="banner ok">✅ Bed available at selected hospital</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="banner warn">⚠️ No vacancy available here — finding nearest option…</div>', unsafe_allow_html=True)
        st.markdown('<div class="sep"></div>', unsafe_allow_html=True)

        # Route lane
        tried = hospital_ui
        st.markdown(f"**Hospital Tried:** {tried}", unsafe_allow_html=True)
        st.markdown('<div class="route" style="margin-top:8px">', unsafe_allow_html=True)
        st.markdown(f'<span class="pill">{tried}</span>', unsafe_allow_html=True)
        st.markdown('<div class="arrow">➡️</div>', unsafe_allow_html=True)
        final_chip = f'<span class="pill">{assigned_av if assigned_av else "—"}</span>'
        st.markdown(final_chip, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.caption(f"Note: **{note}**")

    with right:
        st.markdown("**Summary**")
        st.markdown('<div class="sep"></div>', unsafe_allow_html=True)
        summary = {
            "Severity": severity,
            "Resource": resource,
            "Hospital Tried": tried,
            "Available at Current Hospital": available_status,
            "Assigned Hospital": assigned_av,
            "Distance (km)": float(rerouted_distance) if rerouted_distance is not None else None
        }
        st.markdown('<div class="codebox">', unsafe_allow_html=True)
        st.json(summary)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Progress (visual feel)
    st.progress(sev_percent(severity))

    # Debug drawer
    with st.expander("🧪 Debug: Nearest Hospitals Checked"):
        if debug_checks:
            dbg = pd.DataFrame(debug_checks)
            if assigned_av:
                dbg["Allocated"] = dbg["Neighbor Hospital"].eq(assigned_av)
                dbg = dbg.sort_values(["Allocated","Remaining Beds/ICU"], ascending=[False,False])
            st.dataframe(dbg, use_container_width=True)
        else:
            st.write("No neighbor checks — assigned at selected hospital.")
