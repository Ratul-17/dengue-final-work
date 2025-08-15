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
st.set_page_config(page_title="Dengue Patient Allocation", page_icon="🏥", layout="centered")
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
# Utilities
# ===============================
def ensure_df(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if isinstance(df, pd.DataFrame):
        df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
        df.columns = [str(c).strip() for c in df.columns]
        return df
    return None

def autodetect(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return None
    for patt in candidates:
        for c in df.columns:
            if patt.lower() in str(c).lower():
                return c
    return None

def parse_date_series(s: pd.Series) -> pd.Series:
    def _one(x):
        if isinstance(x, (pd.Timestamp, datetime)):
            return pd.to_datetime(x).date()
        try:
            return pd.to_datetime(x, errors="coerce").date()
        except Exception:
            return pd.NaT
    return s.apply(_one)

def build_distance_matrix(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    if df.shape[1] > 2:
        df = df.set_index(df.columns[0])
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df.combine_first(df.T)
    raise ValueError("Could not interpret Location matrix format.")

STOPWORDS = {"hospital","medical","college","institute","university","center","centre","clinic","and"}
def norm_key(s: str) -> str:
    s = str(s).lower().strip()
    s = s.replace("&", " and ")
    s = re.sub(r"[^a-z0-9\s]", " ", s)
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
# Excel-based severity logic (NS1, IgM, IgG as 0/1)
# ===============================
def excel_based_severity(platelet, age, NS1, IgM, IgG):
    if platelet > 120000:
        H = 0
    elif platelet > 80000:
        H = 0.5
    elif platelet > 60000:
        H = 1
    elif platelet > 30000:
        H = 2
    elif platelet >= 10000:
        H = 3
    else:
        H = 4

    age_factor = 0.5 if (age < 15 or age > 60) else 0
    I = min(4, NS1 + IgM + IgG * 0.5 + age_factor + H)

    if I < 1:
        verdict = "Mild"
    elif I <= 2.5:
        verdict = "Moderate"
    elif I < 3:
        verdict = "Severe"
    else:
        verdict = "Very Severe"
    return verdict, I

def required_resource(severity: str) -> str:
    return "ICU" if severity in ("Severe", "Very Severe") else "General Bed"

# ===============================
# Load files directly from repo
# ===============================
pred_file_path = Path("Predicted dataset AIO.xlsx")
loc_file_path  = Path("Location matrix.xlsx")

if not pred_file_path.exists() or not loc_file_path.exists():
    st.error("Required data files not found in repository folder.")
    st.stop()

df_pred_raw = ensure_df(pd.read_excel(pred_file_path))
df_loc  = ensure_df(pd.read_excel(loc_file_path))

# ===============================
# Sidebar – Time granularity & interpolation
# ===============================
st.sidebar.header("⏱️ Time Resolution")
granularity = st.sidebar.selectbox("Time granularity", ["Daily", "Weekly", "Monthly"], index=0)
interp_method = st.sidebar.selectbox("Interpolation method (if expanding)", ["linear", "ffill"], index=0)

# ===============================
# Build availability with optional expansion
# ===============================
def build_availability_from_predictions(df_pred_raw: pd.DataFrame,
                                        granularity: str,
                                        interp_method: str) -> pd.DataFrame:
    """Returns MultiIndex availability: index (Hospital, Date) with _BedsAvail, _ICUAvail at chosen frequency."""
    df = df_pred_raw.copy()

    # Detect hospital & date-like columns
    hospital_col = autodetect(df, ["hospital","hospital name"])
    if not hospital_col:
        raise ValueError("Couldn't detect hospital column in predictions.")
    df["_Hospital"] = df[hospital_col].astype(str).str.strip()

    date_col  = autodetect(df, ["date"])
    year_col  = autodetect(df, ["year"])
    month_col = autodetect(df, ["month"])
    day_col   = autodetect(df, ["day"])

    if date_col:
        df["_Date"] = pd.to_datetime(df[date_col], errors="coerce")
    elif year_col and month_col:
        # if Day exists, use it; else month start
        if day_col:
            df["_Date"] = pd.to_datetime(dict(year=df[year_col], month=df[month_col], day=df[day_col]), errors="coerce")
        else:
            df["_Date"] = pd.to_datetime(df[year_col].astype(int).astype(str) + "-" +
                                         df[month_col].astype(int).astype(str) + "-01", errors="coerce")
    else:
        raise ValueError("Provide either a Date column or (Year & Month) in predictions.")

    df = df.dropna(subset=["_Date"])
    df["_Date"] = df["_Date"].dt.normalize()  # strip time

    # Availability columns (prefer predicted avail; else totals - occupied)
    # Try to auto-detect common names
    pred_normal_avail_col = autodetect(df, [
        "predicted normal beds available","normal beds available (pred)","beds available predicted","pred beds"
    ])
    pred_icu_avail_col    = autodetect(df, [
        "predicted icu beds available","icu beds available (pred)","icu available predicted","pred icu"
    ])
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
        raise ValueError("Could not find predicted availability columns or a totals/occupied fallback.")

    df["_BedsAvail"] = df["_BedsAvail"].fillna(0)
    df["_ICUAvail"]  = df["_ICUAvail"].fillna(0)

    # If user chose Monthly, we can just aggregate/sum within month (or take mean)
    if granularity == "Monthly":
        df["_Month"] = df["_Date"].dt.to_period("M").dt.to_timestamp()
        grouped = (df.groupby(["_Hospital","_Month"], as_index=False)[["_BedsAvail","_ICUAvail"]]
                     .mean())  # mean snapshot within month
        availability = (grouped.set_index(["_Hospital","_Month"]).sort_index())
        availability.index = availability.index.set_names(["_Hospital","_Date"])
        return availability

    # For Daily/Weekly: expand per hospital via resample + interpolation
    # Pivot to time series per hospital
    df_ts = df.set_index("_Date")
    beds_piv = df_ts.pivot_table(index="_Date", columns="_Hospital", values="_BedsAvail", aggfunc="mean")
    icu_piv  = df_ts.pivot_table(index="_Date", columns="_Hospital", values="_ICUAvail",  aggfunc="mean")

    # Build a complete date range across all data
    full_idx = pd.date_range(start=beds_piv.index.min(), end=beds_piv.index.max(), freq="D")
    beds_piv = beds_piv.reindex(full_idx)
    icu_piv  = icu_piv.reindex(full_idx)

    # Interpolate/propagate
    if interp_method == "linear":
        beds_piv = beds_piv.interpolate(method="time", limit_direction="both")
        icu_piv  = icu_piv.interpolate(method="time", limit_direction="both")
    else:  # ffill
        beds_piv = beds_piv.ffill().bfill()
        icu_piv  = icu_piv.ffill().bfill()

    # If Weekly requested, resample to weekly (use mean, Monday anchored)
    if granularity == "Weekly":
        beds_piv = beds_piv.resample("W-MON").mean()
        icu_piv  = icu_piv.resample("W-MON").mean()

    # Back to long MultiIndex (Hospital, Date)
    beds_long = beds_piv.stack(dropna=False).rename("_BedsAvail").to_frame()
    icu_long  = icu_piv.stack(dropna=False).rename("_ICUAvail").to_frame()
    long_df = beds_long.join(icu_long, how="outer").reset_index()
    long_df.columns = ["_Date","_Hospital","_BedsAvail","_ICUAvail"]

    # Clip negatives, fill NaNs
    long_df["_BedsAvail"] = long_df["_BedsAvail"].fillna(0).clip(lower=0)
    long_df["_ICUAvail"]  = long_df["_ICUAvail"].fillna(0).clip(lower=0)

    availability = (long_df.groupby(["_Hospital","_Date"], as_index=False)[["_BedsAvail","_ICUAvail"]].mean()
                         .set_index(["_Hospital","_Date"]).sort_index())
    return availability

# Build availability at chosen resolution
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
    # Note: availability is float after interpolation; we subtract reservations (ints) and floor
    base = 0.0
    key = (hospital, pd.to_datetime(date).normalize())
    if key in availability.index:
        base = float(availability.loc[key, "_ICUAvail" if bed_type == "ICU" else "_BedsAvail"])
    reserved = st.session_state["reservations"].get((hospital, key[1], bed_type), 0)
    # Floor the effective capacity to keep integers, never below 0
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
        if neighbor_dm == start_dm:
            continue
        neighbor_av = DM_TO_AV.get(neighbor_dm)
        rem = None
        if neighbor_av and ((neighbor_av, pd.to_datetime(date).normalize()) in availability.index):
            rem = get_remaining(neighbor_av, date, bed_key)
        checks.append({"Neighbor Hospital": neighbor_dm, "Remaining Beds/ICU": rem, "Distance (km)": dist})
        if rem and rem > 0:
            return neighbor_av, dist, None, checks
    return None, None, "No hospitals with vacancy found", checks

# ===============================
# UI – Patient inputs
# ===============================
all_dates = sorted(list(set([d for _, d in availability.index])))
min_d, max_d = min(all_dates), max(all_dates)

with st.form("allocation_form"):
    hospital_ui = st.selectbox("Hospital Name", HOSPITALS_UI)
    date_input  = st.date_input("Date", value=max_d, min_value=min_d, max_value=max_d)
    colA, colB = st.columns(2)
    with colA:
        age = st.number_input("Patient Age", min_value=0, max_value=120, value=25)
        weight = st.number_input("Weight (kg)", min_value=1.0, max_value=250.0, value=60.0)
        platelet = st.number_input("Platelet Count", min_value=0, value=120000, step=1000)
    with colB:
        ns1_val = st.selectbox("NS1 (0=Negative, 1=Positive)", [0, 1], index=0)
        igm_val = st.selectbox("IgM (0=Negative, 1=Positive)", [0, 1], index=0)
        igg_val = st.selectbox("IgG (0=Negative, 1=Positive)", [0, 1], index=0)
    submit = st.form_submit_button("🚑 Allocate Patient")

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

    st.subheader("📋 Allocation Result")
    st.json({
        "Date": str(pd.to_datetime(date_input).date()),
        "Time Granularity": granularity,
        "Interpolation": interp_method if granularity in ("Daily","Weekly") else "N/A",
        "Severity": severity,
        "Severity Score": round(float(score), 2),
        "Resource Needed": resource,
        "Hospital Tried": hospital_ui,
        "Available at Current Hospital": available_status,
        "Assigned Hospital": assigned_av,
        "Distance (km)": float(rerouted_distance) if rerouted_distance is not None else None,
        "Note": note
    })

    with st.expander("🧪 Debug: Nearest Hospitals Checked"):
        if debug_checks:
            st.dataframe(pd.DataFrame(debug_checks))
        else:
            st.write("No neighbor checks — patient assigned at selected hospital.")
