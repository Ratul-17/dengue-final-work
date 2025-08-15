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
# Utility functions
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
# Excel-based severity logic (using NS1, IgM, IgG)
# ===============================
def excel_based_severity(platelet, age, NS1, IgM, IgG):
    # Column H logic
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
    # Column I logic with renamed C=NS1, E=IgM, D=IgG
    I = min(4, NS1 + IgM + IgG * 0.5 + age_factor + H)

    if I < 1:
        verdict = "Mild"
    elif I < 2:
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

df_pred = ensure_df(pd.read_excel(pred_file_path))
df_loc  = ensure_df(pd.read_excel(loc_file_path))

# Process predictions
hospital_col = autodetect(df_pred, ["hospital","hospital name"])
df_pred["_Hospital"] = df_pred[hospital_col].astype(str).str.strip()
date_col = autodetect(df_pred, ["date"])
df_pred["_Date"] = parse_date_series(df_pred[date_col])
beds_total_col  = autodetect(df_pred, ["beds total"])
icu_total_col   = autodetect(df_pred, ["icu beds total"])
beds_occ_col    = autodetect(df_pred, ["beds occupied"])
icu_occ_col     = autodetect(df_pred, ["icu beds occupied"])
df_pred["_BedsAvail"] = (pd.to_numeric(df_pred[beds_total_col], errors="coerce") -
                         pd.to_numeric(df_pred[beds_occ_col], errors="coerce")).fillna(0).astype(int)
df_pred["_ICUAvail"]  = (pd.to_numeric(df_pred[icu_total_col], errors="coerce") -
                         pd.to_numeric(df_pred[icu_occ_col], errors="coerce")).fillna(0).astype(int)

availability = (
    df_pred.groupby(["_Hospital","_Date"], as_index=False)[["_BedsAvail","_ICUAvail"]].sum()
    .set_index(["_Hospital","_Date"]).sort_index()
)

# Process location matrix
dist_mat = build_distance_matrix(df_loc)
DM_TO_AV, UI_TO_DM, UI_TO_AV = build_name_maps(availability, dist_mat, HOSPITALS_UI)

# ===============================
# Allocation helpers
# ===============================
if "reservations" not in st.session_state:
    st.session_state["reservations"] = {}

def get_remaining(hospital: str, date, bed_type: str) -> int:
    base = 0
    key = (hospital, date)
    if key in availability.index:
        base = int(availability.loc[key, "_ICUAvail" if bed_type == "ICU" else "_BedsAvail"])
    reserved = st.session_state["reservations"].get((hospital, date, bed_type), 0)
    return max(0, base - reserved)

def reserve_bed(hospital: str, date, bed_type: str, n: int = 1):
    k = (hospital, date, bed_type)
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
        if neighbor_av and ((neighbor_av, date) in availability.index):
            rem = get_remaining(neighbor_av, date, bed_key)
        checks.append({"Neighbor Hospital": neighbor_dm, "Remaining Beds/ICU": rem, "Distance (km)": dist})
        if rem and rem > 0:
            return neighbor_av, dist, None, checks
    return None, None, "No hospitals with vacancy found", checks

# ===============================
# UI – Patient inputs
# ===============================
all_dates = sorted(list(set([d for _, d in availability.index])))
with st.form("allocation_form"):
    hospital_ui = st.selectbox("Hospital Name", HOSPITALS_UI)
    date_input  = st.date_input("Date", value=max(all_dates), min_value=min(all_dates), max_value=max(all_dates))
    age = st.number_input("Patient Age", min_value=0, max_value=120, value=25)
    weight = st.number_input("Weight (kg)", min_value=1.0, max_value=250.0, value=60.0)
    platelet = st.number_input("Platelet Count", min_value=0, value=120000, step=1000)
    ns1_val = st.selectbox("NS1 (0=Negative, 1=Positive)", [0, 1], index=0)
    igm_val = st.selectbox("IgM (0=Negative, 1=Positive)", [0, 1], index=0)
    igg_val = st.selectbox("IgG (0=Negative, 1=Positive)", [0, 1], index=0)
    submit = st.form_submit_button("🚑 Allocate Patient")

if submit:
    severity, score = excel_based_severity(platelet, age, ns1_val, igm_val, igg_val)
    resource = required_resource(severity)
    bed_key  = "ICU" if resource == "ICU" else "Normal"
    start_av = UI_TO_AV.get(hospital_ui) or hospital_ui
    remaining_here = get_remaining(start_av, date_input, bed_key)

    debug_checks = []
    if remaining_here > 0:
        assigned_av, rerouted_distance, note = start_av, None, "Assigned at selected hospital"
        available_status = "Yes"
    else:
        available_status = "No vacancy available here"
        assigned_av, rerouted_distance, err, debug_checks = find_reroute_nearest_first(hospital_ui, date_input, bed_key)
        note = f"Rerouted to {assigned_av}" if assigned_av else err

    if assigned_av:
        reserve_bed(assigned_av, date_input, bed_key, 1)

    st.json({
        "Date": str(date_input),
        "Severity": severity,
        "Severity Score": score,
        "Resource Needed": resource,
        "Hospital Tried": hospital_ui,
        "Available at Current Hospital": available_status,
        "Assigned Hospital": assigned_av,
        "Distance (km)": rerouted_distance,
        "Note": note
    })

    with st.expander("🧪 Debug: Nearest Hospitals Checked"):
        if debug_checks:
            st.dataframe(pd.DataFrame(debug_checks))
        else:
            st.write("No neighbor checks — patient assigned at selected hospital.")
