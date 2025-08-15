import streamlit as st
import pandas as pd
import numpy as np
import re
from difflib import get_close_matches
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Union

# ===============================
# App config
# ===============================
st.set_page_config(page_title="Dengue Patient Allocation", page_icon="🏥", layout="centered")
st.title("🏥 Integrated Hospital Dengue Patient Allocation System")

# Fixed UI list (18 hospitals)
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

def guess_header_row(df_no_header: pd.DataFrame, scan_rows: int = 15) -> int:
    scan = min(len(df_no_header), scan_rows)
    keywords = ["hospital", "facility", "centre", "center", "date", "year", "month", "bed", "icu"]
    best_idx, best_score = 0, -1
    for i in range(scan):
        row = df_no_header.iloc[i].astype(str).str.lower()
        score = row.notna().sum()
        if any(any(kw in str(val) for val in row) for kw in keywords):
            score += 10
        if score > best_score:
            best_score, best_idx = score, i
    return int(best_idx)

# ---------- Predictions loader ----------
def read_predictions(file_obj_or_path: Union[str, Path, "UploadedFile"]) -> Optional[pd.DataFrame]:
    name = str(getattr(file_obj_or_path, "name", file_obj_or_path)).lower()
    if name.endswith(".csv"):
        try:
            df = pd.read_csv(file_obj_or_path)
        except Exception:
            try:
                if hasattr(file_obj_or_path, "seek"): file_obj_or_path.seek(0)
                df = pd.read_csv(file_obj_or_path, sep=None, engine="python")
            except Exception:
                if hasattr(file_obj_or_path, "seek"): file_obj_or_path.seek(0)
                df = pd.read_csv(file_obj_or_path, encoding="latin1")
        return ensure_df(df)

    xl = pd.ExcelFile(file_obj_or_path)
    sizes = {}
    for s in xl.sheet_names:
        try:
            df_sample = pd.read_excel(xl, sheet_name=s, nrows=25, header=None)
            sizes[s] = int(df_sample.dropna(how="all").shape[0])
        except Exception:
            sizes[s] = 0
    default_sheet = sorted(sizes.items(), key=lambda x: x[1], reverse=True)[0][0] if sizes else xl.sheet_names[0]
    sheet = st.sidebar.selectbox("Predictions sheet", xl.sheet_names, index=xl.sheet_names.index(default_sheet))

    df_no_header = pd.read_excel(xl, sheet_name=sheet, header=None).dropna(how="all", axis=1)
    guess = guess_header_row(df_no_header)
    header_row = st.sidebar.number_input("Predictions header row (0-index)", min_value=0,
                                         max_value=max(0, len(df_no_header)-1), value=int(guess), step=1)
    df = pd.read_excel(xl, sheet_name=sheet, header=int(header_row))
    return ensure_df(df)

# ---------- Location loader ----------
def read_location(file_obj_or_path: Union[str, Path, "UploadedFile"]) -> Optional[pd.DataFrame]:
    name = str(getattr(file_obj_or_path, "name", file_obj_or_path)).lower()
    if name.endswith(".csv"):
        try:
            df = pd.read_csv(file_obj_or_path)
        except Exception:
            try:
                if hasattr(file_obj_or_path, "seek"): file_obj_or_path.seek(0)
                df = pd.read_csv(file_obj_or_path, sep=None, engine="python")
            except Exception:
                if hasattr(file_obj_or_path, "seek"): file_obj_or_path.seek(0)
                df = pd.read_csv(file_obj_or_path, encoding="latin1")
        return ensure_df(df)

    xl = pd.ExcelFile(file_obj_or_path)
    sizes = {}
    for s in xl.sheet_names:
        try:
            df_sample = pd.read_excel(xl, sheet_name=s, nrows=25, header=None)
            sizes[s] = int(df_sample.dropna(how="all").shape[0])
        except Exception:
            sizes[s] = 0
    default_sheet = sorted(sizes.items(), key=lambda x: x[1], reverse=True)[0][0] if sizes else xl.sheet_names[0]
    sheet = st.sidebar.selectbox("Location sheet", xl.sheet_names, index=xl.sheet_names.index(default_sheet))

    df_no_header = pd.read_excel(xl, sheet_name=sheet, header=None).dropna(how="all", axis=1)
    guess = guess_header_row(df_no_header)
    header_row = st.sidebar.number_input("Location header row (0-index)", min_value=0,
                                         max_value=max(0, len(df_no_header)-1), value=int(guess), step=1)
    df = pd.read_excel(xl, sheet_name=sheet, header=int(header_row))
    return ensure_df(df)

def build_distance_matrix(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    if df.shape[1] > 2:
        df = df.set_index(df.columns[0])
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df.combine_first(df.T)
    raise ValueError("Could not interpret Location matrix format.")

# ---------- Name normalization ----------
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
    ui_by_key    = {norm_key(u): u for u in ui_names}

    dm_to_av = {}
    for d in dm_names:
        kd = norm_key(d)
        if kd in avail_by_key:
            dm_to_av[d] = avail_by_key[kd]
            continue
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

# ---------- Excel-based severity ----------
def excel_based_severity(platelet, age, C, E, D):
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
    I = min(4, C + E + D * 0.5 + age_factor + H)

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
# Sidebar: files
# ===============================
st.sidebar.header("📁 Data files")
pred_file = st.sidebar.file_uploader("Predictions (CSV/XLSX)", type=["csv","xlsx"])
loc_file  = st.sidebar.file_uploader("Location matrix (CSV/XLSX)", type=["csv","xlsx"])

if not pred_file or not loc_file:
    st.warning("Please upload both Predictions file and Location matrix to proceed.")
    st.stop()

# ===============================
# Load data
# ===============================
df_pred = read_predictions(pred_file)
df_loc  = read_location(loc_file)
if df_pred is None or df_pred.empty or df_loc is None or df_loc.empty:
    st.error("Error loading files.")
    st.stop()

hospital_col = autodetect(df_pred, ["hospital","hospital name"])
df_pred["_Hospital"] = df_pred[hospital_col].astype(str).str.strip()

date_col = autodetect(df_pred, ["date"])
if date_col:
    df_pred["_Date"] = parse_date_series(df_pred[date_col])
df_pred = df_pred.dropna(subset=["_Date"])

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
    if not start_dm or start_dm not in dist_mat.index:
        return None, None, "Hospital not found in distance matrix", []
    row = dist_mat.loc[start_dm].astype(float).dropna().sort_values()
    checks = []
    for neighbor_dm, dist in row.items():
        if neighbor_dm == start_dm:
            continue
        neighbor_av = DM_TO_AV.get(neighbor_dm)
        if neighbor_av and ((neighbor_av, date) in availability.index):
            rem = get_remaining(neighbor_av, date, bed_key)
            checks.append({"neighbor": neighbor_dm, "remaining": rem})
            if rem > 0:
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
    C_val = st.number_input("C value", min_value=0.0, value=0.0, step=0.1)
    E_val = st.number_input("E value", min_value=0.0, value=0.0, step=0.1)
    D_val = st.number_input("D value", min_value=0.0, value=0.0, step=0.1)
    submit = st.form_submit_button("🚑 Allocate Patient")

if submit:
    severity, score = excel_based_severity(platelet, age, C_val, E_val, D_val)
    resource = required_resource(severity)
    bed_key  = "ICU" if resource == "ICU" else "Normal"
    start_av = UI_TO_AV.get(hospital_ui) or hospital_ui
    remaining_here = get_remaining(start_av, date_input, bed_key)

    if remaining_here > 0:
        assigned_av, rerouted_distance, note = start_av, None, "Assigned at selected hospital"
    else:
        assigned_av, rerouted_distance, err, checks = find_reroute_nearest_first(hospital_ui, date_input, bed_key)
        note = f"Rerouted to {assigned_av}" if assigned_av else err

    if assigned_av:
        reserve_bed(assigned_av, date_input, bed_key, 1)

    st.json({
        "Date": str(date_input),
        "Severity": severity,
        "Severity Score": score,
        "Resource Needed": resource,
        "Assigned Hospital": assigned_av,
        "Distance (km)": rerouted_distance,
        "Note": note
    })
