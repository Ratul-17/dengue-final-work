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
st.title("🏥 Dengue Patient Allocation System")

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

# ---------- Predictions loader (robust CSV/XLSX) ----------
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

# ---------- Location loader (now with XLSX sheet + header picker) ----------
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

    # Excel: choose sheet + header row for location matrix too
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

    # Long format
    long_from = autodetect(df, ["from", "source", "origin", "hospital_from", "start"])
    long_to   = autodetect(df, ["to", "dest", "destination", "hospital_to", "end"])
    long_dist = autodetect(df, ["distance", "km", "dist"])
    if all([long_from, long_to, long_dist]):
        piv = df[[long_from, long_to, long_dist]].copy()
        piv.columns = ["from", "to", "distance"]
        mat = piv.pivot_table(index="from", columns="to", values="distance", aggfunc="min")
        return mat.combine_first(mat.T)

    # Wide format
    if df.shape[1] > 2:
        df = df.set_index(df.columns[0])
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df.combine_first(df.T)

    raise ValueError("Could not interpret Location matrix format. Provide long or wide form.")

# ---------- Name normalization / fuzzy matching ----------
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

    # distance -> availability
    dm_to_av = {}
    for d in dm_names:
        kd = norm_key(d)
        if kd in avail_by_key:
            dm_to_av[d] = avail_by_key[kd]
            continue
        m = get_close_matches(kd, list(avail_by_key.keys()), n=1, cutoff=0.6)
        dm_to_av[d] = avail_by_key[m[0]] if m else None

    # ui -> dist / avail
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

# ---------- Severity & resource ----------
def to_bool01(x: int) -> bool:
    return bool(int(x))

def simple_severity(platelet: float, igg: bool, igm: bool, ns1: bool) -> str:
    if platelet is not None and platelet < 20000:
        return "Very Severe"
    if (platelet is not None and platelet < 50000) or (ns1 and platelet is not None and platelet < 80000):
        return "Severe"
    if ns1 or igg or igm:
        return "Moderate"
    return "Normal"

def required_resource(severity: str) -> str:
    return "ICU" if severity in ("Severe", "Very Severe") else "General Bed"

# ===============================
# Sidebar: files
# ===============================
st.sidebar.header("📁 Data files")
pred_file = st.sidebar.file_uploader("Predictions (CSV/XLSX)", type=["csv","xlsx"])
loc_file  = st.sidebar.file_uploader("Location matrix (CSV/XLSX)", type=["csv","xlsx"])

def default_if_exists(name, current):
    return current or (Path(name) if Path(name).exists() else None)

pred_file = default_if_exists("Predicted dataset AIO.xlsx", pred_file)
pred_file = default_if_exists("ensemble_predictions_2026_2027_dynamic.xlsx", pred_file)
loc_file  = default_if_exists("Location matrix.xlsx", loc_file)
loc_file  = default_if_exists("distance matrix.csv", loc_file)

if not pred_file:
    st.error("Please provide a Predictions file.")
    st.stop()
if not loc_file:
    st.error("Please provide a Location matrix.")
    st.stop()

# ===============================
# Load data
# ===============================
df_pred = read_predictions(pred_file)
df_loc  = read_location(loc_file)

if df_pred is None or df_pred.empty:
    st.error("Predictions file could not be read or is empty.")
    st.stop()
if df_loc is None or df_loc.empty:
    st.error("Location matrix could not be read or is empty.")
    st.stop()

with st.expander("🔎 Detected columns (debug)"):
    st.write("**Predictions columns:**", list(df_pred.columns))
    st.write("**Location columns:**", list(df_loc.columns))

# ===============================
# Map predictions -> availability
# ===============================
hospital_col = autodetect(df_pred, ["hospital","hospital name","hosp","facility","center","centre","clinic"])
if hospital_col is None:
    st.sidebar.warning("Couldn't detect Hospital column — select it.")
    hospital_col = st.sidebar.selectbox("Select Hospital column", df_pred.columns.tolist())
else:
    hospital_col = st.sidebar.selectbox("Select Hospital column", df_pred.columns.tolist(),
                                        index=df_pred.columns.tolist().index(hospital_col))
df_pred["_Hospital"] = df_pred[hospital_col].astype(str).str.strip()

date_col  = autodetect(df_pred, ["date"])
year_col  = autodetect(df_pred, ["year"])
month_col = autodetect(df_pred, ["month"])

if date_col:
    df_pred["_Date"] = parse_date_series(df_pred[date_col])
elif year_col and month_col:
    df_pred["_Date"] = pd.to_datetime(
        df_pred[year_col].astype(int).astype(str) + "-" + df_pred[month_col].astype(int).astype(str) + "-01",
        errors="coerce"
    ).dt.date
else:
    st.error("Provide either a Date column or both Year and Month.")
    st.stop()

pred_normal_avail_col = autodetect(df_pred, [
    "predicted normal beds available","normal beds available (pred)","beds available predicted","pred beds"
])
pred_icu_avail_col    = autodetect(df_pred, [
    "predicted icu beds available","icu beds available (pred)","icu available predicted","pred icu"
])
beds_total_col  = autodetect(df_pred, ["beds total","total beds"])
icu_total_col   = autodetect(df_pred, ["icu beds total","total icu"])
beds_occ_col    = autodetect(df_pred, ["beds occupied","occupied beds"])
icu_occ_col     = autodetect(df_pred, ["icu beds occupied","occupied icu"])

if pred_normal_avail_col and pred_icu_avail_col:
    df_pred["_BedsAvail"] = pd.to_numeric(df_pred[pred_normal_avail_col], errors="coerce").fillna(0).astype(int)
    df_pred["_ICUAvail"]  = pd.to_numeric(df_pred[pred_icu_avail_col], errors="coerce").fillna(0).astype(int)
elif all([beds_total_col, beds_occ_col, icu_total_col, icu_occ_col]):
    df_pred["_BedsAvail"] = (pd.to_numeric(df_pred[beds_total_col], errors="coerce") -
                             pd.to_numeric(df_pred[beds_occ_col], errors="coerce")).fillna(0).astype(int)
    df_pred["_ICUAvail"]  = (pd.to_numeric(df_pred[icu_total_col], errors="coerce") -
                             pd.to_numeric(df_pred[icu_occ_col], errors="coerce")).fillna(0).astype(int)
else:
    st.error("Could not find predicted availability columns or fallback (Totals & Occupied).")
    st.stop()

df_pred = df_pred.dropna(subset=["_Date"])
availability = (
    df_pred.groupby(["_Hospital","_Date"], as_index=False)[["_BedsAvail","_ICUAvail"]].sum()
    .set_index(["_Hospital","_Date"]).sort_index()
)

# ===============================
# Build distance matrix
# ===============================
dist_mat = build_distance_matrix(df_loc)
dist_mat.index = dist_mat.index.map(lambda x: str(x).strip())
dist_mat.columns = dist_mat.columns.map(lambda x: str(x).strip())

# ===============================
# Name maps (UI ↔ distance ↔ predictions)
# ===============================
DM_TO_AV, UI_TO_DM, UI_TO_AV = build_name_maps(availability, dist_mat, HOSPITALS_UI)

# ===============================
# Reservations
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

# ===============================
# Nearest → next-nearest rerouting
# ===============================
def find_reroute_nearest_first(start_ui_name: str, date, bed_key: str):
    """
    Walk neighbors by distance:
      1) map UI name -> distance-matrix row (fuzzy)
      2) for each neighbor (sorted asc by distance):
         - neighbor DM name -> availability name (fuzzy)
         - require (hospital, date) in predictions
         - require remaining capacity for bed_key
    Returns: (assigned_av_name, distance_float, error_message_or_None, debug_checks)
    """
    checks = []

    start_dm = UI_TO_DM.get(start_ui_name)
    if not start_dm or start_dm not in dist_mat.index:
        return None, None, "Hospital not found in distance matrix", checks

    row = dist_mat.loc[start_dm].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if row.empty:
        return None, None, "No neighbors with distances in matrix", checks

    # sort nearest → farthest
    row = row.sort_values(kind="mergesort")

    for neighbor_dm, dist in row.items():
        if str(neighbor_dm).strip() == str(start_dm).strip():
            continue  # skip self
        neighbor_av = DM_TO_AV.get(neighbor_dm)
        rem = None
        has_rows = False
        if neighbor_av is not None:
            has_rows = ((neighbor_av, date) in availability.index)
            if has_rows:
                rem = get_remaining(neighbor_av, date, bed_key)

        checks.append({
            "neighbor_dm": neighbor_dm,
            "mapped_av": neighbor_av,
            "distance": float(dist),
            "date_has_rows": bool(has_rows),
            "remaining": int(rem) if rem is not None else None
        })

        if neighbor_av is None:
            continue
        if not has_rows:
            continue
        if rem and rem > 0:
            return neighbor_av, float(dist), None, checks

    return None, None, "No hospitals with vacancy found for selected date and resource", checks

# ===============================
# UI – Patient inputs
# ===============================
all_dates = sorted(list(set([d for _, d in availability.index])))
if not all_dates:
    st.error("No dates found in predictions after processing.")
    st.stop()

with st.form("allocation_form"):
    st.subheader("🔍 Patient Information")
    hospital_ui = st.selectbox("Hospital Name", HOSPITALS_UI)
    date_input  = st.date_input("Date", value=max(all_dates), min_value=min(all_dates), max_value=max(all_dates))

    colA, colB = st.columns(2)
    with colA:
        age = st.number_input("Patient Age (years)", min_value=0, max_value=120, value=25)
        weight = st.number_input("Weight (kg)", min_value=1.0, max_value=250.0, value=60.0, step=0.5)
        platelet = st.number_input("Platelete level (/µL)", min_value=0, value=120000, step=1000)
    with colB:
        igg01 = st.selectbox("IgG (0=Negative, 1=Positive)", [0, 1], index=0)
        ign01 = st.selectbox("igN (treated as IgM) (0=Negative, 1=Positive)", [0, 1], index=0)
        ns101 = st.selectbox("NS1 (0=Negative, 1=Positive)", [0, 1], index=0)

    submit = st.form_submit_button("🚑 Allocate")

if submit:
    igg = bool(int(igg01))
    igm = bool(int(ign01))
    ns1 = bool(int(ns101))

    severity = simple_severity(platelet=float(platelet), igg=igg, igm=igm, ns1=ns1)
    resource = required_resource(severity)      # "ICU" or "General Bed"
    bed_key  = "ICU" if resource == "ICU" else "Normal"

    start_av = (UI_TO_AV.get(hospital_ui) or hospital_ui)
    remaining_here = get_remaining(start_av, date_input, bed_key)

    assigned_av = None
    rerouted_distance = None
    debug_checks = []

    if remaining_here > 0 and (start_av, date_input) in availability.index:
        assigned_av = start_av
        note = "Assigned at selected hospital"
    else:
        assigned_av, rerouted_distance, err, debug_checks = find_reroute_nearest_first(
            start_ui_name=hospital_ui,
            date=date_input,
            bed_key=bed_key,
        )
        if assigned_av:
            note = f"Rerouted to nearest hospital with {resource}"
        else:
            note = err or "No hospitals with vacancy found for selected date and resource"
            assigned_av = start_av

    if assigned_av and "No hospitals" not in note:
        reserve_bed(assigned_av, date_input, bed_key, n=1)

    st.subheader("📋 Allocation Result")
    result = {
        "Date": pd.to_datetime(date_input).strftime("%Y-%m-%d"),
        "Severity": severity,
        "Resource Needed": resource,
        "Hospital Tried": hospital_ui,
        "Available at Current Hospital": "Yes" if (assigned_av == start_av and note.startswith("Assigned")) else "No",
        "Assigned Hospital": assigned_av,
    }
    if rerouted_distance is not None:
        result["Distance (in Km)"] = float(rerouted_distance)
    result["Note"] = note
    st.json(result)

    with st.expander("🧪 Debug: neighbors checked"):
        if debug_checks:
            st.dataframe(pd.DataFrame(debug_checks))
        else:
            st.write("No neighbor checks (assigned at selected hospital).")
