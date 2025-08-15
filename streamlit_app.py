import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Union

# -----------------------------
# App config
# -----------------------------
st.set_page_config(page_title="Dengue Patient Allocation", page_icon="🏥", layout="centered")
st.title("🏥 Dengue Patient Allocation System")

# Fixed list of hospitals (18)
HOSPITALS = [
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

# -----------------------------
# Helpers
# -----------------------------
def read_any(path_or_buf: Union[str, Path], sheet: Optional[str] = None):
    """Read CSV or Excel (single sheet)."""
    try:
        p = str(path_or_buf)
        if p.lower().endswith(".csv"):
            return pd.read_csv(path_or_buf)
        return pd.read_excel(path_or_buf, sheet_name=sheet)
    except Exception as e:
        st.error(f"Failed to read file: {path_or_buf}\n{e}")
        return None

def read_all_sheets_excel(path_or_buf: Union[str, Path]):
    """Read all sheets from an Excel (if provided)."""
    try:
        return pd.read_excel(path_or_buf, sheet_name=None)
    except Exception:
        return None

def ensure_df(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if isinstance(df, pd.DataFrame):
        df.columns = [str(c).strip() for c in df.columns]
    return df

def autodetect(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for patt in candidates:
        for c in df.columns:
            if patt.lower() in str(c).lower():
                return c
    return None

def normalize_hospital(x) -> str:
    return str(x).strip()

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
    """
    Accepts:
    - Long form: columns like [from, to, distance]
    - Wide form: first col = hospital, others = hospitals w/ numeric distances
    Returns symmetric matrix.
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # Long form
    long_from = autodetect(df, ["from", "source", "origin", "hospital_from", "start"])
    long_to   = autodetect(df, ["to", "dest", "destination", "hospital_to", "end"])
    long_dist = autodetect(df, ["distance", "km", "dist"])
    if all([long_from, long_to, long_dist]):
        piv = df[[long_from, long_to, long_dist]].copy()
        piv.columns = ["from", "to", "distance"]
        mat = piv.pivot_table(index="from", columns="to", values="distance", aggfunc="min")
        return mat.combine_first(mat.T)

    # Wide form
    if df.shape[1] > 2:
        df = df.set_index(df.columns[0])
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df.combine_first(df.T)

    raise ValueError("Could not interpret Location matrix format. Provide long or wide form.")

def to_bool01(x: int) -> bool:
    """Convert 0/1 int to boolean."""
    return bool(int(x))

# Severity via simple conservative rules (fallback)
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

def nearest_with_vacancy(dist_mat: pd.DataFrame, start_hospital: str, candidates: List[str]) -> Optional[str]:
    if start_hospital not in dist_mat.index:
        return None
    row = dist_mat.loc[start_hospital, candidates].dropna()
    if row.empty:
        return None
    return row.idxmin()

# -----------------------------
# Sidebar: Files
# -----------------------------
st.sidebar.header("📁 Data files")
pred_file = st.sidebar.file_uploader("Predictions (CSV/XLSX)", type=["csv", "xlsx"])
loc_file  = st.sidebar.file_uploader("Location matrix (CSV/XLSX)", type=["csv", "xlsx"])

# Also allow common filenames if running from a repo
def default_if_exists(name, current):
    return current or (Path(name) if Path(name).exists() else None)

pred_file = default_if_exists("Predicted dataset AIO.xlsx", pred_file)
pred_file = default_if_exists("ensemble_predictions_2026_2027_dynamic.xlsx", pred_file)
loc_file  = default_if_exists("Location matrix.xlsx", loc_file)
loc_file  = default_if_exists("distance matrix.csv", loc_file)

if not pred_file or not loc_file:
    st.error("Please provide a **Predictions** file and a **Location matrix**.")
    st.stop()

# -----------------------------
# Load / normalize predictions
# -----------------------------
df_pred = ensure_df(read_any(pred_file))
df_loc  = ensure_df(read_any(loc_file))
if df_pred is None or df_loc is None:
    st.stop()

# Hospital column (robust detection + manual fallback)
hospital_col = autodetect(df_pred, ["hospital", "hospital name", "hosp", "facility", "center", "centre", "clinic"])
if hospital_col is None:
    st.sidebar.warning("Couldn't detect Hospital column — please select one.")
    hospital_col = st.sidebar.selectbox("Select Hospital column", df_pred.columns.tolist())
else:
    hospital_col = st.sidebar.selectbox("Select Hospital column", df_pred.columns.tolist(),
                                        index=df_pred.columns.tolist().index(hospital_col))

df_pred["_Hospital"] = df_pred[hospital_col].astype(str).map(normalize_hospital)

# Date mapping (Date OR Year+Month)
date_col = autodetect(df_pred, ["date"])
year_col = autodetect(df_pred, ["year"])
month_col = autodetect(df_pred, ["month"])

if date_col:
    df_pred["_Date"] = parse_date_series(df_pred[date_col])
elif year_col and month_col:
    df_pred["_Date"] = pd.to_datetime(
        df_pred[year_col].astype(int).astype(str) + "-" + df_pred[month_col].astype(int).astype(str) + "-01",
        errors="coerce"
    ).dt.date
else:
    st.error("Provide either a **Date** column or both **Year** and **Month** in predictions.")
    st.stop()

# Prefer predicted availability columns; else fallback to Totals − Occupied
pred_normal_avail_col = autodetect(df_pred, ["predicted normal beds available", "normal beds available (pred)", "beds available predicted", "pred beds"])
pred_icu_avail_col    = autodetect(df_pred, ["predicted icu beds available", "icu beds available (pred)", "icu available predicted", "pred icu"])

beds_total_col  = autodetect(df_pred, ["beds total", "total beds"])
icu_total_col   = autodetect(df_pred, ["icu beds total", "total icu"])
beds_occ_col    = autodetect(df_pred, ["beds occupied", "occupied beds"])
icu_occ_col     = autodetect(df_pred, ["icu beds occupied", "occupied icu"])

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

# -----------------------------
# Distance matrix
# -----------------------------
try:
    dist_mat = build_distance_matrix(df_loc)
    dist_mat.index = dist_mat.index.map(normalize_hospital)
    dist_mat.columns = dist_mat.columns.map(normalize_hospital)
except Exception as e:
    st.error(f"Location matrix error: {e}")
    st.stop()

# -----------------------------
# Session reservations
# -----------------------------
if "reservations" not in st.session_state:
    st.session_state["reservations"] = {}  # (hospital, date, type) -> count

def get_remaining(hospital: str, date, bed_type: str) -> int:
    key = (hospital, date)
    base = 0
    if key in availability.index:
        base = int(availability.loc[key, "_ICUAvail" if bed_type == "ICU" else "_BedsAvail"])
    reserved = st.session_state["reservations"].get((hospital, date, bed_type), 0)
    return max(0, base - reserved)

def reserve_bed(hospital: str, date, bed_type: str, n: int = 1):
    k = (hospital, date, bed_type)
    st.session_state["reservations"][k] = st.session_state["reservations"].get(k, 0) + n

# -----------------------------
# UI – Patient Inputs (as requested)
# -----------------------------
with st.form("allocation_form"):
    st.subheader("🔍 Patient Information")

    hospital = st.selectbox("Hospital Name", HOSPITALS)
    # restrict selectable dates to those present in predictions for the chosen hospital
    hosp_dates = sorted({d for (h, d) in availability.index if h == hospital})
    if not hosp_dates:
        st.error("Selected hospital has no dates in predictions. Choose another or check your dataset.")
        st.stop()
    date_input = st.date_input("Date", value=max(hosp_dates), min_value=min(hosp_dates), max_value=max(hosp_dates))

    colA, colB = st.columns(2)
    with colA:
        age = st.number_input("Patient Age (years)", min_value=0, max_value=120, value=25)
        weight = st.number_input("Weight (kg)", min_value=1.0, max_value=250.0, value=60.0, step=0.5)
        platelet = st.number_input("Platelet level (/µL)", min_value=0, value=120000, step=1000)
    with colB:
        igg01 = st.selectbox("IgG (0=Negative, 1=Positive)", [0, 1], index=0)
        ign01 = st.selectbox("igN (treated as IgM) (0=Negative, 1=Positive)", [0, 1], index=0)
        ns101 = st.selectbox("NS1 (0=Negative, 1=Positive)", [0, 1], index=0)

    submit = st.form_submit_button("🚑 Allocate")

if submit:
    # Convert 0/1 to booleans
    igg = to_bool01(igg01)
    igm = to_bool01(ign01)  # treating igN as IgM per your instruction
    ns1 = to_bool01(ns101)

    # Determine severity (you can replace this with a data-driven rule if you share an Allocation dataset)
    severity = simple_severity(platelet=float(platelet), igg=igg, igm=igm, ns1=ns1)
    resource = required_resource(severity)
    bed_key = "ICU" if resource == "ICU" else "Normal"

    # Check vacancy at selected hospital/date
    remaining_here = get_remaining(hospital, date_input, bed_key)

    assigned_hospital = hospital
    note = ""
    rerouted_distance = None

    if remaining_here > 0:
        note = "Assigned at selected hospital"
    else:
        # Find nearest with vacancy
        candidates = [h for h in HOSPITALS if get_remaining(h, date_input, bed_key) > 0]
        reroute = nearest_with_vacancy(dist_mat, hospital, candidates)
        if reroute:
            assigned_hospital = reroute
            try:
                rerouted_distance = dist_mat.loc[hospital, reroute]
            except Exception:
                rerouted_distance = None
            note = f"Rerouted to nearest hospital with {resource}"
        else:
            note = "No hospitals with vacancy found for selected date and resource"

    # Reserve upon success
    if "No hospitals" not in note:
        reserve_bed(assigned_hospital, date_input, bed_key, n=1)

    st.subheader("📋 Allocation Result")
    result = {
        "Date": pd.to_datetime(date_input).strftime("%Y-%m-%d"),
        "Severity": severity,
        "Resource Needed": resource,
        "Hospital Tried": hospital,
        "Available at Current Hospital": "Yes" if assigned_hospital == hospital and note.startswith("Assigned") else "No",
        "Assigned Hospital": assigned_hospital,
    }
    if rerouted_distance is not None:
        result["Distance (matrix units)"] = float(rerouted_distance)
    result["Note"] = note
    st.json(result)
