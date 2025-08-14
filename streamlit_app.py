
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Tuple, Optional, List

st.set_page_config(page_title="Dengue Bed Allocation", page_icon="🏥", layout="wide")

st.title("🏥 Dengue Bed & ICU Allocation (Sep–Dec 2024)")

with st.expander("ℹ️ How this works", expanded=False):
    st.markdown("""
    - **Inputs**: Upload three Excel files — **Predicted dataset AIO**, **Allocation dataset**, and **Location matrix**.
    - **Severity**: The app will calculate severity using the **Allocation dataset**:
        - If that file has a column named **`equation`** (or **`severity_equation`**) that is a valid expression using variables `platelet`, `igg`, `igm`, `ns1`, `age`, `weight`, we evaluate it.
        - If it yields a numeric **score**, thresholds can be supplied below (or auto-detected if present in the file as `mild_max`, `moderate_max`, `severe_max`).
        - If no equation is found, we use a conservative fallback:
            - **Very Severe** if `platelet < 20000`
            - **Severe** if `platelet < 50000` or (`ns1` is True and `platelet < 80000`)
            - **Moderate** otherwise if `ns1` is True or (`igg` or `igm` is True)
            - **Mild** otherwise
        - **Severe/Very Severe** → ICU required; **Mild/Moderate** → Normal bed.
    - **Beds**: For each (Hospital, Date), we use *predicted* availability columns you choose in the sidebar.
    - **Routing**: If no vacancy at the chosen hospital, we'll route to the **nearest** hospital with vacancy using distances from the **Location matrix**.
    - **State**: Each successful allocation **reserves** a bed for that (Hospital, Date) in the app session.
    """)

# ----------------------
# Utility helpers
# ----------------------

@st.cache_data(show_spinner=False)
def read_excel_any(path_or_buffer, sheet=None):
    try:
        return pd.read_excel(path_or_buffer, sheet_name=sheet)
    except Exception as e:
        st.error(f"Failed to read Excel: {e}")
        return None

def autodetect_columns(df: pd.DataFrame, candidates: List[str]):
    """Return the first matching column by case-insensitive partial match."""
    cols_lower = {c.lower(): c for c in df.columns}
    for patt in candidates:
        for c in df.columns:
            if patt.lower() in c.lower():
                return c
    return None

def build_distance_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Accept either:
    1) Long form: columns like [from, to, distance_km]
    2) Wide form: first column = hospital, others = hospitals with numeric distances
    Returns a square DataFrame indexed/columns by hospital names.
    """
    dfc = df.copy()
    # Try long form detection
    long_from = autodetect_columns(dfc, ["from", "source", "origin", "hospital_from", "start"])
    long_to = autodetect_columns(dfc, ["to", "dest", "destination", "hospital_to", "end"])
    long_dist = autodetect_columns(dfc, ["distance", "km", "dist"])
    if all([long_from, long_to, long_dist]):
        piv = dfc[[long_from, long_to, long_dist]].copy()
        piv.columns = ["from", "to", "distance"]
        mat = piv.pivot_table(index="from", columns="to", values="distance", aggfunc="min")
        # make symmetric if needed
        mat2 = mat.combine_first(mat.T)
        return mat2
    # Try wide form
    if dfc.shape[1] > 2:
        # assume first column is hospital name/id
        dfc = dfc.set_index(dfc.columns[0])
        # coerce numeric
        for c in dfc.columns:
            dfc[c] = pd.to_numeric(dfc[c], errors="coerce")
        # make symmetric
        mat2 = dfc.combine_first(dfc.T)
        return mat2
    raise ValueError("Could not interpret Location matrix format. Provide either long or wide distance matrix.")

def safe_eval_equation(eq: str, vars_dict: Dict) -> Optional[float]:
    """
    Evaluate an equation string safely with restricted globals.
    Returns numeric score or a string severity if the expression yields text.
    Allowed names: basic math + numpy via np.
    Variables available: platelet, igg, igm, ns1, age, weight.
    """
    allowed_builtins = {}
    safe_globals = {"__builtins__": allowed_builtins, "np": np, "min": min, "max": max, "abs": abs, "round": round}
    try:
        res = eval(eq, safe_globals, vars_dict)
        return res
    except Exception as e:
        st.warning(f"Equation eval failed: {e}")
        return None

def severity_from_equation(df_alloc: pd.DataFrame, patient_vars: Dict, thresholds: Dict) -> Tuple[str, Optional[float]]:
    """
    Try to find an equation and compute severity.
    - If result is string among {mild, moderate, severe, very severe} (any case), return it.
    - If numeric, map to severity using thresholds.
    """
    cand_cols = [c for c in df_alloc.columns if c.lower() in ("equation", "severity_equation")]
    eq = None
    if cand_cols:
        # pick the first non-null equation
        ser = df_alloc[cand_cols[0]].dropna().astype(str)
        if not ser.empty:
            eq = ser.iloc[0]
    if not eq:
        return ("__fallback__", None)

    res = safe_eval_equation(eq, patient_vars)
    if res is None:
        return ("__fallback__", None)

    if isinstance(res, str):
        s = res.strip().lower()
        mapping = {"mild": "Mild", "moderate": "Moderate", "severe": "Severe", "very severe": "Very Severe", "very_severe": "Very Severe"}
        if s in mapping:
            return (mapping[s], None)
        # if string but not recognized, fallback
        return ("__fallback__", None)

    # Numeric score → thresholds
    if isinstance(res, (int, float, np.integer, np.floating)):
        score = float(res)
        mild_max = thresholds.get("mild_max", 0.25)
        moderate_max = thresholds.get("moderate_max", 0.5)
        severe_max = thresholds.get("severe_max", 0.8)
        if score <= mild_max:
            return ("Mild", score)
        elif score <= moderate_max:
            return ("Moderate", score)
        elif score <= severe_max:
            return ("Severe", score)
        else:
            return ("Very Severe", score)

    return ("__fallback__", None)

def severity_fallback(patient_vars: Dict) -> str:
    plate = patient_vars["platelet"]
    ns1 = bool(patient_vars["ns1"])
    igg = bool(patient_vars["igg"])
    igm = bool(patient_vars["igm"])
    # conservative placeholder rules
    if plate is not None and plate < 20000:
        return "Very Severe"
    if (plate is not None and plate < 50000) or (ns1 and plate is not None and plate < 80000):
        return "Severe"
    if ns1 or igg or igm:
        return "Moderate"
    return "Mild"

def required_bed_type(severity: str) -> str:
    return "ICU" if severity in ("Severe", "Very Severe") else "Normal"

def nearest_with_vacancy(dist_mat: pd.DataFrame, start_hospital: str, candidates: List[str]) -> Optional[str]:
    if start_hospital not in dist_mat.index:
        return None
    row = dist_mat.loc[start_hospital, candidates]
    row = row.dropna()
    if row.empty:
        return None
    return row.idxmin()

def normalize_hospital(x):
    return str(x).strip()

# ----------------------
# Sidebar: Upload & mapping
# ----------------------

st.sidebar.header("📁 Data files")
predicted_file = st.sidebar.file_uploader("Predicted dataset AIO (.xlsx)", type=["xlsx"], key="pred_file")
alloc_file = st.sidebar.file_uploader("Allocation dataset (.xlsx)", type=["xlsx"], key="alloc_file")
loc_file = st.sidebar.file_uploader("Location matrix (.xlsx)", type=["xlsx"], key="loc_file")

# Try defaults if present in working directory
if predicted_file is None:
    default_pred = "Predicted dataset AIO.xlsx"
    if Path(default_pred).exists():
        predicted_file = default_pred
if alloc_file is None:
    default_alloc = "Allocation dataset.xlsx"
    if Path(default_alloc).exists():
        alloc_file = default_alloc
if loc_file is None:
    default_loc = "Location matrix.xlsx"
    if Path(default_loc).exists():
        loc_file = default_loc

if not all([predicted_file, alloc_file, loc_file]):
    st.info("Please upload all three files (or place them alongside this app with the exact names).")
    st.stop()

df_pred = read_excel_any(predicted_file)
df_alloc = read_excel_any(alloc_file)
df_loc = read_excel_any(loc_file)

if df_pred is None or df_alloc is None or df_loc is None:
    st.stop()

# Clean / standardize a bit
for d in [df_pred, df_alloc, df_loc]:
    d.columns = [str(c).strip() for c in d.columns]

# Build distance matrix
try:
    dist_mat = build_distance_matrix(df_loc)
    dist_mat.index = dist_mat.index.map(normalize_hospital)
    dist_mat.columns = dist_mat.columns.map(normalize_hospital)
except Exception as e:
    st.error(f"Location matrix error: {e}")
    st.stop()

# Column mapping for predicted dataset
st.sidebar.header("🧭 Column mapping (Predicted)")
# Attempt auto-detects
pred_hosp_col = autodetect_columns(df_pred, ["hospital", "facility", "center", "centre"])
pred_date_col = autodetect_columns(df_pred, ["date", "day"])
pred_bed_col = autodetect_columns(df_pred, ["available bed", "bed available", "normal bed", "free bed", "beds free", "beds available"])
pred_icu_col = autodetect_columns(df_pred, ["icu available", "icu bed", "icu_free", "free icu", "icu"])

pred_hosp_col = st.sidebar.selectbox("Hospital column", options=df_pred.columns.tolist(), index=(df_pred.columns.tolist().index(pred_hosp_col) if pred_hosp_col in df_pred.columns else 0))
pred_date_col = st.sidebar.selectbox("Date column", options=df_pred.columns.tolist(), index=(df_pred.columns.tolist().index(pred_date_col) if pred_date_col in df_pred.columns else 0))
pred_bed_col = st.sidebar.selectbox("**Predicted** Normal beds available", options=df_pred.columns.tolist(), index=(df_pred.columns.tolist().index(pred_bed_col) if pred_bed_col in df_pred.columns else 0))
pred_icu_col = st.sidebar.selectbox("**Predicted** ICU beds available", options=df_pred.columns.tolist(), index=(df_pred.columns.tolist().index(pred_icu_col) if pred_icu_col in df_pred.columns else 0))

# Normalize core fields
df_pred["_Hospital"] = df_pred[pred_hosp_col].apply(normalize_hospital)
# Parse date
def _parse_dt(x):
    if isinstance(x, (pd.Timestamp, datetime)):
        return pd.to_datetime(x).date()
    try:
        return pd.to_datetime(x, dayfirst=True, errors="coerce").date()
    except Exception:
        return pd.NaT
df_pred["_Date"] = df_pred[pred_date_col].apply(_parse_dt)
df_pred["_BedsAvail"] = pd.to_numeric(df_pred[pred_bed_col], errors="coerce").fillna(0).astype(int)
df_pred["_ICUAvail"] = pd.to_numeric(df_pred[pred_icu_col], errors="coerce").fillna(0).astype(int)

# Filter to Sep–Dec 2024 (as per request)
mask_range = (pd.to_datetime(df_pred["_Date"]) >= pd.to_datetime("2024-09-01")) & (pd.to_datetime(df_pred["_Date"]) <= pd.to_datetime("2024-12-31"))
df_pred = df_pred.loc[mask_range].copy()

if df_pred.empty:
    st.warning("Predicted dataset has no rows in Sep–Dec 2024 after parsing. Check date column mapping.")
    st.stop()

# Build an availability index
availability = (
    df_pred.groupby(["_Hospital", "_Date"], as_index=False)[["_BedsAvail", "_ICUAvail"]].sum()
    .set_index(["_Hospital", "_Date"])
    .sort_index()
)

# Session state to hold reservations
if "reservations" not in st.session_state:
    st.session_state["reservations"] = {}  # key: (hospital, date, type) -> count reserved

def get_remaining(hospital: str, date, bed_type: str) -> int:
    key = (hospital, date)
    base = availability.loc[key, "_ICUAvail" if bed_type == "ICU" else "_BedsAvail"] if key in availability.index else 0
    # subtract reservations of that type
    rkey = (hospital, date, bed_type)
    reserved = st.session_state["reservations"].get(rkey, 0)
    remain = int(max(0, base - reserved))
    return remain

def reserve_bed(hospital: str, date, bed_type: str, n: int = 1):
    rkey = (hospital, date, bed_type)
    st.session_state["reservations"][rkey] = st.session_state["reservations"].get(rkey, 0) + n

# Severity thresholds (for numeric equation outputs)
with st.sidebar.expander("⚙️ Thresholds (only if equation returns a numeric score)"):
    mild_max = st.number_input("mild_max", value=0.25, min_value=0.0, max_value=1.0, step=0.01)
    moderate_max = st.number_input("moderate_max", value=0.5, min_value=0.0, max_value=1.0, step=0.01)
    severe_max = st.number_input("severe_max", value=0.8, min_value=0.0, max_value=1.0, step=0.01)

# ----------------------
# Patient intake
# ----------------------

st.subheader("🧑‍⚕️ Patient Intake")

colA, colB, colC = st.columns([1.2, 1, 1.2])

with colA:
    hospital_input = st.selectbox("Hospital visited", sorted(availability.index.get_level_values(0).unique().tolist()))
    visit_date = st.date_input("Visit date", value=max(availability.index.get_level_values(1)), min_value=min(availability.index.get_level_values(1)), max_value=max(availability.index.get_level_values(1)))
with colB:
    platelet = st.number_input("Platelet count (/µL)", min_value=0, step=1000, value=80000)
    ns1 = st.selectbox("NS1", options=["Negative", "Positive"])
with colC:
    igg = st.selectbox("IgG", options=["Negative", "Positive"])
    igm_label = "IgM (a.k.a. 'ign')"
    igm = st.selectbox(igm_label, options=["Negative", "Positive"])

colD, colE = st.columns([1, 1])
with colD:
    age = st.number_input("Age (years)", min_value=0, max_value=120, value=25)
with colE:
    weight = st.number_input("Weight (kg)", min_value=1.0, max_value=250.0, value=60.0, step=0.5)

patient_vars = {
    "platelet": float(platelet),
    "ns1": True if ns1 == "Positive" else False,
    "igg": True if igg == "Positive" else False,
    "igm": True if igm == "Positive" else False,
    "age": float(age),
    "weight": float(weight),
}

# Compute severity
severity, score = severity_from_equation(df_alloc, patient_vars, {"mild_max": mild_max, "moderate_max": moderate_max, "severe_max": severe_max})
if severity == "__fallback__":
    severity = severity_fallback(patient_vars)
bed_type = required_bed_type(severity)

st.markdown(f"### 🩺 Severity: **{severity}** {'(score: {:.3f})'.format(score) if score is not None else ''}")
st.markdown(f"### 🛏️ Required bed: **{bed_type}**")

# Availability at visited hospital
rem_here = get_remaining(hospital_input, visit_date, bed_type)
st.metric(label=f"Remaining {bed_type} beds at {hospital_input} on {visit_date}", value=rem_here)

# Candidate hospitals with vacancy
hospitals = sorted(availability.index.get_level_values(0).unique().tolist())
candidates = [h for h in hospitals if get_remaining(h, visit_date, bed_type) > 0]

if rem_here > 0:
    chosen_hospital = hospital_input
    reason = "Requested hospital has vacancy."
else:
    # route to nearest with vacancy using distance matrix
    reroute = nearest_with_vacancy(dist_mat, hospital_input, candidates)
    chosen_hospital = reroute
    reason = "No vacancy at requested hospital; routed to nearest with vacancy."

if chosen_hospital is None:
    st.error("❌ No hospitals with vacancy found for the selected date and bed type.")
else:
    with st.container(border=True):
        st.markdown("#### 🧭 Allocation Decision")
        st.write(f"- **Assigned Hospital**: **{chosen_hospital}**  ")
        st.write(f"- **Date**: **{visit_date}**  ")
        st.write(f"- **Bed Type**: **{bed_type}**  ")
        st.write(f"- **Reason**: {reason}")
        # Distance info if rerouted
        if chosen_hospital != hospital_input:
            try:
                dist_val = dist_mat.loc[normalize_hospital(hospital_input), normalize_hospital(chosen_hospital)]
            except Exception:
                dist_val = np.nan
            if pd.notna(dist_val):
                st.write(f"- **Distance from visited hospital**: {dist_val:.2f} (units as per file)")
        # Confirm allocation
        if st.button("✅ Confirm & Reserve Bed", type="primary"):
            reserve_bed(chosen_hospital, visit_date, bed_type, n=1)
            st.success(f"Reserved 1 {bed_type} bed at {chosen_hospital} for {visit_date}.")

# ----------------------
# Explorer
# ----------------------
st.divider()
st.subheader("📊 Availability Explorer")

sel_bed_type = st.selectbox("Bed type", ["Normal", "ICU"], index=0)
sel_hospital = st.selectbox("Hospital", hospitals, index=(hospitals.index(hospital_input) if hospital_input in hospitals else 0))

dates = sorted(list(set([d for h, d in availability.index])))
records = []
for d in dates:
    key = (sel_hospital, d)
    base = availability.loc[key, "_ICUAvail" if sel_bed_type == "ICU" else "_BedsAvail"] if key in availability.index else 0
    reserved = st.session_state["reservations"].get((sel_hospital, d, "ICU" if sel_bed_type=="ICU" else "Normal"), 0)
    remain = max(0, int(base - reserved))
    records.append({"Date": d, "Base Available": int(base), "Reserved": int(reserved), "Remaining": remain})
df_view = pd.DataFrame(records)
st.dataframe(df_view, use_container_width=True, hide_index=True)

# ----------------------
# Notes for operators
# ----------------------
with st.expander("📝 Notes & Assumptions", expanded=False):
    st.markdown("""
    - The app **does not** modify your source Excel files; reservations are tracked only **in session**.
    - To make severity rules *data-driven*, add a column named **`equation`** in the **Allocation dataset** (any sheet).
      Example equations (Python-like):
      - `"0.9 if platelet < 20000 else 0.7 if platelet < 50000 else 0.5 if ns1 else 0.2"`
      - `" 'Very Severe' if platelet < 20000 else ('Severe' if platelet < 50000 else 'Moderate') "`
      Variables available: `platelet`, `igg`, `igm`, `ns1`, `age`, `weight`.
    - If your equation returns a **numeric score** in [0,1], we map it to categories using the thresholds in the sidebar.
    - The **Location matrix** can be:
      - **Long form**: columns like `from`, `to`, `distance` (case-insensitive, partial names OK)
      - **Wide form**: first column hospital names, remaining columns the distance to each hospital.
    - Availability columns from the **Predicted dataset** are configurable in the sidebar mapping.
    """)
