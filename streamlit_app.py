import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Union

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Dengue Patient Allocation System", page_icon="🏥", layout="wide")
st.title("🏥 Dengue Patient Allocation System")

with st.expander("ℹ️ How this app works", expanded=False):
    st.markdown("""
- **Predictions (required):** Provide a prediction file (XLSX/CSV). The app prefers **predicted availability columns** (e.g., "Predicted Normal Beds Available", "Predicted ICU Beds Available").
  - If those don’t exist, it falls back to `Beds Total - Beds Occupied` and `ICU Beds Total - ICU Beds Occupied`.
  - Supports either a **Date** column or **Year/Month** columns (monthly).
- **Allocation dataset (optional but recommended):** 
  - If it has an `equation`/`severity_equation` column, it will be safely evaluated with variables: `platelet, igg, igm, ns1, age, weight`.
  - If it has a **rules table** (e.g., `platelet_min/max, age_min/max, weight_min/max, ns1_required, igg_required, igm_required, severity`), it applies the first matching rule.
  - Otherwise, a conservative fallback is used.
- **Location matrix (required for rerouting):** CSV or XLSX; either **wide** (matrix) or **long** (`from, to, distance`). Used to reroute to nearest hospital with vacancy.
- **Rerouting & Reservations:** If no vacancy at selected hospital, the app reroutes to nearest hospital with capacity. When you confirm, the bed is **reserved in-session** (does not modify source files).
""")

# -----------------------------
# Helpers
# -----------------------------
def read_any(path_or_buf: Union[str, Path], sheet: Optional[str] = None):
    """Read CSV or Excel (single sheet)."""
    try:
        p = str(path_or_buf)
        if p.lower().endswith(".csv"):
            return pd.read_csv(path_or_buf)
        # Excel
        return pd.read_excel(path_or_buf, sheet_name=sheet)
    except Exception as e:
        st.error(f"Failed to read file: {path_or_buf}\n{e}")
        return None

def read_all_sheets_excel(path_or_buf: Union[str, Path]):
    """Read all sheets from an Excel file. Returns dict or None."""
    try:
        return pd.read_excel(path_or_buf, sheet_name=None)
    except Exception as e:
        st.warning(f"Could not read all sheets from: {path_or_buf}\n{e}")
        return None

def autodetect(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Return first column that contains any candidate (case-insensitive substring)."""
    if not isinstance(df, pd.DataFrame):
        return None
    for patt in candidates:
        for c in df.columns:
            if patt.lower() in str(c).lower():
                return c
    return None

def ensure_dataframe_columns_stripped(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if isinstance(df, pd.DataFrame):
        df.columns = [str(c).strip() for c in df.columns]
    return df

def normalize_hospital_name(x) -> str:
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
    Accept:
    - Long form: columns like [from, to, distance]
    - Wide form: first col = hospital, others = hospitals with numeric distances
    Returns symmetric matrix.
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    # Try long form
    long_from = autodetect(df, ["from", "source", "origin", "hospital_from", "start"])
    long_to   = autodetect(df, ["to", "dest", "destination", "hospital_to", "end"])
    long_dist = autodetect(df, ["distance", "km", "dist"])
    if all([long_from, long_to, long_dist]):
        piv = df[[long_from, long_to, long_dist]].copy()
        piv.columns = ["from", "to", "distance"]
        mat = piv.pivot_table(index="from", columns="to", values="distance", aggfunc="min")
        return mat.combine_first(mat.T)

    # Try wide form
    if df.shape[1] > 2:
        df = df.set_index(df.columns[0])
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df.combine_first(df.T)

    raise ValueError("Could not interpret Location matrix format. Provide long or wide form.")

def to_bool_or_none(x):
    if pd.isna(x):
        return None
    s = str(x).strip().lower()
    if s in ["1", "true", "yes", "y", "t", "required", "need", "needed", "pos", "positive"]:
        return True
    if s in ["0", "false", "no", "n", "f", "not required", "neg", "negative"]:
        return False
    return None

def get_rules_from_df(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    # Need a severity column
    sev_col = None
    for c in df.columns:
        if "severity" in str(c).lower():
            sev_col = c
            break
    if not sev_col:
        return None

    def pick(alts):
        for patt in alts:
            hit = autodetect(df, [patt])
            if hit:
                return hit
        return None

    col_map = {
        "platelet_min": pick(["platelet_min", "plt_min", "platelet lower", "min_platelet"]),
        "platelet_max": pick(["platelet_max", "plt_max", "platelet upper", "max_platelet"]),
        "age_min":      pick(["age_min", "min_age"]),
        "age_max":      pick(["age_max", "max_age"]),
        "weight_min":   pick(["weight_min", "min_weight"]),
        "weight_max":   pick(["weight_max", "max_weight"]),
        "ns1_required": pick(["ns1_required", "ns1 req", "require ns1", "ns1_flag"]),
        "igg_required": pick(["igg_required", "igg req", "require igg"]),
        "igm_required": pick(["igm_required", "igm req", "require igm", "ign_required"]),
    }

    rules = df[df[sev_col].notna()].copy()
    if rules.empty:
        return None

    # Normalize numeric bounds
    for k in ["platelet_min","platelet_max","age_min","age_max","weight_min","weight_max"]:
        src = col_map.get(k)
        rules[k] = pd.to_numeric(rules[src], errors="coerce") if src else np.nan

    # Normalize boolean requirements
    for k in ["ns1_required","igg_required","igm_required"]:
        src = col_map.get(k)
        rules[k] = rules[src].apply(to_bool_or_none) if src else None

    rules["severity"] = rules[sev_col].astype(str).str.strip().str.title()
    return rules[["platelet_min","platelet_max","age_min","age_max","weight_min","weight_max",
                  "ns1_required","igg_required","igm_required","severity"]]

def match_rules(rules: pd.DataFrame, vars: Dict) -> Optional[str]:
    for _, r in rules.iterrows():
        ok = True
        if pd.notna(r["platelet_min"]) and not (vars["platelet"] >= r["platelet_min"]): ok = False
        if pd.notna(r["platelet_max"]) and not (vars["platelet"] <= r["platelet_max"]): ok = False
        if pd.notna(r["age_min"])      and not (vars["age"]      >= r["age_min"]): ok = False
        if pd.notna(r["age_max"])      and not (vars["age"]      <= r["age_max"]): ok = False
        if pd.notna(r["weight_min"])   and not (vars["weight"]   >= r["weight_min"]): ok = False
        if pd.notna(r["weight_max"])   and not (vars["weight"]   <= r["weight_max"]): ok = False
        for flag in ["ns1_required","igg_required","igm_required"]:
            req = r[flag]
            if req is not None:
                key = flag.split("_")[0]  # ns1/igg/igm
                if req is True and not vars[key]: ok = False
                if req is False and vars[key]: ok = False
        if ok:
            return r["severity"]
    return None

def safe_eval_equation(eq: str, vars_dict: Dict) -> Optional[Union[float, str]]:
    safe_globals = {"__builtins__": {}, "np": np, "min": min, "max": max, "abs": abs, "round": round}
    try:
        return eval(eq, safe_globals, vars_dict)
    except Exception:
        return None

def compute_severity_from_allocation(alloc: Optional[Dict[str, pd.DataFrame]], patient_vars: Dict) -> Tuple[str, Optional[float]]:
    """
    1) If any sheet has an equation column, evaluate it (string category or numeric score).
    2) Else if any sheet looks like a rules table, apply first matching rule.
    3) Else fallback.
    """
    # 1) Equation
    if isinstance(alloc, dict):
        for _, df in alloc.items():
            df = ensure_dataframe_columns_stripped(df)
            if not isinstance(df, pd.DataFrame):
                continue
            eq_col = None
            for c in df.columns:
                if "equation" in c.lower():
                    eq_col = c
                    break
            if eq_col:
                series = df[eq_col].dropna().astype(str)
                if not series.empty:
                    eq = series.iloc[0]
                    res = safe_eval_equation(eq, patient_vars)
                    if isinstance(res, str):
                        s = res.strip().lower()
                        mapping = {"mild": "Mild", "moderate": "Moderate", "severe": "Severe",
                                   "very severe": "Very Severe", "very_severe": "Very Severe", "normal": "Normal"}
                        return (mapping.get(s, "Moderate"), None)
                    if isinstance(res, (int, float, np.integer, np.floating)):
                        score = float(res)
                        # try thresholds if present
                        th = {"mild_max": 0.25, "moderate_max": 0.5, "severe_max": 0.8}
                        for tc in df.columns:
                            lc = tc.lower()
                            if "mild" in lc and "max" in lc:
                                val = pd.to_numeric(df[tc], errors="coerce").dropna()
                                if not val.empty: th["mild_max"] = float(val.iloc[0])
                            if "moderate" in lc and "max" in lc:
                                val = pd.to_numeric(df[tc], errors="coerce").dropna()
                                if not val.empty: th["moderate_max"] = float(val.iloc[0])
                            if "severe" in lc and "max" in lc:
                                val = pd.to_numeric(df[tc], errors="coerce").dropna()
                                if not val.empty: th["severe_max"] = float(val.iloc[0])
                        if score <= th["mild_max"]: return ("Mild", score)
                        if score <= th["moderate_max"]: return ("Moderate", score)
                        if score <= th["severe_max"]: return ("Severe", score)
                        return ("Very Severe", score)

    # 2) Rules table
    if isinstance(alloc, dict):
        for _, df in alloc.items():
            df = ensure_dataframe_columns_stripped(df)
            if not isinstance(df, pd.DataFrame):
                continue
            rules = get_rules_from_df(df)
            if rules is not None and not rules.empty:
                sev = match_rules(rules, patient_vars)
                if sev:
                    return (sev, None)

    # 3) Fallback (conservative)
    plate, ns1, igg, igm = patient_vars["platelet"], patient_vars["ns1"], patient_vars["igg"], patient_vars["igm"]
    if plate is not None and plate < 20000: return ("Very Severe", None)
    if (plate is not None and plate < 50000) or (ns1 and plate is not None and plate < 80000): return ("Severe", None)
    if ns1 or igg or igm: return ("Moderate", None)
    return ("Normal", None)

def required_resource(severity: str) -> str:
    return "ICU" if severity in ("Severe", "Very Severe") else "General Bed"

def nearest_with_vacancy(dist_mat: pd.DataFrame, start_hospital: str, candidates: List[str]) -> Optional[str]:
    if start_hospital not in dist_mat.index:
        return None
    try:
        row = dist_mat.loc[start_hospital, candidates].dropna()
    except Exception:
        # Some labels might not match exactly; try normalized labels
        return None
    if row.empty: return None
    return row.idxmin()

# -----------------------------
# Sidebar: File inputs
# -----------------------------
st.sidebar.header("📁 Data")
pred_file = st.sidebar.file_uploader("Predictions (CSV/XLSX)", type=["csv","xlsx"])
alloc_file = st.sidebar.file_uploader("Allocation dataset (XLSX) — optional", type=["xlsx"])
loc_file = st.sidebar.file_uploader("Location matrix (CSV/XLSX)", type=["csv","xlsx"])

# Also allow common repo names for zero-config
def default_if_exists(name, current):
    return current or (Path(name) if Path(name).exists() else None)

pred_file = default_if_exists("Predicted dataset AIO.xlsx", pred_file)
pred_file = default_if_exists("ensemble_predictions_2026_2027_dynamic.xlsx", pred_file)

alloc_file = default_if_exists("Allocation dataset.xlsx", alloc_file)

loc_file = default_if_exists("Location matrix.xlsx", loc_file)
loc_file = default_if_exists("distance matrix.csv", loc_file)

if not pred_file or not loc_file:
    st.error("Please provide at least a **Predictions** file and a **Location matrix** (CSV/XLSX).")
    st.stop()

# -----------------------------
# Load data
# -----------------------------
df_pred = read_any(pred_file)
df_loc = read_any(loc_file)
alloc_sheets = read_all_sheets_excel(alloc_file) if alloc_file else None

df_pred = ensure_dataframe_columns_stripped(df_pred)
df_loc = ensure_dataframe_columns_stripped(df_loc)

if df_pred is None or df_loc is None:
    st.stop()

# -----------------------------
# Prediction mapping & normalization
# -----------------------------
# Try to find hospital, date OR year/month
hospital_col = autodetect(df_pred, [
    "hospital", "Hospital name", "hosp", "facility", "center", "centre", "clinic"
])
date_col = autodetect(df_pred, ["date"])
year_col = autodetect(df_pred, ["year"])
month_col = autodetect(df_pred, ["month"])

if hospital_col is None:
    st.error("Couldn't detect a hospital column in predictions. Try naming it like 'Hospital'.")
    st.stop()

df_pred["_Hospital"] = df_pred[hospital_col].astype(str).map(normalize_hospital_name)

if date_col:
    df_pred["_Date"] = parse_date_series(df_pred[date_col])
elif year_col and month_col:
    try:
        df_pred["_Date"] = pd.to_datetime(df_pred[year_col].astype(int).astype(str) + "-" +
                                          df_pred[month_col].astype(int).astype(str) + "-01",
                                          errors="coerce").dt.date
    except Exception as e:
        st.error(f"Failed constructing Date from Year/Month: {e}")
        st.stop()
else:
    st.error("Provide either a Date column or both Year and Month in predictions.")
    st.stop()

# Prefer predicted availability columns; otherwise compute availability from totals/occupied
pred_normal_avail_col = autodetect(df_pred, ["predicted normal beds available", "normal beds available (pred)", "beds available predicted", "pred beds"])
pred_icu_avail_col    = autodetect(df_pred, ["predicted icu beds available", "icu beds available (pred)", "icu available predicted", "pred icu"])

beds_total_col     = autodetect(df_pred, ["beds total","total beds"])
icu_total_col      = autodetect(df_pred, ["icu beds total","total icu"])
beds_occ_col       = autodetect(df_pred, ["beds occupied","occupied beds"])
icu_occ_col        = autodetect(df_pred, ["icu beds occupied","occupied icu"])

if pred_normal_avail_col and pred_icu_avail_col:
    df_pred["_BedsAvail"] = pd.to_numeric(df_pred[pred_normal_avail_col], errors="coerce").fillna(0).astype(int)
    df_pred["_ICUAvail"]  = pd.to_numeric(df_pred[pred_icu_avail_col], errors="coerce").fillna(0).astype(int)
elif all([beds_total_col, beds_occ_col, icu_total_col, icu_occ_col]):
    df_pred["_BedsAvail"] = (pd.to_numeric(df_pred[beds_total_col], errors="coerce") -
                             pd.to_numeric(df_pred[beds_occ_col], errors="coerce")).fillna(0).astype(int)
    df_pred["_ICUAvail"]  = (pd.to_numeric(df_pred[icu_total_col], errors="coerce") -
                             pd.to_numeric(df_pred[icu_occ_col], errors="coerce")).fillna(0).astype(int)
else:
    st.error("Could not find predicted availability columns or fallback (Totals & Occupied) columns.")
    st.stop()

# Clean by date and hospital
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
    dist_mat.index = dist_mat.index.map(normalize_hospital_name)
    dist_mat.columns = dist_mat.columns.map(normalize_hospital_name)
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
# UI – Patient intake
# -----------------------------
hospitals = sorted(availability.index.get_level_values(0).unique().tolist())
if not hospitals:
    st.error("No hospitals found in predictions after processing.")
    st.stop()

dates = sorted(list(set([d for _, d in availability.index])))
if not dates:
    st.error("No dates found in predictions after processing.")
    st.stop()

with st.form("allocation_form"):
    st.subheader("🔍 Patient Information")
    hospital = st.selectbox("Hospital Name", hospitals)
    date_input = st.date_input("Admission/Test Date", value=max(dates), min_value=min(dates), max_value=max(dates))
    colA, colB = st.columns(2)
    with colA:
        age = st.number_input("Age", min_value=0, max_value=120, value=25)
        weight = st.number_input("Weight (kg)", min_value=1, max_value=250, value=60)
        platelet = st.number_input("Platelet Count (/µL)", min_value=0, value=120000, step=1000)
    with colB:
        igg = st.selectbox("IgG", ["Positive", "Negative"])
        igm = st.selectbox("IgM (a.k.a. 'ign')", ["Positive", "Negative"])
        ns1 = st.selectbox("NS1", ["Positive", "Negative"])
    submit = st.form_submit_button("🚑 Allocate Patient")

if submit:
    # severity
    patient_vars = {
        "platelet": float(platelet),
        "igg": igg == "Positive",
        "igm": igm == "Positive",
        "ns1": ns1 == "Positive",
        "age": float(age),
        "weight": float(weight),
    }
    severity, score = compute_severity_from_allocation(alloc_sheets, patient_vars)
    resource_needed = required_resource(severity)

    # availability check
    rem_here = get_remaining(hospital, date_input, "ICU" if resource_needed == "ICU" else "General Bed")
    # We store only "ICU" or "Normal" internally
    bed_key = "ICU" if resource_needed == "ICU" else "Normal"

    assigned_hospital = hospital
    rerouted_distance = None
    note = ""

    if rem_here > 0:
        note = "Assigned at selected hospital"
    else:
        cands = [h for h in hospitals if get_remaining(h, date_input, bed_key) > 0]
        reroute = nearest_with_vacancy(dist_mat, hospital, cands)
        if reroute:
            assigned_hospital = reroute
            try:
                rerouted_distance = dist_mat.loc[hospital, reroute]
            except Exception:
                rerouted_distance = None
            note = f"Rerouted to nearest hospital with {resource_needed}"
        else:
            note = "No hospitals with vacancy found for selected date and resource"

    # Reserve if we succeeded
    if "No hospitals" not in note:
        reserve_bed(assigned_hospital, date_input, bed_key, n=1)

    st.subheader("📋 Allocation Result")
    result = {
        "Date": pd.to_datetime(date_input).strftime("%Y-%m-%d"),
        "Severity": severity if score is None else f"{severity} (score={score:.3f})",
        "Resource Needed": resource_needed,
        "Hospital Tried": hospital,
        "Available at Current Hospital": "Yes" if assigned_hospital == hospital and note.startswith("Assigned") else "No",
        "Assigned Hospital": assigned_hospital,
    }
    if rerouted_distance is not None:
        result["Distance (matrix units)"] = float(rerouted_distance)
    result["Note"] = note
    st.json(result)

# -----------------------------
# Explorer
# -----------------------------
st.divider()
st.subheader("📊 Availability Explorer")

sel_type = st.selectbox("Bed type", ["General Bed","ICU"], index=0)
sel_hosp = st.selectbox("Hospital", hospitals, index=0)

rows = []
for d in dates:
    base = 0
    key = (sel_hosp, d)
    if key in availability.index:
        base = int(availability.loc[key, "_ICUAvail" if sel_type == "ICU" else "_BedsAvail"])
    reserved = st.session_state["reservations"].get((sel_hosp, d, "ICU" if sel_type == "ICU" else "Normal"), 0)
    rows.append({"Date": d, "Base Available": base, "Reserved (this session)": reserved, "Remaining": max(0, base - reserved)})

st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
