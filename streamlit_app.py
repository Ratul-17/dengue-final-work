# streamlit_app.py
import traceback
import streamlit as st
import pandas as pd
import numpy as np
import re
import math
import requests
import smtplib, ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from functools import lru_cache
from difflib import get_close_matches
from datetime import datetime
from pathlib import Path
from typing import Optional, List

# try optional imports for mapping
try:
    import folium
    from streamlit_folium import st_folium
    FOLIUM_AVAILABLE = True
except Exception:
    import pydeck as pdk
    FOLIUM_AVAILABLE = False

# -------------------------
# App config + header
# -------------------------
st.set_page_config(page_title="Dengue Allocation", page_icon="🏥", layout="wide")
st.title("🏥 Integrated Hospital Dengue Patient Allocation System (DSCC Region)")

# -------------------------
# Static lists
# -------------------------
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

DHAKA_AREAS = [
    "Dhanmondi","Mohammadpur","Gulshan","Banani","Baridhara","Uttara","Mirpur","Kafrul","Pallabi",
    "Tejgaon","Farmgate","Kawran Bazar","Panthapath","Kalabagan","New Market","Science Lab",
    "Elephant Road","Lalmatia","Shyamoli","Agargaon","Sher-e-Bangla Nagar","Kallyanpur","Gabtoli",
    "Hazaribagh","Rayer Bazar","Jhigatola","Azimpur","Lalbagh","Chankharpul","Shahbagh","Paltan",
    "Motijheel","Dilkusha","Wari","Sutrapur","Kotwali","Bangshal","Chawkbazar","Sadarghat","Narinda",
    "Ramna","Eskaton","Moghbazar","Mouchak","Malibagh","Rampura","Banasree","Aftabnagar","Badda",
    "Khilgaon","Basabo","Shantinagar","Kakrail","Khilkhet","Nikunja","Airport","Cantonment","Mohakhali",
    "Jatrabari","Demra","Keraniganj","Kamalapur","Sayedabad","Tikatuli","Arambagh","Paribagh"
]

# -------------------------
# Theme-aware CSS (keeps UI readable in light/dark)
# -------------------------
st.markdown("""
<style>
:root{
  --bg:#f5f7fb; --bg2:#ffffff; --text:#0f172a; --muted:#475569; --card:rgba(255,255,255,.85);
  --border:rgba(2,6,23,.08); --shadow:0 8px 22px rgba(2,6,23,.08);
}
@media (prefers-color-scheme: dark){
  :root{
    --bg:#0b1220; --bg2:#111827; --text:#e5e7eb; --muted:#94a3b8; --card:rgba(255,255,255,.06);
    --border:rgba(255,255,255,.10); --shadow:0 10px 30px rgba(0,0,0,.25);
  }
}
html, body, [data-testid="stAppViewContainer"]{background:linear-gradient(135deg,var(--bg) 0%,var(--bg) 40%,var(--bg2) 100%) !important;color:var(--text)}
.card{border-radius:12px;padding:14px;background:var(--card);border:1px solid var(--border);box-shadow:var(--shadow)}
.kpi{font-weight:800;font-size:1.6rem}
.kpi-label{color:var(--muted);font-size:.9rem}
.pill{padding:8px 12px;border-radius:999px;background:rgba(0,0,0,0.06);border:1px solid var(--border)}
.codebox{background:transparent;border:1px dashed var(--border);padding:10px;border-radius:8px}
</style>
""", unsafe_allow_html=True)

# -------------------------
# Utilities
# -------------------------
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

def norm_key(s: str) -> str:
    STOPWORDS = {"hospital","medical","college","institute","university","center","centre","clinic","and"}
    s = str(s).lower().strip().replace("&"," and ")
    s = re.sub(r"[^a-z0-9\s]"," ", s)
    tokens = [t for t in s.split() if t and t not in STOPWORDS]
    return "".join(tokens)

# -------------------------
# Severity logic
# -------------------------
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

# -------------------------
# Mapping / Geocoding / Distances
# -------------------------
DHAKA_VIEWBOX = (90.30, 23.69, 90.50, 23.90)   # lon_min, lat_min, lon_max, lat_max

def _contact_email_for_user_agent() -> str:
    try:
        return st.secrets.get("contact", {}).get("email", "dengue-allocator@example.com")
    except Exception:
        return "dengue-allocator@example.com"

def _with_bd_context(q: str) -> str:
    s = (q or "").strip()
    s_l = s.lower()
    if ("bangladesh" not in s_l) and ("dhaka" not in s_l):
        s = f"{s}, Dhaka, Bangladesh"
    return s

@lru_cache(maxsize=512)
def geocode_nominatim(query: str):
    """Return (lat, lon) or None using Nominatim (bounded to Dhaka viewbox)."""
    try:
        q = _with_bd_context(query)
        url = "https://nominatim.openstreetmap.org/search"
        headers = {"User-Agent": f"dscc-dengue-allocator/1.0 ({_contact_email_for_user_agent()})"}
        params = {
            "q": q, "format": "json", "limit": 1, "countrycodes": "bd",
            "viewbox": f"{DHAKA_VIEWBOX[0]},{DHAKA_VIEWBOX[1]},{DHAKA_VIEWBOX[2]},{DHAKA_VIEWBOX[3]}",
            "bounded": 1,
        }
        r = requests.get(url, headers=headers, params=params, timeout=12)
        r.raise_for_status()
        js = r.json()
        if not js: return None
        lat = float(js[0]["lat"]); lon = float(js[0]["lon"])
        return (lat, lon)
    except Exception:
        return None

def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0088
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    dφ = math.radians(lat2 - lat1)
    dλ = math.radians(lon2 - lon1)
    a = math.sin(dφ/2)**2 + math.cos(φ1)*math.cos(φ2)*math.sin(dλ/2)**2
    return R * (2 * math.atan2(math.sqrt(a), math.sqrt(1-a)))

def osrm_drive(origin_ll, dest_ll):
    """Optional: use OSRM demo server for driving distance/time (may be rate-limited)."""
    if not origin_ll or not dest_ll: return None
    o_lat, o_lon = origin_ll; d_lat, d_lon = dest_ll
    try:
        url = f"https://router.project-osrm.org/route/v1/driving/{o_lon},{o_lat};{d_lon},{d_lat}"
        params = {"overview": "false", "alternatives": "false", "steps": "false", "annotations": "false"}
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        js = r.json()
        if js.get("code") != "Ok" or not js.get("routes"): return None
        dist_km = js["routes"][0]["distance"] / 1000.0
        dur_min = js["routes"][0]["duration"] / 60.0
        return (dist_km, dur_min)
    except Exception:
        return None

# -------------------------
# Accurate hospital coordinates (lat, lon)
# Source: compiles OpenStreetMap / DGHS / hospital websites — adjust if you have better data
# -------------------------
HOSPITAL_COORDS = {
    "Dhaka Medical College Hospital": (23.72591, 90.39805),
    "SSMC & Mitford Hospital": (23.710255, 90.401435),
    "Bangladesh Shishu Hospital & Institute": (23.77296, 90.36861),
    "Shaheed Suhrawardy Medical College hospital": (23.76918, 90.37103),
    "Bangabandhu Shiekh Mujib Medical University": (23.73890, 90.39480),
    "Police Hospital, Rajarbagh": (23.74063, 90.41737),
    "Mugda Medical College": (23.73248, 90.43003),
    "Bangladesh Medical College Hospital": (23.75014, 90.36973),
    "Holy Family Red Cresent Hospital": (23.73790, 90.39150),
    "BIRDEM Hospital": (23.73896, 90.39644),
    "Ibn Sina Hospital": (23.75153, 90.36898),
    "Square Hospital": (23.75302, 90.38163),
    "Samorita Hospital": (23.75239, 90.38533),
    "Central Hospital Dhanmondi": (23.74306, 90.38389),
    "Lab Aid Hospital": (23.74200, 90.38304),
    "Green Life Medical Hospital": (23.74655, 90.38576),
    "Sirajul Islam Medical College Hospital": (23.74720, 90.41050),
    "Ad-Din Medical College Hospital": (23.74806, 90.40528),
}
HOSPITAL_PLACES = {k: f"{k}, Dhaka, Bangladesh" for k in HOSPITALS_UI}

@lru_cache(maxsize=256)
def geocode_hospital(ui_name: str):
    if ui_name in HOSPITAL_COORDS:
        return HOSPITAL_COORDS[ui_name]
    # fallback to nominatim
    q = HOSPITAL_PLACES.get(ui_name, f"{ui_name}, Dhaka, Bangladesh")
    return geocode_nominatim(q)

# -------------------------
# Availability helpers (we'll build availability from predictions)
# -------------------------
def build_distance_matrix(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy(); df.columns = [str(c).strip() for c in df.columns]
    if df.shape[1] > 2:
        df = df.set_index(df.columns[0])
        for c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
        return df.combine_first(df.T)
    raise ValueError("Could not interpret Location matrix format.")

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

# -------------------------
# Email helpers (multi-recipient)
# -------------------------
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
def is_valid_email(addr: str) -> bool:
    return isinstance(addr, str) and bool(EMAIL_RE.match(addr.strip()))

def send_email_multi(recipients, subject, html_body):
    try:
        if "smtp" not in st.secrets:
            raise RuntimeError("SMTP config missing in Streamlit secrets.")
        cfg = st.secrets["smtp"]
        smtp_host = cfg.get("host", "smtp.gmail.com")
        smtp_port = int(cfg.get("port", 465))
        smtp_user = cfg.get("user")
        smtp_pass = cfg.get("password")
        sender = cfg.get("sender", smtp_user)
        if not all([smtp_host, smtp_port, smtp_user, smtp_pass, sender]):
            raise RuntimeError("Incomplete SMTP config in secrets.")
        # parse recipients string
        if isinstance(recipients, str):
            recipients = [p.strip() for p in re.split(r"[;,]", recipients) if p.strip()]
        recipients = [r for r in recipients if is_valid_email(r)]
        if not recipients: raise ValueError("No valid recipients provided.")
        context = ssl.create_default_context()
        if smtp_port == 465:
            server = smtplib.SMTP_SSL(smtp_host, smtp_port, context=context, timeout=20)
        else:
            server = smtplib.SMTP(smtp_host, smtp_port, timeout=20); server.ehlo(); server.starttls(context=context); server.ehlo()
        server.login(smtp_user, smtp_pass)
        sent_ok, sent_fail = [], []
        for rcp in recipients:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject; msg["From"] = sender; msg["To"] = rcp
            msg.attach(MIMEText(html_body, "html"))
            try:
                server.sendmail(sender, rcp, msg.as_string())
                sent_ok.append(rcp)
            except Exception as e:
                sent_fail.append((rcp, str(e)))
        try: server.quit()
        except: pass
        return {"sent_ok": sent_ok, "sent_fail": sent_fail}
    except Exception as e:
        return {"sent_ok": [], "sent_fail": [("ALL", str(e))]}

def build_allocation_email_html(*, patient_age:int, severity:str, resource:str,
                                tried_hospital_ui:str, assigned_hospital_av:str,
                                date_any, distance_km:Optional[float],
                                beds_avail:int, icu_avail:int,
                                nearest:list[dict] | None = None,
                                user_location:str | None = None) -> str:
    dt_txt = pd.to_datetime(date_any).date().isoformat()
    dist_txt = f"{distance_km:.1f} km" if distance_km is not None else "0.0 km"
    nearest_rows = ""
    if nearest is not None:
        if nearest:
            for i, n in enumerate(nearest, start=1):
                nearest_rows += f"<tr><td>{i}</td><td>{n['ui_name']}</td><td style='text-align:right'>{n['remaining']}</td><td style='text-align:right'>{n['distance_km']:.1f} km</td><td style='text-align:right'>{(int(round(n['duration_min'])) if n.get('duration_min') else '—')}</td></tr>"
        else:
            nearest_rows = "<tr><td colspan='5'>No nearby vacancies right now.</td></tr>"
    nearest_block = f"<h3>Nearest hospitals from: {user_location}</h3><table border='1' cellpadding='6'>{nearest_rows}</table>" if nearest is not None else ""
    html = f"""
    <div>
      <h2>Dengue Patient Allocation Summary</h2>
      <p>Date: <strong>{dt_txt}</strong></p>
      <table border="1" cellpadding="6">
        <tr><td><strong>Patient Age</strong></td><td>{patient_age}</td></tr>
        <tr><td><strong>Severity</strong></td><td>{severity}</td></tr>
        <tr><td><strong>Required Resource</strong></td><td>{resource}</td></tr>
        <tr><td><strong>Hospital Tried</strong></td><td>{tried_hospital_ui}</td></tr>
        <tr><td><strong>Assigned Hospital</strong></td><td>{assigned_hospital_av}</td></tr>
        <tr><td><strong>Distance from Tried Hospital</strong></td><td>{dist_txt}</td></tr>
        <tr><td><strong>Predicted Normal Beds Available</strong></td><td>{beds_avail}</td></tr>
        <tr><td><strong>Predicted ICU Beds Available</strong></td><td>{icu_avail}</td></tr>
      </table>
      {nearest_block}
    </div>
    """
    return html

# -------------------------
# Main app logic wrapped
# -------------------------
def main():
    # -------------------------
    # Robust loader: repo files or uploader fallback
    # -------------------------
    pred_file_path = Path("Predicted dataset AIO.xlsx")
    loc_file_path  = Path("Location matrix.xlsx")

    def read_excel_from_source(src):
        try:
            return ensure_df(pd.read_excel(src))
        except Exception as ex:
            st.error(f"Failed to read Excel: {ex}")
            return None

    df_pred_raw = None
    df_loc = None
    if pred_file_path.exists() and loc_file_path.exists():
        df_pred_raw = read_excel_from_source(pred_file_path)
        df_loc = read_excel_from_source(loc_file_path)
    else:
        st.warning("Required Excel files not found in repository root. You can upload them below.")
        up1 = st.file_uploader("Upload Predicted dataset AIO.xlsx", type=["xls","xlsx"], key="uploader_pred")
        up2 = st.file_uploader("Upload Location matrix.xlsx", type=["xls","xlsx"], key="uploader_loc")
        if up1: df_pred_raw = read_excel_from_source(up1)
        if up2: df_loc = read_excel_from_source(up2)

    # fallback sample (keeps UI alive for testing)
    if df_pred_raw is None or df_loc is None:
        st.info("Using a minimal sample dataset to keep app running (replace with real files for production).")
        sample_pred = pd.DataFrame({
            "Hospital": ["Dhaka Medical College Hospital", "Mugda Medical College"],
            "Date": [pd.Timestamp.today().normalize(), pd.Timestamp.today().normalize()],
            "Predicted Normal Beds Available": [10, 5],
            "Predicted ICU Beds Available": [1, 0]
        })
        sample_loc = pd.DataFrame({
            "Location": ["Dhaka Medical College Hospital", "Mugda Medical College"],
            "Dhaka Medical College Hospital": [0, 6.0],
            "Mugda Medical College": [6.0, 0],
        })
        if df_pred_raw is None:
            df_pred_raw = ensure_df(sample_pred)
        if df_loc is None:
            df_loc = ensure_df(sample_loc)

    if df_pred_raw is None or df_loc is None:
        st.error("Unable to load required data. Please upload the Excel files or add them to the repo root.")
        st.stop()

    # -------------------------
    # Sidebar: time resolution
    # -------------------------
    st.sidebar.header("⏱️ Time Resolution")
    granularity = st.sidebar.selectbox("Time granularity", ["Daily","Weekly","Monthly"], index=0)
    interp_method = st.sidebar.selectbox("Interpolation (when expanding)", ["linear","ffill"], index=0)

    # -------------------------
    # Build availability from predictions
    # -------------------------
    def build_availability_from_predictions(df_pred_raw: pd.DataFrame, granularity: str, interp_method: str) -> pd.DataFrame:
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
            if day_col:
                df["_Date"] = pd.to_datetime(dict(year=df[year_col], month=df[month_col], day=df[day_col]), errors="coerce")
            else:
                df["_Date"] = pd.to_datetime(df[year_col].astype(int).astype(str) + "-" +
                                               df[month_col].astype(int).astype(str) + "-01", errors="coerce")
        else:
            raise ValueError("Provide either a Date column or (Year & Month) in predictions.")
        df = df.dropna(subset=["_Date"]); df["_Date"] = df["_Date"].dt.normalize()

        pred_normal_avail_col = autodetect(df, ["predicted normal beds available","normal beds available (pred)","beds available predicted","pred beds","predicted normal"])
        pred_icu_avail_col    = autodetect(df, ["predicted icu beds available","icu beds available (pred)","icu available predicted","pred icu","predicted icu"])
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
            # try to guess columns if names differ
            # attempt to find any numeric column for beds/icu
            numcols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            if len(numcols) >= 2:
                df["_BedsAvail"] = pd.to_numeric(df[numcols[0]], errors="coerce")
                df["_ICUAvail"] = pd.to_numeric(df[numcols[1]], errors="coerce")
            else:
                raise ValueError("Could not find predicted availability columns or totals/occupied fallback.")
        df["_BedsAvail"] = df["_BedsAvail"].fillna(0); df["_ICUAvail"] = df["_ICUAvail"].fillna(0)

        if granularity == "Monthly":
            df["_Month"] = df["_Date"].dt.to_period("M").dt.to_timestamp()
            grouped = (df.groupby(["_Hospital","_Month"], as_index=False)[["_BedsAvail","_ICUAvail"]].mean())
            availability = (grouped.set_index(["_Hospital","_Month"]).sort_index())
            availability.index = availability.index.set_names(["_Hospital","_Date"])
            return availability

        # Expand to daily
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
        st.text(traceback.format_exc())
        st.stop()

    # distance matrix
    try:
        dist_mat = build_distance_matrix(df_loc)
    except Exception as e:
        st.error(f"Error interpreting location matrix: {e}")
        st.text(traceback.format_exc())
        st.stop()

    # name maps
    DM_TO_AV, UI_TO_DM, UI_TO_AV = build_name_maps(availability, dist_mat, HOSPITALS_UI)

    # -------------------------
    # State (session)
    # -------------------------
    if "reservations" not in st.session_state: st.session_state["reservations"] = {}
    if "served" not in st.session_state: st.session_state["served"] = {}
    if "reroute_log" not in st.session_state: st.session_state["reroute_log"] = []

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

    def month_str_from_date(dt) -> str:
        return pd.to_datetime(dt).strftime("%Y-%m")

    def increment_served(hospital_av_name: str, date) -> None:
        if not hospital_av_name: return
        m = month_str_from_date(date)
        key = (hospital_av_name, m)
        st.session_state["served"][key] = st.session_state["served"].get(key, 0) + 1

    def log_reroute(original_ui: str, assigned_av: str, date) -> None:
        m = month_str_from_date(date)
        st.session_state["reroute_log"].append({
            "date": pd.to_datetime(date).date().isoformat(),
            "original_ui": original_ui,
            "assigned_av": assigned_av,
            "month": m
        })

    def get_month_served(hospital_av_name: str, month_str: str) -> int:
        return st.session_state["served"].get((hospital_av_name, month_str), 0)

    def get_avail_counts(hospital_av_name: str, date) -> dict:
        key = (hospital_av_name, pd.to_datetime(date).normalize())
        out = {"beds_available": None, "icu_available": None}
        if key in availability.index:
            row = availability.loc[key]
            out["beds_available"] = int(np.floor(float(row["_BedsAvail"]))) if not pd.isna(row["_BedsAvail"]) else 0
            out["icu_available"]  = int(np.floor(float(row["_ICUAvail"])))  if not pd.isna(row["_ICUAvail"])  else 0
        return out

    def served_df_for_month(month_str: str) -> pd.DataFrame:
        rows = []
        for (h, m), cnt in st.session_state["served"].items():
            if m == month_str:
                rows.append({"Hospital": h, "Served": cnt})
        if not rows:
            return pd.DataFrame(columns=["Hospital","Served"])
        return pd.DataFrame(rows).sort_values("Served", ascending=False).reset_index(drop=True)

    # -------------------------
    # Patient intake form
    # -------------------------
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

        pick_area = st.selectbox("Pick a Dhaka area (optional)", ["—"] + DHAKA_AREAS, index=0)
        user_location_query = st.text_input("Or type your exact location", placeholder="e.g., House 10, Road 5, Dhanmondi")
        use_driving_eta = st.checkbox("Use driving ETA (beta via OSRM demo)", value=False)

        email_addresses = st.text_input("📧 Recipient Email(s)",
            placeholder="e.g., patient@gmail.com; doctor@hospital.org; admin@health.gov.bd")
        email_opt_in = st.checkbox("Send dengue allocation report via email", value=False)

        submit = st.form_submit_button("🚑 Allocate")

    # -------------------------
    # On submit: compute severity, allocate, optionally reroute and email
    # -------------------------
    assigned_av = None
    rerouted_distance = None
    note = ""
    debug_checks = []

    if submit:
        _, s_score = compute_severity_score(age, ns1_val, igm_val, igg_val, platelet)
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
            increment_served(assigned_av, date_input)
            if assigned_av != (start_av or hospital_ui):
                log_reroute(hospital_ui, assigned_av, date_input)

        # Allocation result UI
        st.subheader("Allocation Result")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f'<div class="kpi">{s_score}</div><div class="kpi-label">Severity Score</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="margin-top:8px">{severity}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card" style="margin-top:12px">', unsafe_allow_html=True)
        st.write(f"**Hospital Tried:** {hospital_ui}")
        st.write(f"**Assigned Hospital:** {assigned_av if assigned_av else '—'}")
        st.write(f"**Note:** {note}")
        st.markdown('</div>', unsafe_allow_html=True)

        # Nearest hospitals by user location
        chosen_loc = user_location_query.strip() if user_location_query.strip() else (pick_area if pick_area != "—" else "")
        nearest_list = []
        user_ll = None
        if chosen_loc:
            try:
                bed_key_needed = "ICU" if resource == "ICU" else "Normal"
                # geocode user then find nearest available hospitals
                user_ll = geocode_nominatim(chosen_loc)
                if user_ll is None:
                    st.error("Could not locate the entered place. Please try a different area name (e.g., 'Dhanmondi') or type a more specific address.")
                else:
                    # collect candidate hospitals with vacancy
                    cand = []
                    for ui_name in HOSPITALS_UI:
                        av_name = UI_TO_AV.get(ui_name) or ui_name
                        rem = get_remaining(av_name, date_input, bed_key_needed)
                        if rem and rem > 0:
                            h_ll = geocode_hospital(ui_name)
                            if not h_ll: continue
                            dist_km = haversine_km(user_ll[0], user_ll[1], h_ll[0], h_ll[1])
                            dur_min = None
                            if use_driving_eta:
                                osrm = osrm_drive(user_ll, h_ll)
                                if osrm:
                                    dist_km, dur_min = osrm
                            if dist_km <= 80:
                                cand.append({
                                    "ui_name": ui_name, "av_name": av_name, "remaining": rem,
                                    "distance_km": float(dist_km), "duration_min": (float(dur_min) if dur_min is not None else None),
                                    "lat": float(h_ll[0]), "lng": float(h_ll[1])
                                })
                    cand.sort(key=lambda x: x["distance_km"])
                    nearest_list = cand[:3]
            except Exception as e:
                st.warning(f"Could not fetch nearest hospitals: {e}")

        st.markdown("### 🗺️ Nearest hospitals with vacancy (by your location)")
        if chosen_loc and nearest_list:
            df_near = pd.DataFrame([{
                "Hospital": n["ui_name"],
                "Vacancy (Beds/ICU)": n["remaining"],
                "Distance (km)": round(n["distance_km"], 1),
                "ETA (min)": (int(round(n["duration_min"])) if n.get("duration_min") else None),
            } for n in nearest_list])
            st.dataframe(df_near, use_container_width=True)

            # MAP: folium preferred (free tiles). White pin for user, red for hospitals.
            if FOLIUM_AVAILABLE:
                # center map at user
                m = folium.Map(location=[user_ll[0], user_ll[1]], zoom_start=13)
                # user marker (white circle)
                folium.CircleMarker(location=[user_ll[0], user_ll[1]],
                                    radius=8, color="#ffffff", fill=True, fill_color="#ffffff", fill_opacity=0.9,
                                    tooltip="You (selected location)").add_to(m)
                # hospital markers
                for n in nearest_list:
                    folium.CircleMarker(location=[n["lat"], n["lng"]],
                                        radius=7, color="#ff3333", fill=True, fill_color="#ff3333",
                                        fill_opacity=0.9, tooltip=f"{n['ui_name']} — {n['remaining']} free").add_to(m)
                # show map
                st_folium(m, width="100%", height=480)
            else:
                # pydeck fallback
                layers = []
                user_df = pd.DataFrame([{"name":"You","lat":user_ll[0],"lon":user_ll[1]}])
                layers.append(pdk.Layer("ScatterplotLayer", user_df, get_position="[lon, lat]", get_radius=80, get_fill_color=[255,255,255,220], pickable=False))
                hosp_df = pd.DataFrame([{"name":n["ui_name"],"lat":n["lat"],"lon":n["lng"]} for n in nearest_list])
                layers.append(pdk.Layer("ScatterplotLayer", hosp_df, get_position="[lon, lat]", get_radius=70, get_fill_color=[255,0,0,220], pickable=True))
                center_lat, center_lon = (user_ll[0], user_ll[1])
                view_state = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=12, pitch=0)
                st.pydeck_chart(pdk.Deck(map_style=None, initial_view_state=view_state, layers=layers), use_container_width=True)
        else:
            st.info("Enter a Dhaka area (pick or type) to see nearest hospitals with vacancy.")

        # Email option
        beds_pred = icu_pred = 0
        if assigned_av:
            assigned_counts = get_avail_counts(assigned_av, date_input)
            beds_pred = assigned_counts["beds_available"] or 0
            icu_pred  = assigned_counts["icu_available"] or 0

        if email_opt_in and email_addresses.strip():
            html = build_allocation_email_html(
                patient_age=age, severity=severity, resource=resource,
                tried_hospital_ui=hospital_ui, assigned_hospital_av=(assigned_av or "—"),
                date_any=date_input, distance_km=(float(rerouted_distance) if rerouted_distance is not None else 0.0),
                beds_avail=beds_pred, icu_avail=icu_pred,
                nearest=(nearest_list if chosen_loc else None),
                user_location=(f"{chosen_loc} — {'driving (OSRM)' if use_driving_eta else 'straight-line'}" if chosen_loc else None),
            )
            subj = f"[Dengue Allocation] {severity} — {resource} · {pd.to_datetime(date_input).date()}"
            res = send_email_multi(email_addresses, subj, html)
            if res["sent_ok"]:
                st.success(f"Email sent to: {', '.join(res['sent_ok'])}")
            if res["sent_fail"]:
                st.warning(f"Some emails failed: {res['sent_fail']}")

        # Debug expandable
        with st.expander("🧪 Debug: Nearest Hospitals Checked"):
            if debug_checks:
                dbg = pd.DataFrame(debug_checks)
                if assigned_av:
                    dbg["Allocated"] = dbg["Neighbor Hospital"].eq(assigned_av)
                    dbg = dbg.sort_values(["Allocated","Remaining Beds/ICU"], ascending=[False,False])
                st.dataframe(dbg, use_container_width=True)
            else:
                st.write("No neighbor checks — assigned at selected hospital or none needed.")

    # -------------------------
    # Management Dashboard (monthly view, leaderboards, logs)
    # -------------------------
    st.markdown("---")
    st.header("📊 Hospital Monthly Dashboard")
    dash_col1, dash_col2 = st.columns([1,1])
    with dash_col1:
        dashboard_ui_hospital = st.selectbox("Choose hospital to view dashboard", HOSPITALS_UI, index=0)
    with dash_col2:
        dashboard_date = st.date_input("View month (pick any date in month)", value=max_d, min_value=min_d, max_value=max_d)
    dashboard_start_av = UI_TO_AV.get(dashboard_ui_hospital) or dashboard_ui_hospital
    dashboard_month = month_str_from_date(dashboard_date)
    st.markdown(f"### Dashboard — {dashboard_ui_hospital}  (month: {dashboard_month})")
    h_avail = get_avail_counts(dashboard_start_av, dashboard_date)
    served_count = get_month_served(dashboard_start_av, dashboard_month)
    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        st.markdown(f'<div class="card"><div class="kpi">{h_avail["beds_available"] if h_avail["beds_available"] is not None else "—"}</div><div class="kpi-label">Normal Beds Available (on selected day)</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="card"><div class="kpi">{h_avail["icu_available"] if h_avail["icu_available"] is not None else "—"}</div><div class="kpi-label">ICU Beds Available (on selected day)</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="card"><div class="kpi">{served_count}</div><div class="kpi-label">Total Patients Served (this month)</div></div>', unsafe_allow_html=True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    reroutes_this_month = [r for r in st.session_state["reroute_log"] if r["month"] == dashboard_month]
    if reroutes_this_month:
        st.markdown("#### Rerouted Assignments (this month)")
        assigned_counts = {}
        for r in reroutes_this_month:
            assigned_counts[r["assigned_av"]] = assigned_counts.get(r["assigned_av"], 0) + 1
        df_rerouted = pd.DataFrame([{"Assigned Hospital":k, "Rerouted Count":v} for k,v in assigned_counts.items()]).sort_values("Rerouted Count", ascending=False).reset_index(drop=True)
        st.dataframe(df_rerouted, use_container_width=True)

        st.markdown("#### Rerouted Hospital Dashboards")
        for assigned_h in assigned_counts.keys():
            st.markdown(f"**{assigned_h}** — total rerouted to here this month: {assigned_counts[assigned_h]}")
            av = get_avail_counts(assigned_h, dashboard_date)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f'<div class="card"><div class="kpi">{av["beds_available"] if av["beds_available"] is not None else "—"}</div><div class="kpi-label">Normal Beds Available (on selected day)</div></div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="card"><div class="kpi">{get_month_served(assigned_h, dashboard_month)}</div><div class="kpi-label">Patients Served (this month)</div></div>', unsafe_allow_html=True)
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    else:
        st.info("No reroutes logged for the selected month.")

    # Leaderboard & time series trend
    st.markdown("### Monthly Leaderboard — Patients Served")
    served_df = served_df_for_month(dashboard_month)
    if not served_df.empty:
        st.bar_chart(data=served_df.set_index("Hospital")["Served"])
        st.dataframe(served_df, use_container_width=True)
    else:
        st.write("No patients served data for this month yet.")

    # full reroute log
    with st.expander("🔁 Full Reroute Log"):
        if st.session_state["reroute_log"]:
            st.dataframe(pd.DataFrame(st.session_state["reroute_log"]), use_container_width=True)
        else:
            st.write("No reroute events logged yet.")

    # raw reservations
    with st.expander("🗂️ Raw Reservations (debug)"):
        if st.session_state["reservations"]:
            rows = []
            for (h, date, bed_type), cnt in st.session_state["reservations"].items():
                rows.append({"Hospital":h, "Date": date, "Bed Type": bed_type, "Reserved": cnt})
            st.dataframe(pd.DataFrame(rows).sort_values(["Date","Hospital"]), use_container_width=True)
        else:
            st.write("No reservations yet.")

# Run main with robust error reporting
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error("App startup error — traceback shown below (copy this for debugging):")
        st.text(traceback.format_exc())
        raise
