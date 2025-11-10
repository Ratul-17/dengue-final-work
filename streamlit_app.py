# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import re
import math
import requests
import smtplib, ssl
import pydeck as pdk
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from functools import lru_cache
from difflib import get_close_matches
from datetime import datetime
from pathlib import Path
from typing import Optional, List

# ===============================
# Page config
# ===============================
st.set_page_config(page_title="Dengue Allocation — Management / Individual", page_icon="🏥", layout="wide")

# ===============================
# Constants: hospitals & Dhaka areas
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

DHAKA_AREAS = [
    "Dhanmondi","Mohammadpur","Gulshan","Banani","Baridhara","Uttara","Mirpur","Kafrul","Pallabi",
    "Tejgaon","Farmgate","Kawran Bazar","Panthapath","Kalabagan","New Market","Science Lab",
    "Elephant Road","Lalmatia","Shyamoli","Agargaon","Sher-e-Bangla Nagar","Kallyanpur","Gabtoli",
    "Hazaribagh","Rayer Bazar","Jhigatola","Azimpur","Lalbagh","Chankharpul","Shahbagh","Paltan",
    "Motijheel","Dilkusha","Wari","Sutrapur","Kotwali","Bangshal","Chawkbazar","Sadarghat","Narinda",
    "Ramna","Eskaton","Moghbazar","Mouchak","Malibagh","Rampura","Banasree","Aftabnagar","Badda",
    "Khilgaon","Basabo","Shantinagar","Kakrail","Motsho Bhaban","Khilkhet","Nikunja","Airport",
    "Cantonment","Mohakhali","Banani DOHS","Baridhara DOHS","Bashundhara R/A","Notun Bazar",
    "Jatrabari","Demra","Keraniganj","Kamalapur","Sayedabad","Tikatuli","Arambagh","Paribagh"
]

# ===============================
# Theme-aware CSS (light/dark)
# ===============================
st.markdown("""
<style>
:root{
  --bg:#f5f7fb; --bg2:#ffffff; --text:#0f172a; --muted:#475569;
  --card:rgba(255,255,255,.85); --border:rgba(2,6,23,.08); --shadow:0 8px 22px rgba(2,6,23,.08); --chip:rgba(2,6,23,.04);
  --ring:#0891b2; --good:#16a34a; --warn:#d97706; --bad:#dc2626; --info:#2563eb;
}
@media (prefers-color-scheme: dark){
  :root{
    --bg:#0b1220; --bg2:#111827; --text:#e5e7eb; --muted:#94a3b8;
    --card:rgba(255,255,255,.06); --border:rgba(255,255,255,.10); --shadow:0 10px 30px rgba(0,0,0,.25); --chip:rgba(255,255,255,.09);
    --ring:#22d3ee; --good:#10b981; --warn:#f59e0b; --bad:#ef4444; --info:#3b82f6;
  }
}
html, body, [data-testid="stAppViewContainer"]{
  background:linear-gradient(135deg,var(--bg) 0%,var(--bg) 40%,var(--bg2) 100%) !important;
  color:var(--text);
}
.card{border-radius:18px;padding:18px 20px;background:var(--card);backdrop-filter:blur(8px);
      border:1px solid var(--border); box-shadow:var(--shadow);}
.grid{display:grid; gap:14px;}
.grid-4{grid-template-columns:repeat(4,minmax(0,1fr));}
.kpi{font-weight:800;font-size:2rem;line-height:1;margin:0;}
.kpi-label{margin:2px 0 0 0;color:var(--muted);font-size:.9rem;}
.ribbon{display:inline-flex;align-items:center;gap:.6rem;margin-top:8px}
.badge{padding:6px 12px;border-radius:9999px;font-weight:700;color:#fff;display:inline-flex;align-items:center;gap:.4rem}
.badge.red{background:linear-gradient(135deg,var(--bad),#b91c1c)}
.badge.amber{background:linear-gradient(135deg,var(--warn),#b45309)}
.badge.green{background:linear-gradient(135deg,var(--good),#047857)}
.badge.blue{background:linear-gradient(135deg,var(--info),#1d4ed8)}
.pill{padding:8px 14px;border-radius:9999px;background:var(--chip);border:1px solid var(--border);font-weight:700;color:var(--text);display:inline-flex;align-items:center}
.arrow{width:38px;height:38px;border-radius:10px;background:var(--chip);display:grid;place-items:center;border:1px solid var(--border);margin:0 8px}
.sep{height:1px;background:var(--border);margin:12px 0}
.banner{padding:10px 14px;border-radius:12px;display:inline-flex;align-items:center;gap:.6rem;font-weight:700}
.banner.ok{background:color-mix(in oklab, var(--good) 16%, transparent); color:color-mix(in oklab, var(--good) 85%, white); border:1px solid color-mix(in oklab, var(--good) 30%, transparent)}
.banner.warn{background:color-mix(in oklab, var(--warn) 16%, transparent); color:color-mix(in oklab, var(--warn) 85%, white); border:1px solid color-mix(in oklab, var(--warn) 30%, transparent)}
.route{display:flex;align-items:center;flex-wrap:wrap;gap:6px}
.ticket{display:grid;grid-template-columns:1.2fr .8fr;gap:16px}
.codebox{background:var(--bg);border:1px dashed var(--border);border-radius:12px;padding:10px}
</style>
""", unsafe_allow_html=True)

# ===============================
# Helper UI badge functions
# ===============================
def severity_badge(sev:str)->str:
    color = {"Mild":"green","Moderate":"amber","Severe":"red","Very Severe":"red"}.get(sev,"blue")
    return f'<span class="badge {color}">{sev}</span>'

def resource_badge(res:str)->str:
    color = "red" if res=="ICU" else "blue"
    return f'<span class="badge {color}">{res}</span>'

def sev_percent(sev:str)->int:
    return {"Mild":25,"Moderate":50,"Severe":75,"Very Severe":100}.get(sev,50)

# ===============================
# Utilities (DF cleanup, detection)
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
# Severity scoring
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
# Geocoding & distances (Dhaka-biased)
# ===============================
DHAKA_VIEWBOX = (90.30, 23.69, 90.50, 23.90)

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
    q = _with_bd_context(query)
    url = "https://nominatim.openstreetmap.org/search"
    headers = {"User-Agent": f"dscc-dengue-allocator/1.1 ({_contact_email_for_user_agent()})"}
    params = {
        "q": q,
        "format": "json",
        "limit": 1,
        "countrycodes": "bd",
        "viewbox": f"{DHAKA_VIEWBOX[0]},{DHAKA_VIEWBOX[1]},{DHAKA_VIEWBOX[2]},{DHAKA_VIEWBOX[3]}",
        "bounded": 1,
    }
    try:
        r = requests.get(url, headers=headers, params=params, timeout=15)
        r.raise_for_status()
        js = r.json()
        if not js:
            return None
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
    if not origin_ll or not dest_ll:
        return None
    o_lat, o_lon = origin_ll; d_lat, d_lon = dest_ll
    url = f"https://router.project-osrm.org/route/v1/driving/{o_lon},{o_lat};{d_lon},{d_lat}"
    params = {"overview": "false", "alternatives": "false", "steps": "false", "annotations": "false"}
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        js = r.json()
        if js.get("code") != "Ok" or not js.get("routes"):
            return None
        dist_km = js["routes"][0]["distance"] / 1000.0
        dur_min = js["routes"][0]["duration"] / 60.0
        return (dist_km, dur_min)
    except Exception:
        return None

HOSPITAL_PLACES = {
    "Dhaka Medical College Hospital": "Dhaka Medical College Hospital, Dhaka, Bangladesh",
    "SSMC & Mitford Hospital": "SSMC & Mitford Hospital, Dhaka, Bangladesh",
    "Bangladesh Shishu Hospital & Institute": "Bangladesh Shishu Hospital & Institute, Dhaka, Bangladesh",
    "Shaheed Suhrawardy Medical College hospital": "Shaheed Suhrawardy Medical College Hospital, Dhaka, Bangladesh",
    "Bangabandhu Shiekh Mujib Medical University": "BSMMU, Dhaka, Bangladesh",
    "Police Hospital, Rajarbagh": "Police Hospital, Rajarbagh, Dhaka, Bangladesh",
    "Mugda Medical College": "Mugda Medical College Hospital, Dhaka, Bangladesh",
    "Bangladesh Medical College Hospital": "Bangladesh Medical College Hospital, Dhaka, Bangladesh",
    "Holy Family Red Cresent Hospital": "Holy Family Red Crescent Medical College, Dhaka, Bangladesh",
    "BIRDEM Hospital": "BIRDEM General Hospital, Dhaka, Bangladesh",
    "Ibn Sina Hospital": "Ibn Sina Hospital Dhanmondi, Dhaka, Bangladesh",
    "Square Hospital": "Square Hospital, Dhaka, Bangladesh",
    "Samorita Hospital": "Samorita Hospital, Dhaka, Bangladesh",
    "Central Hospital Dhanmondi": "Central Hospital, Dhanmondi, Dhaka, Bangladesh",
    "Lab Aid Hospital": "Labaid Hospital, Dhaka, Bangladesh",
    "Green Life Medical Hospital": "Green Life Medical College Hospital, Dhaka, Bangladesh",
    "Sirajul Islam Medical College Hospital": "Sirajul Islam Medical College & Hospital, Dhaka, Bangladesh",
    "Ad-Din Medical College Hospital": "Ad-Din Medical College Hospital, Dhaka, Bangladesh",
}

@lru_cache(maxsize=256)
def geocode_hospital(ui_name: str):
    q1 = HOSPITAL_PLACES.get(ui_name, f"{ui_name}, Dhaka, Bangladesh")
    ll = geocode_nominatim(q1)
    if ll: return ll
    cleaned = re.sub(r"hospital|medical|college|&|,"," ", ui_name, flags=re.I).strip()
    return geocode_nominatim(cleaned)

# ===============================
# Placeholder for shared state & helpers used by both views
# ===============================
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

def hospitals_with_vacancy_on_date(date_any, bed_key: str) -> list[dict]:
    results = []
    for ui_name in HOSPITALS_UI:
        av_name = UI_TO_AV.get(ui_name) or ui_name
        rem = get_remaining(av_name, date_any, "ICU" if bed_key == "ICU" else "Normal")
        if rem and rem > 0:
            results.append({"ui_name": ui_name, "av_name": av_name, "remaining": rem})
    return results

def nearest_available_by_user_location_no_key(user_query: str, date_any, bed_key: str,
                                              top_k: int = 3, prefer_driving_eta: bool = False):
    user_ll = geocode_nominatim(user_query)
    if not user_ll:
        return [], None

    cand = hospitals_with_vacancy_on_date(date_any, bed_key)
    enriched = []
    for h in cand:
        h_ll = geocode_hospital(h["ui_name"])
        if not h_ll:
            continue
        dist_km = haversine_km(user_ll[0], user_ll[1], h_ll[0], h_ll[1])
        dur_min = None
        if prefer_driving_eta:
            osrm = osrm_drive(user_ll, h_ll)
            if osrm:
                dist_km, dur_min = osrm
        if dist_km > 80:
            continue
        enriched.append({
            **h,
            "distance_km": float(dist_km),
            "duration_min": (float(dur_min) if dur_min is not None else None),
            "lat": h_ll[0], "lng": h_ll[1],
        })
    enriched.sort(key=lambda x: x["distance_km"])
    return enriched[:top_k], user_ll

# ===============================
# Email helpers (multi-recipient)
# ===============================
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

def is_valid_email(addr: str) -> bool:
    return isinstance(addr, str) and bool(EMAIL_RE.match(addr.strip()))

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
                nearest_rows += f"""
                  <tr>
                    <td style="padding:8px 10px;border:1px solid #e2e8f0">{i}</td>
                    <td style="padding:8px 10px;border:1px solid #e2e8f0">{n['ui_name']}</td>
                    <td style="padding:8px 10px;border:1px solid #e2e8f0; text-align:right">{n['remaining']}</td>
                    <td style="padding:8px 10px;border:1px solid #e2e8f0; text-align:right">{n['distance_km']:.1f} km</td>
                    <td style="padding:8px 10px;border:1px solid #e2e8f0; text-align:right">{(int(round(n['duration_min'])) if n.get('duration_min') is not None else '—')}</td>
                  </tr>
                """
        else:
            nearest_rows = '<tr><td colspan="5" style="padding:8px 10px;border:1px solid #e2e8f0">No nearby vacancies right now.</td></tr>'
    nearest_block = f"""
      <h3 style="margin:18px 0 8px">Nearest hospitals from: {user_location}</h3>
      <table style="border-collapse:collapse;width:100%;max-width:720px">
        <tr style="background:#f8fafc">
          <th style="padding:8px 10px;border:1px solid #e2e8f0">#</th>
          <th style="padding:8px 10px;border:1px solid #e2e8f0; text-align:left">Hospital</th>
          <th style="padding:8px 10px;border:1px solid #e2e8f0; text-align:right">Beds/ICU free</th>
          <th style="padding:8px 10px;border:1px solid #e2e8f0; text-align:right">Distance</th>
          <th style="padding:8px 10px;border:1px solid #e2e8f0; text-align:right">ETA</th>
        </tr>
        {nearest_rows}
      </table>
    """ if nearest is not None else ""

    return f"""
    <div style="font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;line-height:1.6;color:#0f172a">
      <h2 style="margin:0 0 8px">Dengue Patient Allocation Summary</h2>
      <p style="margin:0 0 14px;color:#334155">Date: <strong>{dt_txt}</strong></p>
      <table style="border-collapse:collapse;width:100%;max-width:720px">
        <tr><td style="padding:10px 12px;border:1px solid #e2e8f0;background:#f8fafc;width:40%"><strong>Patient Age</strong></td><td style="padding:10px 12px;border:1px solid #e2e8f0">{patient_age}</td></tr>
        <tr><td style="padding:10px 12px;border:1px solid #e2e8f0;background:#f8fafc"><strong>Severity</strong></td><td style="padding:10px 12px;border:1px solid #e2e8f0">{severity}</td></tr>
        <tr><td style="padding:10px 12px;border:1px solid #e2e8f0;background:#f8fafc"><strong>Required Resource</strong></td><td style="padding:10px 12px;border:1px solid #e2e8f0">{resource}</td></tr>
        <tr><td style="padding:10px 12px;border:1px solid #e2e8f0;background:#f8fafc"><strong>Hospital Tried</strong></td><td style="padding:10px 12px;border:1px solid #e2e8f0">{tried_hospital_ui}</td></tr>
        <tr><td style="padding:10px 12px;border:1px solid #e2e8f0;background:#f8fafc"><strong>Assigned Hospital</strong></td><td style="padding:10px 12px;border:1px solid #e2e8f0">{assigned_hospital_av}</td></tr>
        <tr><td style="padding:10px 12px;border:1px solid #e2e8f0;background:#f8fafc"><strong>Distance from Tried Hospital</strong></td><td style="padding:10px 12px;border:1px solid #e2e8f0">{dist_txt}</td></tr>
        <tr><td style="padding:10px 12px;border:1px solid #e2e8f0;background:#f8fafc"><strong>Predicted Normal Beds Available</strong></td><td style="padding:10px 12px;border:1px solid #e2e8f0">{beds_avail}</td></tr>
        <tr><td style="padding:10px 12px;border:1px solid #e2e8f0;background:#f8fafc"><strong>Predicted ICU Beds Available</strong></td><td style="padding:10px 12px;border:1px solid #e2e8f0">{icu_avail}</td></tr>
      </table>
      {nearest_block}
      <p style="margin-top:16px;color:#475569">Distances are estimates (straight-line or OSRM driving if enabled); availability is based on your selected date.</p>
    </div>
    """

def send_email_multi(recipients, subject, html_body):
    """Send HTML email to multiple recipients using st.secrets['smtp']."""
    try:
        if "smtp" not in st.secrets:
            raise RuntimeError("SMTP secrets not configured in Streamlit (Settings → Secrets).")
        cfg = st.secrets["smtp"]
        smtp_host = cfg.get("host", "smtp.gmail.com")
        smtp_port = int(cfg.get("port", 465))
        smtp_user = cfg.get("user")
        smtp_pass = cfg.get("password")
        sender = cfg.get("sender", smtp_user)
        if not all([smtp_host, smtp_port, smtp_user, smtp_pass, sender]):
            raise RuntimeError("Incomplete SMTP config. Set host/port/user/password/sender in secrets.")

        # normalize recipients
        if isinstance(recipients, str):
            recipients = [p.strip() for p in re.split(r"[;,]", recipients) if p.strip()]
        recipients = [r for r in recipients if is_valid_email(r)]
        if not recipients:
            raise ValueError("No valid recipient emails provided.")

        context = ssl.create_default_context()
        if smtp_port == 465:
            server = smtplib.SMTP_SSL(smtp_host, smtp_port, context=context, timeout=20)
        else:
            server = smtplib.SMTP(smtp_host, smtp_port, timeout=20)
            server.ehlo(); server.starttls(context=context); server.ehlo()
        server.login(smtp_user, smtp_pass)

        sent_ok, sent_fail = [], []
        for rcp in recipients:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = sender
            msg["To"] = rcp
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

# ===============================
# Load prediction & location files (shared)
# ===============================
pred_file_path = Path("Predicted dataset AIO.xlsx")
loc_file_path  = Path("Location matrix.xlsx")
if not pred_file_path.exists() or not loc_file_path.exists():
    st.error("Required files not found in repo folder: 'Predicted dataset AIO.xlsx' and 'Location matrix.xlsx'")
    st.stop()

# Read them once (used by both views)
df_pred_raw = ensure_df(pd.read_excel(pred_file_path))
df_loc       = ensure_df(pd.read_excel(loc_file_path))

# Build availability (re-using your earlier utility)
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

# Default build (management controls can change granularity later)
granularity_default = "Daily"
interp_method_default = "linear"
try:
    availability = build_availability_from_predictions(df_pred_raw, granularity_default, interp_method_default)
except Exception as e:
    st.error(f"Error building availability: {e}")
    st.stop()

# Distance matrix + name maps
dist_mat = build_distance_matrix(df_loc)
DM_TO_AV, UI_TO_DM, UI_TO_AV = build_name_maps(availability, dist_mat, HOSPITALS_UI)

# ===============================
# UI: Landing — choose mode
# ===============================
st.title("🏥 Integrated Hospital Dengue Patient Allocation System")
st.write("Choose mode to continue:")

mode = st.radio("Mode", ("Management View", "Individual View"), horizontal=True)

# Optional management password if present in secrets; otherwise open
mgmt_password_secret = st.secrets.get("management_password") if st.secrets else None

# -------------------------------
# Management view function
# -------------------------------
def management_view():
    st.header("🔧 Management View")

    # Optional password check
    if mgmt_password_secret:
        pw = st.text_input("Management Password", type="password")
        if not pw:
            st.info("Enter management password to view dashboard.")
            return
        if pw != mgmt_password_secret:
            st.error("Incorrect password.")
            return

    # Sidebar options
    st.sidebar.header("Management Controls")
    gran = st.sidebar.selectbox("Time granularity", ["Daily","Weekly","Monthly"], index=0)
    interp = st.sidebar.selectbox("Interpolation (when expanding)", ["linear","ffill"], index=0)

    if st.sidebar.button("🔄 Rebuild availability"):
        try:
            global availability, dist_mat, DM_TO_AV, UI_TO_DM, UI_TO_AV
            availability = build_availability_from_predictions(df_pred_raw, gran, interp)
            dist_mat = build_distance_matrix(df_loc)
            DM_TO_AV, UI_TO_DM, UI_TO_AV = build_name_maps(availability, dist_mat, HOSPITALS_UI)
            st.sidebar.success("✅ Rebuilt successfully.")
        except Exception as e:
            st.sidebar.error(f"Error rebuilding data: {e}")

    # ===============================
    # Allocation Section
    # ===============================
    st.subheader("🧾 Patient Intake (Management)")

    with st.form("mgmt_allocation_form"):
        c1, c2, c3, c4 = st.columns([1.3, 1, 1, 1])
        all_dates = sorted(list(set([d for _, d in availability.index])))
        with c1:
            hospital_ui = st.selectbox("Hospital Name", HOSPITALS_UI)
            date_input = st.date_input("Date", value=all_dates[-1])
            weight = st.number_input("Weight (kg)", min_value=1.0, max_value=250.0, value=60.0)
        with c2:
            age = st.number_input("Age (years)", min_value=0, max_value=120, value=25)
            platelet = st.number_input("Platelet Count (/µL)", min_value=0, value=120000, step=1000)
        with c3:
            ns1_val = st.selectbox("NS1", [0,1], index=0)
            igm_val = st.selectbox("IgM", [0,1], index=0)
        with c4:
            igg_val = st.selectbox("IgG", [0,1], index=0)
            st.caption(f"Time: **{gran}** · Interp: **{interp if gran!='Monthly' else 'N/A'}**")
        submit = st.form_submit_button("🚑 Allocate (Management)")

    # Allocation logic
    if submit:
        p_score, s_score = compute_severity_score(age, ns1_val, igm_val, igg_val, platelet)
        severity = verdict_from_score(s_score)
        resource = required_resource(severity)
        bed_key = "ICU" if resource == "ICU" else "Normal"
        start_av = UI_TO_AV.get(hospital_ui) or hospital_ui
        remaining_here = get_remaining(start_av, date_input, bed_key)

        assigned_av = None; rerouted_distance = None; debug_checks = []
        if remaining_here > 0:
            assigned_av, rerouted_distance, note = start_av, None, "Assigned at selected hospital"
            available_status = "Yes"
        else:
            available_status = "No vacancy — rerouting..."
            assigned_av, rerouted_distance, err, debug_checks = find_reroute_nearest_first(
                hospital_ui, date_input, bed_key
            )
            note = f"Rerouted to {assigned_av}" if assigned_av else err

        if assigned_av:
            reserve_bed(assigned_av, date_input, bed_key, 1)
            increment_served(assigned_av, date_input)
            if assigned_av != (start_av or hospital_ui):
                log_reroute(hospital_ui, assigned_av, date_input)

        # Show summary cards
        st.markdown("### Allocation Result")
        st.markdown('<div class="grid grid-4">', unsafe_allow_html=True)
        st.markdown(f'<div class="card"><div class="kpi">{s_score}</div><div class="kpi-label">Severity Score</div><div class="ribbon">{severity_badge(severity)}</div></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="card"><div class="kpi">{resource}</div><div class="kpi-label">Resource Needed</div><div class="ribbon">{resource_badge(resource)}</div></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="card"><div class="kpi">{pd.to_datetime(date_input).date()}</div><div class="kpi-label">Date</div></div>', unsafe_allow_html=True)
        dist_txt = f"{float(rerouted_distance):.1f} km" if rerouted_distance is not None else "—"
        st.markdown(f'<div class="card"><div class="kpi">{dist_txt}</div><div class="kpi-label">Travel Distance</div></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.progress(sev_percent(severity))

        if debug_checks:
            with st.expander("🧪 Debug: Reroute checks"):
                st.dataframe(pd.DataFrame(debug_checks), use_container_width=True)

    # ===============================
    # Interactive Dashboard Section
    # ===============================
    st.markdown("---")
    st.header("📊 Interactive Hospital Dashboard")

    # Month selector
    all_dates = sorted(list(set([d for _, d in availability.index])))
    dashboard_date = st.date_input("Pick a date to view month", value=all_dates[-1])
    dashboard_month = month_str_from_date(dashboard_date)

    # Create DataFrame safely (fix)
    served_rows = [
        {"Hospital": h, "Served": cnt}
        for (h, m), cnt in st.session_state["served"].items()
        if m == dashboard_month
    ]
    if served_rows:
        served_df = pd.DataFrame(served_rows).sort_values("Served", ascending=False)
    else:
        served_df = pd.DataFrame(columns=["Hospital","Served"])

    total_served = int(served_df["Served"].sum()) if not served_df.empty else 0
    reroutes_this_month = [r for r in st.session_state["reroute_log"] if r["month"] == dashboard_month]
    total_reroutes = len(reroutes_this_month)
    total_hospitals = len(served_df)

    col1, col2, col3 = st.columns(3)
    col1.metric("🏥 Total Hospitals Active", total_hospitals)
    col2.metric("👥 Total Patients Served", total_served)
    col3.metric("🔁 Reroutes This Month", total_reroutes)

    if not served_df.empty:
        st.bar_chart(served_df.set_index("Hospital")["Served"])
        st.dataframe(served_df, use_container_width=True)

        # Pie chart (ICU vs Normal usage ratio if logged)
        if total_served:
            st.subheader("Resource Demand Overview")
            icu_share = served_df["Served"].sum() * 0.4  # dummy ratio
            data_pie = pd.DataFrame({
                "Resource": ["ICU", "General Bed"],
                "Count": [icu_share, total_served - icu_share]
            })
            st.altair_chart(
                st.altair_chart = (
                    alt.Chart(data_pie)
                    .mark_arc(innerRadius=40)
                    .encode(theta="Count", color="Resource", tooltip=["Resource", "Count"])
                ),
                use_container_width=True
            )
    else:
        st.info("No served data for this month yet.")

    # Reroute log expander
    with st.expander("🔁 View Reroute Log"):
        if reroutes_this_month:
            st.dataframe(pd.DataFrame(reroutes_this_month), use_container_width=True)
        else:
            st.write("No reroutes recorded this month.")


# -------------------------------
# Individual view function
# -------------------------------
def individual_view():
    st.header("🧍 Individual View — Patient Self-Assessment")
    st.write("Enter patient data and your location — we'll show nearest facilities with available beds appropriate to severity.")

    all_dates = sorted(list(set([d for _, d in availability.index])))
    min_d, max_d = all_dates[0], all_dates[-1]

    with st.form("indiv_form"):
        c1,c2,c3,c4 = st.columns([1.3,1,1,1])
        with c1:
            date_input = st.date_input("Date of assessment", value=max_d, min_value=min_d, max_value=max_d)
            pick_area = st.selectbox("Pick a Dhaka area (optional)", ["—"] + DHAKA_AREAS, index=0)
            user_location_query = st.text_input("Or type your exact location", placeholder="e.g., House 10, Road 5, Dhanmondi")
        with c2:
            age = st.number_input("Age (years)", min_value=0, max_value=120, value=25)
            ns1_val = st.selectbox("NS1", [0,1], index=0)
        with c3:
            platelet = st.number_input("Platelet Count (/µL)", min_value=0, value=120000, step=1000)
            igm_val = st.selectbox("IgM", [0,1], index=0)
        with c4:
            weight = st.number_input("Weight (kg)", min_value=1.0, max_value=250.0, value=60.0)
            igg_val = st.selectbox("IgG", [0,1], index=0)
        use_driving_eta = st.checkbox("Use driving ETA (beta)", value=False)
        email_addresses = st.text_input("📧 Recipient Email(s) (optional, ; or , or ;) ")
        email_opt_in = st.checkbox("Send results via email to recipients above", value=False)
        submit = st.form_submit_button("Check nearest facilities")

    if submit:
        p_score, s_score = compute_severity_score(age, ns1_val, igm_val, igg_val, platelet)
        severity = verdict_from_score(s_score)
        resource = required_resource(severity)
        bed_key = "ICU" if resource == "ICU" else "Normal"

        chosen_loc = user_location_query.strip() if user_location_query.strip() else (pick_area if pick_area != "—" else "")
        nearest_list = []
        user_ll = None
        if chosen_loc:
            try:
                nearest_list, user_ll = nearest_available_by_user_location_no_key(chosen_loc, date_input, bed_key, top_k=3, prefer_driving_eta=use_driving_eta)
            except Exception as e:
                st.warning(f"Could not get nearest hospitals: {e}")

        st.markdown("### Result")
        st.markdown(f"- **Severity:** {severity} {severity_badge(severity)}")
        st.markdown(f"- **Resource needed:** {resource} {resource_badge(resource)}")
        if not chosen_loc:
            st.info("Please pick an area or type a location to show nearest available hospitals.")
        else:
            if nearest_list:
                # If resource is ICU, ensure we highlight ICU available (we already filtered by bed_key)
                df_near = pd.DataFrame([{"Hospital": n["ui_name"], "Vacancy (Beds/ICU)": n["remaining"], "Distance (km)": round(n["distance_km"],1), "ETA (min)": (int(round(n["duration_min"])) if n.get("duration_min") is not None else None)} for n in nearest_list])
                st.dataframe(df_near, use_container_width=True)

                # Map w white user pin + red hospital pins
                layers = []
                if user_ll:
                    user_df = pd.DataFrame([{"name":"You","lat":user_ll[0],"lon":user_ll[1]}])
                    layers.append(pdk.Layer("ScatterplotLayer", user_df, get_position="[lon, lat]", get_radius=80, get_fill_color=[255,255,255,220], pickable=False))
                hosp_rows = []
                for n in nearest_list:
                    if n["lat"] and n["lng"]:
                        hosp_rows.append({"name": n["ui_name"], "lat": n["lat"], "lon": n["lng"]})
                if hosp_rows:
                    hosp_df = pd.DataFrame(hosp_rows)
                    layers.append(pdk.Layer("ScatterplotLayer", hosp_df, get_position="[lon, lat]", get_radius=70, get_fill_color=[255,0,0,220], pickable=True))
                if layers:
                    center_lat, center_lon = (user_ll if user_ll else (hosp_rows[0]["lat"], hosp_rows[0]["lon"]))
                    view_state = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=12, pitch=0)
                    st.pydeck_chart(pdk.Deck(map_style=None, initial_view_state=view_state, layers=layers), use_container_width=True)
            else:
                st.warning("No nearby hospitals with available beds of required type found for the selected date.")

        # Allow reservation & email optionally for individual (if they choose)
        if nearest_list:
            chosen_hospital = st.selectbox("Choose a hospital to reserve (optional)", [n["ui_name"] for n in nearest_list] + ["—"], index=0)
            if st.button("Reserve 1 bed at chosen hospital (temporary)"):
                if chosen_hospital and chosen_hospital != "—":
                    av_name = UI_TO_AV.get(chosen_hospital) or chosen_hospital
                    reserve_bed(av_name, date_input, "ICU" if bed_key=="ICU" else "Normal", 1)
                    increment_served(av_name, date_input)
                    st.success(f"Reserved 1 {'ICU' if bed_key=='ICU' else 'General'} bed at {chosen_hospital} for {pd.to_datetime(date_input).date()} (temporary).")
                else:
                    st.info("Select a hospital to reserve.")

        # Send email if opted
        beds_pred = icu_pred = 0
        # we won't assign an 'assigned_av' automatically here — this is individual suggestion
        assigned_av = nearest_list[0]["ui_name"] if nearest_list else None
        if assigned_av:
            assigned_counts = get_avail_counts(UI_TO_AV.get(assigned_av) or assigned_av, date_input)
            beds_pred = assigned_counts["beds_available"] if assigned_counts["beds_available"] is not None else 0
            icu_pred  = assigned_counts["icu_available"]  if assigned_counts["icu_available"]  is not None else 0

        if email_opt_in and email_addresses.strip():
            html = build_allocation_email_html(
                patient_age=age, severity=severity, resource=resource,
                tried_hospital_ui="(self-assessment)", assigned_hospital_av=(assigned_av or "—"),
                date_any=date_input, distance_km=(nearest_list[0]["distance_km"] if nearest_list else 0.0),
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

# -------------------------------
# Route to the selected view
# -------------------------------
if mode == "Management View":
    management_view()
else:
    individual_view()
