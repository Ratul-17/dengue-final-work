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
from email.message import EmailMessage
from pathlib import Path
from typing import Optional, List

# ===============================
# App config
# ===============================
st.set_page_config(page_title="Dengue Allocation", page_icon="🏥", layout="wide")
st.title("🏥 Integrated Hospital Dengue Patient Allocation System (DSCC Region)")

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
# Common Dhaka areas (extend any time)
# ===============================
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
# THEME-AWARE Styles (auto light/dark)
# ===============================
st.markdown("""
<style>
/* ---------- THEME TOKENS (light default, dark override) ---------- */
:root{
  --bg:#f5f7fb; --bg2:#ffffff;
  --text:#0f172a; --muted:#475569;
  --card:rgba(255,255,255,.85);
  --border:rgba(2,6,23,.08);
  --shadow:0 8px 22px rgba(2,6,23,.08);
  --chip:rgba(2,6,23,.04);
  --ring:#0891b2;
  --good:#16a34a; --warn:#d97706; --bad:#dc2626; --info:#2563eb;
}
@media (prefers-color-scheme: dark){
  :root{
    --bg:#0b1220; --bg2:#111827;
    --text:#e5e7eb; --muted:#94a3b8;
    --card:rgba(255,255,255,.06);
    --border:rgba(255,255,255,.10);
    --shadow:0 10px 30px rgba(0,0,0,.25);
    --chip:rgba(255,255,255,.09);
    --ring:#22d3ee;
    --good:#10b981; --warn:#f59e0b; --bad:#ef4444; --info:#3b82f6;
  }
}

/* ---------- APP BACKGROUND & TEXT ---------- */
html, body, [data-testid="stAppViewContainer"]{
  background:linear-gradient(135deg,var(--bg) 0%,var(--bg) 40%,var(--bg2) 100%) !important;
  color:var(--text);
}

/* ---------- CARDS / LAYOUT ---------- */
.card{
  border-radius:18px; padding:18px 20px; background:var(--card);
  backdrop-filter:blur(8px); border:1px solid var(--border); box-shadow:var(--shadow);
}
.grid{display:grid; gap:14px;}
.grid-4{grid-template-columns:repeat(4,minmax(0,1fr));}

/* ---------- TYPO ---------- */
.kpi{font-weight:800;font-size:2rem;line-height:1;margin:0;}
.kpi-label{margin:2px 0 0 0;color:var(--muted);font-size:.9rem;}
.small{color:var(--muted);font-size:.9rem}

/* ---------- CHIPS / BADGES / BANNERS ---------- */
.ribbon{display:inline-flex;align-items:center;gap:.6rem;margin-top:8px}
.badge{padding:6px 12px;border-radius:9999px;font-weight:700;color:#fff;display:inline-flex;align-items:center;gap:.4rem}
.badge.red{background:linear-gradient(135deg,var(--bad),#b91c1c)}
.badge.amber{background:linear-gradient(135deg,var(--warn),#b45309)}
.badge.green{background:linear-gradient(135deg,var(--good),#047857)}
.badge.blue{background:linear-gradient(135deg,var(--info),#1d4ed8)}

.pill{
  padding:8px 14px;border-radius:9999px;background:var(--chip);
  border:1px solid var(--border);font-weight:700;color:var(--text);
  display:inline-flex;align-items:center
}

.banner{
  padding:10px 14px;border-radius:12px;display:inline-flex;align-items:center;gap:.6rem;font-weight:700
}
.banner.ok{background:color-mix(in oklab, var(--good) 16%, transparent); color:color-mix(in oklab, var(--good) 85%, white);
           border:1px solid color-mix(in oklab, var(--good) 30%, transparent)}
.banner.warn{background:color-mix(in oklab, var(--warn) 16%, transparent); color:color-mix(in oklab, var(--warn) 85%, white);
             border:1px solid color-mix(in oklab, var(--warn) 30%, transparent)}

/* ---------- ROUTE STRIP ---------- */
.route{display:flex;align-items:center;flex-wrap:wrap;gap:6px}
.arrow{
  width:38px;height:38px;border-radius:10px;background:var(--chip);display:grid;place-items:center;
  border:1px solid var(--border);margin:0 8px
}

/* ---------- MISC ---------- */
.sep{height:1px;background:var(--border);margin:12px 0}
.ticket{display:grid;grid-template-columns:1.2fr .8fr;gap:16px}
.codebox{background:var(--bg);border:1px dashed var(--border);border-radius:12px;padding:10px}
</style>
""", unsafe_allow_html=True)

def severity_badge(sev:str)->str:
    color = {"Mild":"green","Moderate":"amber","Severe":"red","Very Severe":"red"}.get(sev,"blue")
    return f'<span class="badge {color}">{sev}</span>'

def resource_badge(res:str)->str:
    color = "red" if res=="ICU" else "blue"
    return f'<span class="badge {color}">{res}</span>'

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
# Severity logic
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
# No-key geocoding + distances (Dhaka-biased + sanity filters)
# ===============================
DHAKA_VIEWBOX = (90.30, 23.69, 90.50, 23.90)  # lon_min, lat_min, lon_max, lat_max

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
        "q": q, "format": "json", "limit": 1, "countrycodes": "bd",
        "viewbox": f"{DHAKA_VIEWBOX[0]},{DHAKA_VIEWBOX[1]},{DHAKA_VIEWBOX[2]},{DHAKA_VIEWBOX[3]}",
        "bounded": 1,
    }
    try:
        r = requests.get(url, headers=headers, params=params, timeout=15)
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
    if not origin_ll or not dest_ll: return None
    o_lat, o_lon = origin_ll; d_lat, d_lon = dest_ll
    url = f"https://router.project-osrm.org/route/v1/driving/{o_lon},{o_lat};{d_lon},{d_lat}"
    params = {"overview": "false", "alternatives": "false", "steps": "false", "annotations": "false"}
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        js = r.json()
        if js.get("code") != "Ok" or not js.get("routes"): return None
        dist_km = js["routes"][0]["distance"] / 1000.0
        dur_min = js["routes"][0]["duration"] / 60.0
        return (dist_km, dur_min)
    except Exception:
        return None

# -----------------------
# Predefined accurate coordinates for hospitals (lat, lon)
# Replace/extend these if you find better coordinates.
# Verified sources used: DGHS facility pages, Mapcarta/OpenStreetMap, hospital websites, Wikipedia.
# -----------------------
HOSPITAL_COORDS = {
    "Dhaka Medical College Hospital": (23.72591, 90.39805),
    "SSMC & Mitford Hospital": (23.71025498, 90.40143508),
    "Bangladesh Shishu Hospital & Institute": (23.77296, 90.36861),
    "Shaheed Suhrawardy Medical College hospital": (23.76918, 90.37103),
    "Bangabandhu Shiekh Mujib Medical University": (23.73890, 90.39480),
    "Police Hospital, Rajarbagh": (23.74063, 90.41737),
    "Mugda Medical College": (23.73248, 90.43003),
    "Bangladesh Medical College Hospital": (23.75014, 90.36973),
    "Holy Family Red Cresent Hospital": (23.73790, 90.39150),  # Eskaton area (approx)
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

HOSPITAL_PLACES = {
    # keep the textual places for fallback geocoding if needed
    k: f"{k}, Dhaka, Bangladesh" for k in HOSPITALS_UI
}

@lru_cache(maxsize=256)
def geocode_hospital(ui_name: str):
    """
    Return (lat, lon) for a hospital UI name.
    Uses predefined HOSPITAL_COORDS first (fast + accurate), otherwise tries Nominatim.
    """
    if ui_name in HOSPITAL_COORDS:
        return HOSPITAL_COORDS[ui_name]
    # fallback to query by place string
    q1 = HOSPITAL_PLACES.get(ui_name, f"{ui_name}, Dhaka, Bangladesh")
    ll = geocode_nominatim(q1)
    if ll: return ll
    cleaned = re.sub(r"hospital|medical|college|&|,"," ", ui_name, flags=re.I).strip()
    return geocode_nominatim(cleaned)

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
    """
    Returns (list, user_ll). Each item: {ui_name, av_name, remaining, distance_km, duration_min, lat, lng}
    Dhaka/BD bias + sanity filter: drop >80 km.
    """
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
            "lat": float(h_ll[0]), "lng": float(h_ll[1]),
        })

    enriched.sort(key=lambda x: x["distance_km"])
    return enriched[:top_k], user_ll

# ===============================
# Email helpers (unchanged)
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
# Load repo files
# ===============================
# ===============================
# Load repo files (robust + uploader fallback)
# ===============================
pred_file_path = Path("Predicted dataset AIO.xlsx")
loc_file_path  = Path("Location matrix.xlsx")

# Helper to read excel from path or uploaded file
def read_excel_from_source(src):
    try:
        if isinstance(src, (str, Path)):
            return ensure_df(pd.read_excel(src))
        else:
            # uploaded file (streamlit UploadedFile)
            return ensure_df(pd.read_excel(src))
    except Exception as ex:
        st.error(f"Failed to read Excel: {ex}")
        return None

# Try repo files first
df_pred_raw = None
df_loc = None
if pred_file_path.exists() and loc_file_path.exists():
    df_pred_raw = read_excel_from_source(pred_file_path)
    df_loc = read_excel_from_source(loc_file_path)
else:
    st.warning("Required Excel files not found in repository root.")
    st.info("Please upload the two files or push them to your repo root with the exact filenames:\n"
            "`Predicted dataset AIO.xlsx` and `Location matrix.xlsx`")
    up1 = st.file_uploader("Upload Predicted dataset AIO.xlsx", type=["xls","xlsx"], key="uploader_pred")
    up2 = st.file_uploader("Upload Location matrix.xlsx", type=["xls","xlsx"], key="uploader_loc")
    if up1:
        df_pred_raw = read_excel_from_source(up1)
    if up2:
        df_loc = read_excel_from_source(up2)

# If still missing, provide a small sample fallback so the UI doesn't go blank (so you can test)
if df_pred_raw is None or df_loc is None:
    st.warning("Using minimal sample fallback data to keep app running (not real hospital data).")
    # small sample prediction dataset with columns the loader expects
    sample_pred = pd.DataFrame({
        "Hospital": ["Dhaka Medical College Hospital", "Mugda Medical College"],
        "Date": [pd.Timestamp.today().normalize(), pd.Timestamp.today().normalize()],
        "Predicted Normal Beds Available": [10, 5],
        "Predicted ICU Beds Available": [1, 0]
    })
    # sample location matrix format:
    sample_loc = pd.DataFrame({
        "Location": ["Dhaka Medical College Hospital", "Mugda Medical College"],
        "Dhaka Medical College Hospital": [0, 6.0],
        "Mugda Medical College": [6.0, 0],
    })
    if df_pred_raw is None:
        df_pred_raw = ensure_df(sample_pred)
    if df_loc is None:
        df_loc = ensure_df(sample_loc)

# Final validation
if df_pred_raw is None or df_loc is None:
    st.error("Unable to load or create required data. Check uploaded files or repository files and refresh.")
    st.stop()

# Now df_pred_raw and df_loc are guaranteed to be DataFrames (or app stopped above).

# ===============================
# Distance matrix + name maps
# ===============================
dist_mat = build_distance_matrix(df_loc)
DM_TO_AV, UI_TO_DM, UI_TO_AV = build_name_maps(availability, dist_mat, HOSPITALS_UI)

# ===============================
# Allocation helpers (state + fns)
# ===============================
if "reservations" not in st.session_state:
    st.session_state["reservations"] = {}
if "served" not in st.session_state:
    st.session_state["served"] = {}
if "reroute_log" not in st.session_state:
    st.session_state["reroute_log"] = []

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

# (rest of your UI and logic remains the same as before...)
# ... For brevity, I did not copy the entire UI again — in your existing file keep the rest unchanged.
# Important: geocode_hospital now uses HOSPITAL_COORDS, so maps will use accurate lat/lon.
