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
from difflib import get_close_matches
from datetime import datetime
from pathlib import Path
from typing import Optional, List

# optional folium map
try:
    import folium
    from streamlit_folium import st_folium
    FOLIUM_AVAILABLE = True
except Exception:
    FOLIUM_AVAILABLE = False

st.set_page_config(page_title="Dengue Allocation", page_icon="🏥", layout="wide")
st.title("🏥 Integrated Hospital Dengue Patient Allocation System (DSCC Region)")

# ✅✅✅ MAP PERSISTENCE (BLINK FIX)
if "latest_nearest_list" not in st.session_state:
    st.session_state["latest_nearest_list"] = []

if "latest_user_ll" not in st.session_state:
    st.session_state["latest_user_ll"] = None

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
    "Banani DOHS","Baridhara DOHS","Bashundhara R/A","Notun Bazar","Jatrabari","Demra","Keraniganj",
    "Kamalapur","Sayedabad","Tikatuli","Arambagh","Paribagh"
]

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

def compute_severity_score(age: int, ns1: int, igm: int, igg: int, platelet: int):
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
# Geocoding
# -------------------------
DHAKA_CENTER = (23.780887, 90.407049)

AREA_COORDS = {
    "dhanmondi": (23.7465, 90.3669),
    "gulshan": (23.7925, 90.4079),
    "uttara": (23.8756, 90.3983),
}

HOSPITAL_COORDS = {
    "Dhaka Medical College Hospital": (23.7276, 90.3970),
    "Square Hospital": (23.7488, 90.3821),
    "Mugda Medical College": (23.7346, 90.4301),
}

def geocode_area(area_name: str):
    return AREA_COORDS.get(area_name.lower().strip(), DHAKA_CENTER)

def geocode_hospital(ui_name: str):
    return HOSPITAL_COORDS.get(ui_name)

def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0088
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    dφ = math.radians(lat2 - lat1)
    dλ = math.radians(lon2 - lon1)
    a = math.sin(dφ/2)**2 + math.cos(φ1)*math.cos(φ2)*math.sin(dλ/2)**2
    return R * (2 * math.atan2(math.sqrt(a), math.sqrt(1-a)))

# -------------------------
# Dummy availability (keeps app runnable)
# -------------------------
availability = {
    ("Dhaka Medical College Hospital", datetime.today().date()): {"Normal": 5, "ICU": 1},
    ("Square Hospital", datetime.today().date()): {"Normal": 3, "ICU": 1},
    ("Mugda Medical College", datetime.today().date()): {"Normal": 0, "ICU": 0},
}

def get_remaining(hospital, date, bed_type):
    return availability.get((hospital, date), {}).get(bed_type, 0)

# -------------------------
# Patient Intake UI
# -------------------------
with st.form("allocation_form"):
    hospital_ui = st.selectbox("Hospital", HOSPITALS_UI)
    date_input = st.date_input("Date", value=datetime.today().date())
    age = st.number_input("Age", 0, 120, 25)
    platelet = st.number_input("Platelets", 0, 500000, 120000)
    ns1_val = st.selectbox("NS1", [0,1])
    igm_val = st.selectbox("IgM", [0,1])
    igg_val = st.selectbox("IgG", [0,1])
    pick_area = st.selectbox("Pick Dhaka Area", ["—"] + DHAKA_AREAS)
    submit = st.form_submit_button("🚑 Allocate")

# -------------------------
# Allocation + MAP DATA SAVE
# -------------------------
if submit:
    _, s_score = compute_severity_score(age, ns1_val, igm_val, igg_val, platelet)
    severity = verdict_from_score(s_score)
    bed_key  = "ICU" if severity in ("Severe", "Very Severe") else "Normal"

    chosen_loc = pick_area if pick_area != "—" else "Dhanmondi"
    user_ll = geocode_area(chosen_loc)

    nearest_list = []
    for ui_name in HOSPITALS_UI:
        rem = get_remaining(ui_name, date_input, bed_key)
        if rem > 0:
            h_ll = geocode_hospital(ui_name)
            if h_ll:
                dist = haversine_km(user_ll[0], user_ll[1], h_ll[0], h_ll[1])
                nearest_list.append({
                    "ui_name": ui_name,
                    "remaining": rem,
                    "lat": h_ll[0],
                    "lng": h_ll[1],
                    "distance_km": dist
                })

    nearest_list.sort(key=lambda x: x["distance_km"])

    # ✅✅✅ STORE MAP DATA (BLINK FIX)
    st.session_state["latest_nearest_list"] = nearest_list[:3]
    st.session_state["latest_user_ll"] = user_ll

    st.success("Allocation Completed")

# -------------------------
# ✅✅✅ STABLE PERSISTENT MAP (NO BLINK EVER)
# -------------------------
if FOLIUM_AVAILABLE and st.session_state["latest_nearest_list"]:

    user_ll = st.session_state["latest_user_ll"]
    nearest_list = st.session_state["latest_nearest_list"]

    st.markdown("### 🗺️ Nearest Hospitals (Stable Map View)")

    m = folium.Map(location=[user_ll[0], user_ll[1]], zoom_start=13)

    folium.CircleMarker(
        [user_ll[0], user_ll[1]],
        radius=9,
        color="blue",
        fill=True,
        tooltip="Your Location"
    ).add_to(m)

    for n in nearest_list:
        folium.Marker(
            [n["lat"], n["lng"]],
            popup=f"{n['ui_name']} — {n['remaining']}",
            icon=folium.Icon(color="red")
        ).add_to(m)

    st_folium(m, key="stable_map", height=500, width=900)
