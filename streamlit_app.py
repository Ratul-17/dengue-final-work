# ================================
# ✅ FULL STABLE VERSION (MERGED)
# ================================

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

# ✅✅✅ MAP STATE (PERMANENT FIX)
if "latest_nearest_list" not in st.session_state:
    st.session_state["latest_nearest_list"] = []
if "latest_user_ll" not in st.session_state:
    st.session_state["latest_user_ll"] = None

# -------------------------
# Static lists
# -------------------------
HOSPITALS_UI = [
    "Dhaka Medical College Hospital","SSMC & Mitford Hospital",
    "Bangladesh Shishu Hospital & Institute","Shaheed Suhrawardy Medical College hospital",
    "Bangabandhu Shiekh Mujib Medical University","Police Hospital, Rajarbagh",
    "Mugda Medical College","Bangladesh Medical College Hospital",
    "Holy Family Red Cresent Hospital","BIRDEM Hospital","Ibn Sina Hospital",
    "Square Hospital","Samorita Hospital","Central Hospital Dhanmondi",
    "Lab Aid Hospital","Green Life Medical Hospital",
    "Sirajul Islam Medical College Hospital","Ad-Din Medical College Hospital"
]

DHAKA_AREAS = [
    "Dhanmondi","Mohammadpur","Gulshan","Banani","Baridhara","Uttara",
    "Mirpur","Tejgaon","Farmgate","Malibagh","Khilgaon","Paltan",
    "Motijheel","Jatrabari","Rampura","Badda","Moghbazar","Shahbagh"
]

# -------------------------
# Severity Logic
# -------------------------
def compute_platelet_score(platelet):
    if platelet >= 150_000: return 0
    if platelet >= 100_000: return 1
    if platelet >= 50_000: return 2
    if platelet >= 20_000: return 3
    return 4

def compute_severity_score(age, ns1, igm, igg, platelet):
    p_score = compute_platelet_score(platelet)
    age_weight = 1 if (age < 15 or age > 60) else 0
    secondary = 1 if (igg == 1 and (ns1 == 1 or igm == 1)) else 0
    severity_score = min(4, round(p_score + age_weight + secondary))
    return p_score, severity_score

def verdict_from_score(score):
    if score >= 3: return "Very Severe"
    if score == 2: return "Severe"
    if score == 1: return "Moderate"
    return "Mild"

def required_resource(severity):
    return "ICU" if severity in ("Severe", "Very Severe") else "General Bed"

# -------------------------
# Geo Helpers
# -------------------------
DHAKA_CENTER = (23.7809, 90.4070)
AREA_COORDS = {
    "dhanmondi": (23.7465, 90.3669),
    "gulshan": (23.7925, 90.4079),
    "uttara": (23.8756, 90.3983)
}
HOSPITAL_COORDS = {
    "Dhaka Medical College Hospital": (23.7276, 90.3970),
    "Square Hospital": (23.7488, 90.3821),
    "Mugda Medical College": (23.7346, 90.4301)
}

def geocode_area(area): return AREA_COORDS.get(area.lower(), DHAKA_CENTER)
def geocode_hospital(h): return HOSPITAL_COORDS.get(h)

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2-lat1)
    dlon = math.radians(lon2-lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

# -------------------------
# Dummy Availability (REPLACE WITH YOUR EXCEL DATA)
# -------------------------
availability = {
    ("Dhaka Medical College Hospital", datetime.today().date(), "General"): 10,
    ("Dhaka Medical College Hospital", datetime.today().date(), "ICU"): 2,
    ("Square Hospital", datetime.today().date(), "General"): 5,
    ("Square Hospital", datetime.today().date(), "ICU"): 1,
}

def get_remaining(h, d, b):
    return availability.get((h, d, b), 0)

# -------------------------
# Patient Intake Form
# -------------------------
with st.form("allocation"):
    hospital_ui = st.selectbox("Hospital", HOSPITALS_UI)
    date_input = st.date_input("Date", datetime.today())
    age = st.number_input("Age", 0, 120, 25)
    platelet = st.number_input("Platelet", 0, 500000, 120000)
    ns1_val = st.selectbox("NS1", [0,1])
    igm_val = st.selectbox("IgM", [0,1])
    igg_val = st.selectbox("IgG", [0,1])
    pick_area = st.selectbox("Area", ["—"] + DHAKA_AREAS)

    submit = st.form_submit_button("🚑 Allocate")

# -------------------------
# Allocation + Stable Map Storage
# -------------------------
if submit:
    _, sev_score = compute_severity_score(age, ns1_val, igm_val, igg_val, platelet)
    severity = verdict_from_score(sev_score)
    bed_key = "ICU" if severity in ("Severe","Very Severe") else "General"

    chosen_loc = pick_area if pick_area != "—" else "Dhanmondi"
    user_ll = geocode_area(chosen_loc)

    nearest_list = []
    for h in HOSPITALS_UI:
        rem = get_remaining(h, date_input, bed_key)
        if rem > 0:
            h_ll = geocode_hospital(h)
            if h_ll:
                dist = haversine_km(user_ll[0], user_ll[1], h_ll[0], h_ll[1])
                nearest_list.append({
                    "ui_name": h,
                    "remaining": rem,
                    "lat": h_ll[0],
                    "lng": h_ll[1],
                    "distance_km": dist
                })

    nearest_list.sort(key=lambda x: x["distance_km"])

    # ✅✅✅ STORE IN SESSION (FINAL FIX)
    st.session_state["latest_nearest_list"] = nearest_list[:3]
    st.session_state["latest_user_ll"] = user_ll

    st.success("✅ Allocation Completed")

# -------------------------
# ✅✅✅ STABLE MAP RENDER (NEVER BLINKS)
# -------------------------
if FOLIUM_AVAILABLE and st.session_state["latest_nearest_list"]:
    user_ll = st.session_state["latest_user_ll"]
    nearest_list = st.session_state["latest_nearest_list"]

    st.markdown("### 🗺️ Nearest Hospitals (Stable Map)")

    m = folium.Map(location=user_ll, zoom_start=13)

    folium.CircleMarker(user_ll, radius=9, color="blue", fill=True,
                        tooltip="Your Location").add_to(m)

    for n in nearest_list:
        folium.Marker(
            location=[n["lat"], n["lng"]],
            popup=f"{n['ui_name']} — {n['remaining']} free",
            icon=folium.Icon(color="red")
        ).add_to(m)

    st_folium(m, key="stable_map", height=500, width=900)
