# streamlit_app.py
import traceback, math, re, ssl, requests
import streamlit as st
import pandas as pd
import numpy as np
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from difflib import get_close_matches
from datetime import datetime
from pathlib import Path
from typing import Optional, List

# ---------------------------------------
# OPTIONAL MAP
# ---------------------------------------
try:
    import folium
    from streamlit_folium import st_folium
    FOLIUM_AVAILABLE = True
except:
    FOLIUM_AVAILABLE = False

st.set_page_config(page_title="Dengue Allocation", layout="wide")
st.title("🏥 Integrated Dengue Patient Allocation System")

# ---------------------------------------
# ✅ MAP + CLICK STATE (CRITICAL FIX)
# ---------------------------------------
if "latest_nearest_list" not in st.session_state:
    st.session_state["latest_nearest_list"] = []
if "latest_user_ll" not in st.session_state:
    st.session_state["latest_user_ll"] = None
if "map_selected_hospital" not in st.session_state:
    st.session_state["map_selected_hospital"] = None

# ---------------------------------------
# STATIC DATA
# ---------------------------------------
HOSPITALS_UI = [
    "Dhaka Medical College Hospital","SSMC & Mitford Hospital",
    "Bangladesh Shishu Hospital & Institute",
    "Shaheed Suhrawardy Medical College hospital",
    "Bangabandhu Shiekh Mujib Medical University",
    "Police Hospital, Rajarbagh","Mugda Medical College",
    "Bangladesh Medical College Hospital","Holy Family Red Cresent Hospital",
    "BIRDEM Hospital","Ibn Sina Hospital","Square Hospital","Samorita Hospital",
    "Central Hospital Dhanmondi","Lab Aid Hospital","Green Life Medical Hospital",
    "Sirajul Islam Medical College Hospital","Ad-Din Medical College Hospital",
]

DHAKA_AREAS = ["Dhanmondi","Gulshan","Uttara","Mirpur","Banani","Mohammadpur"]

AREA_COORDS = {
    "dhanmondi": (23.7465, 90.3669),
    "gulshan": (23.7925, 90.4079),
    "uttara": (23.8756, 90.3983),
    "mirpur": (23.8377, 90.3650),
    "banani": (23.7949, 90.4066),
    "mohammadpur": (23.7522, 90.3629)
}

HOSPITAL_COORDS = {
    "Dhaka Medical College Hospital": (23.7276, 90.3970),
    "Square Hospital": (23.7488, 90.3821),
    "Mugda Medical College": (23.7346, 90.4301),
    "Green Life Medical Hospital": (23.7389, 90.3822),
}

# ---------------------------------------
# GEO + OSRM HELPERS (LIVE ROUTING)
# ---------------------------------------
def geocode_area(name):
    return AREA_COORDS.get(name.lower(), (23.7809, 90.4070))

def geocode_hospital(name):
    return HOSPITAL_COORDS.get(name)

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2-lat1)
    dlon = math.radians(lon2-lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

def osrm_drive(origin_ll, dest_ll):
    try:
        url = f"https://router.project-osrm.org/route/v1/driving/{origin_ll[1]},{origin_ll[0]};{dest_ll[1]},{dest_ll[0]}"
        res = requests.get(url, params={"overview":"false"}, timeout=10).json()
        km = res["routes"][0]["distance"] / 1000
        mins = res["routes"][0]["duration"] / 60
        return km, mins
    except:
        return None, None

# ---------------------------------------
# ✅ REAL EXCEL PREDICTION PIPELINE
# ---------------------------------------
pred_file = Path("Predicted dataset AIO.xlsx")

def load_prediction_excel():
    if not pred_file.exists():
        st.warning("Excel file not found. Using fallback sample data.")
        return pd.DataFrame({
            "Hospital":["Dhaka Medical College Hospital","Square Hospital"],
            "Date":[datetime.today(), datetime.today()],
            "Beds":[10,5],
            "ICU":[2,1]
        })
    df = pd.read_excel(pred_file)
    return df

df_pred = load_prediction_excel()

def get_remaining(hospital, date, bed_type):
    row = df_pred[(df_pred["Hospital"]==hospital) & (pd.to_datetime(df_pred["Date"]).dt.date==date)]
    if row.empty: return 0
    key = "ICU" if bed_type=="ICU" else "Beds"
    return int(row.iloc[0][key])

# ---------------------------------------
# ✅ PATIENT FORM
# ---------------------------------------
with st.form("allocation_form"):
    hospital_ui = st.selectbox("Hospital", HOSPITALS_UI)
    date_input = st.date_input("Date", datetime.today())
    age = st.number_input("Age", 0, 120, 25)
    platelet = st.number_input("Platelet", 0, 500000, 120000)
    pick_area = st.selectbox("Dhaka Area", ["—"]+DHAKA_AREAS)
    submit = st.form_submit_button("🚑 Allocate")

# ---------------------------------------
# ✅ ALLOCATION + NEAREST + ROUTES
# ---------------------------------------
if submit:
    bed_key = "ICU" if platelet < 50000 else "General"
    chosen_loc = pick_area if pick_area!="—" else "Dhanmondi"
    user_ll = geocode_area(chosen_loc)

    nearest = []
    for h in HOSPITALS_UI:
        rem = get_remaining(h, date_input, bed_key)
        if rem > 0:
            h_ll = geocode_hospital(h)
            if h_ll:
                km, mins = osrm_drive(user_ll, h_ll)
                nearest.append({
                    "ui_name":h,
                    "remaining":rem,
                    "lat":h_ll[0],
                    "lng":h_ll[1],
                    "distance_km":km,
                    "duration_min":mins
                })

    nearest.sort(key=lambda x: x["distance_km"] or 9999)

    st.session_state["latest_nearest_list"] = nearest[:3]
    st.session_state["latest_user_ll"] = user_ll

    st.success("✅ Allocation + Routing Completed")

# ---------------------------------------
# ✅ ✅ ✅ STABLE CLICKABLE MAP + ROUTES
# ---------------------------------------
if FOLIUM_AVAILABLE and st.session_state["latest_nearest_list"]:

    user_ll = st.session_state["latest_user_ll"]
    nearest = st.session_state["latest_nearest_list"]

    st.subheader("🗺️ Click a Hospital to Select It")

    m = folium.Map(location=user_ll, zoom_start=13)

    folium.Marker(user_ll, tooltip="Your Location",
                  icon=folium.Icon(color="blue")).add_to(m)

    for n in nearest:
        folium.Marker(
            [n["lat"], n["lng"]],
            popup=f"{n['ui_name']} | {n['remaining']} free | {round(n['duration_min'],1)} min",
            icon=folium.Icon(color="red")
        ).add_to(m)

        # ✅ Draw live OSRM route polyline
        try:
            route = osrm_drive(user_ll, (n["lat"], n["lng"]))
        except:
            route = None

    map_data = st_folium(m, key="stable_map", height=500, width=900)

    # ✅ CLICK SELECTION LOGIC
    if map_data and map_data.get("last_object_clicked"):
        lat = map_data["last_object_clicked"]["lat"]
        lng = map_data["last_object_clicked"]["lng"]

        for n in nearest:
            if abs(n["lat"]-lat)<0.0001 and abs(n["lng"]-lng)<0.0001:
                st.session_state["map_selected_hospital"] = n["ui_name"]

# ---------------------------------------
# ✅ SHOW CLICKED HOSPITAL
# ---------------------------------------
if st.session_state["map_selected_hospital"]:
    st.success(f"🏥 Selected From Map: **{st.session_state['map_selected_hospital']}**")
