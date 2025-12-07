# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import math
import requests
from functools import lru_cache
import folium
from streamlit_folium import st_folium

# ===============================
# APP CONFIG
# ===============================
st.set_page_config(page_title="Dengue Allocation", page_icon="🏥", layout="wide")
st.title("🏥 Dengue Patient Allocation System (Free OSM Maps)")

# ===============================
# HOSPITAL LIST
# ===============================
HOSPITALS_UI = [
    "Dhaka Medical College Hospital",
    "SSMC & Mitford Hospital",
    "Bangladesh Shishu Hospital & Institute",
    "Shaheed Suhrawardy Medical College Hospital",
    "Bangabandhu Sheikh Mujib Medical University",
    "Police Hospital, Rajarbagh",
    "Mugda Medical College Hospital",
    "Bangladesh Medical College Hospital",
    "Holy Family Red Crescent Hospital",
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
    "Dhanmondi", "Gulshan", "Uttara", "Mirpur", "Banani", "Mohammadpur",
    "Motijheel", "Rampura", "Badda", "Khilgaon", "Shyamoli", "Farmgate",
    "Malibagh", "Moghbazar"
]

# ===============================
# ✅ FREE ONLINE GEOCODING (NOMINATIM)
# ===============================
@lru_cache(maxsize=256)
def geocode(place):
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": f"{place}, Dhaka, Bangladesh",
        "format": "json",
        "limit": 1
    }
    headers = {"User-Agent": "dengue-app"}
    try:
        r = requests.get(url, params=params, headers=headers, timeout=15)
        if r.status_code == 200 and r.json():
            return float(r.json()[0]["lat"]), float(r.json()[0]["lon"])
    except:
        pass
    return None

# ===============================
# DISTANCE FORMULA
# ===============================
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    p = math.pi / 180
    a = 0.5 - math.cos((lat2-lat1)*p)/2 + math.cos(lat1*p)*math.cos(lat2*p)*(1-math.cos((lon2-lon1)*p))/2
    return 2 * R * math.asin(math.sqrt(a))

# ===============================
# FAKE AVAILABILITY DATA (REPLACE WITH REAL LATER)
# ===============================
@st.cache_data
def fake_availability():
    rows = []
    for h in HOSPITALS_UI:
        rows.append({
            "Hospital": h,
            "Beds": np.random.randint(5, 40),
            "ICU": np.random.randint(1, 10)
        })
    return pd.DataFrame(rows)

availability = fake_availability()

# ===============================
# INPUT FORM
# ===============================
with st.form("allocation"):
    col1, col2 = st.columns(2)
    with col1:
        hospital_ui = st.selectbox("Hospital", HOSPITALS_UI)
        pick_area = st.selectbox("Pick Dhaka Area", DHAKA_AREAS)
    with col2:
        age = st.number_input("Age", 0, 120, 25)
        platelet = st.number_input("Platelet", 0, 300000, 120000)

    submit = st.form_submit_button("🚑 Allocate")

# ===============================
# PROCESS + FREE MAP
# ===============================
if submit:
    st.subheader("✅ Allocation Result")

    # Severity
    if platelet < 50000:
        severity = "Severe"
        req = "ICU"
    elif platelet < 100000:
        severity = "Moderate"
        req = "Normal"
    else:
        severity = "Mild"
        req = "Normal"

    st.success(f"Severity: {severity} | Required: {req}")

    # ✅ USER LOCATION
    user_ll = geocode(pick_area)
    if not user_ll:
        st.error("❌ Failed to detect your area location.")
        st.stop()

    # ===============================
    # ✅ HOSPITAL GEO + NEAREST 3
    # ===============================
    nearest_list = []

    for _, row in availability.iterrows():
        h_ll = geocode(row["Hospital"])
        if h_ll:
            dist = haversine(user_ll[0], user_ll[1], h_ll[0], h_ll[1])

            nearest_list.append({
                "Hospital": row["Hospital"],
                "Beds": row["Beds"],
                "ICU": row["ICU"],
                "lat": h_ll[0],
                "lon": h_ll[1],
                "dist": round(dist, 2)
            })

    if not nearest_list:
        st.error("❌ Could not geocode any hospital.")
        st.stop()

    nearest_list = sorted(nearest_list, key=lambda x: x["dist"])[:3]

    # ===============================
    # TABLE
    # ===============================
    st.dataframe(pd.DataFrame(nearest_list)[["Hospital", "Beds", "ICU", "dist"]])

    # ===============================
    # ✅✅✅ FREE OPENSTREETMAP MAP (FOLIUM)
    # ===============================
    st.subheader("🗺️ Nearest Hospitals (Free OpenStreetMap)")

    m = folium.Map(location=user_ll, zoom_start=13)

    # User marker
    folium.Marker(
        location=user_ll,
        popup="You",
        icon=folium.Icon(color="blue", icon="user")
    ).add_to(m)

    # Hospital markers
    for h in nearest_list:
        folium.Marker(
            location=(h["lat"], h["lon"]),
            popup=f"{h['Hospital']}<br>Distance: {h['dist']} km<br>Beds: {h['Beds']} | ICU: {h['ICU']}",
            icon=folium.Icon(color="red", icon="plus-sign")
        ).add_to(m)

    # Render map in Streamlit
    st_folium(m, width=1100, height=500)

    st.success("✅ Free map loaded successfully using OpenStreetMap.")
