# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import math
import requests
import re
from functools import lru_cache
from pathlib import Path

# ===============================
# APP CONFIG
# ===============================
st.set_page_config(page_title="Dengue Allocation", page_icon="🏥", layout="wide")
st.title("🏥 Integrated Hospital Dengue Patient Allocation System")

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
    "Dhanmondi","Gulshan","Uttara","Mirpur","Banani","Mohammadpur","Motijheel",
    "Rampura","Badda","Khilgaon","Shyamoli","Farmgate","Malibagh","Moghbazar"
]

# ===============================
# STABLE GEOCODER
# ===============================
@lru_cache(maxsize=256)
def geocode_place(place):
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": f"{place}, Dhaka, Bangladesh",
        "format": "json",
        "limit": 1
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        if r.status_code == 200 and r.json():
            return float(r.json()[0]["lat"]), float(r.json()[0]["lon"])
    except:
        pass
    return None

# ===============================
# DISTANCE FUNCTION
# ===============================
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    p = math.pi / 180
    a = 0.5 - math.cos((lat2-lat1)*p)/2 + math.cos(lat1*p)*math.cos(lat2*p)*(1-math.cos((lon2-lon1)*p))/2
    return 2 * R * math.asin(math.sqrt(a))

# ===============================
# FAKE AVAILABILITY (FOR MAP DEMO)
# ===============================
@st.cache_data
def fake_availability():
    rows = []
    for h in HOSPITALS_UI:
        rows.append({
            "Hospital": h,
            "Beds": np.random.randint(0, 30),
            "ICU": np.random.randint(0, 10)
        })
    return pd.DataFrame(rows)

availability = fake_availability()

# ===============================
# FORM INPUT
# ===============================
with st.form("allocation"):
    col1, col2 = st.columns(2)
    with col1:
        hospital_ui = st.selectbox("Hospital", HOSPITALS_UI)
        pick_area = st.selectbox("Pick Dhaka Area", ["—"] + DHAKA_AREAS)
    with col2:
        age = st.number_input("Age", 0, 120, 25)
        platelet = st.number_input("Platelet", 0, 300000, 120000)

    submit = st.form_submit_button("🚑 Allocate")

# ===============================
# ALLOCATION + MAP
# ===============================
if submit:
    st.subheader("✅ Allocation Result")

    # Decide severity
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

    # ===============================
    # GEO USER LOCATION
    # ===============================
    if pick_area != "—":
        user_ll = geocode_place(pick_area)
    else:
        user_ll = None

    if not user_ll:
        st.error("❌ Could not detect your location. Pick a valid Dhaka area.")
        st.stop()

    # ===============================
    # NEAREST HOSPITAL CALCULATION
    # ===============================
    nearest_list = []

    for _, row in availability.iterrows():
        h_ll = geocode_place(row["Hospital"])
        if h_ll:
            dist = haversine(user_ll[0], user_ll[1], h_ll[0], h_ll[1])
            nearest_list.append({
                "Hospital": row["Hospital"],
                "Beds": row["Beds"],
                "ICU": row["ICU"],
                "lat": h_ll[0],
                "lon": h_ll[1],
                "dist": round(dist, 1)
            })

    nearest_list = sorted(nearest_list, key=lambda x: x["dist"])[:3]

    if not nearest_list:
        st.error("❌ No hospitals could be geocoded.")
        st.stop()

    # ===============================
    # TABLE OUTPUT
    # ===============================
    st.dataframe(pd.DataFrame(nearest_list)[["Hospital","Beds","ICU","dist"]])

    # ===============================
    # ✅✅✅ GUARANTEED WORKING MAP
    # ===============================
    st.subheader("🗺️ Nearest Hospitals Map")

    map_rows = []

    # User pin
    map_rows.append({
        "lat": user_ll[0],
        "lon": user_ll[1],
        "label": "You"
    })

    # Hospital pins
    for h in nearest_list:
        map_rows.append({
            "lat": h["lat"],
            "lon": h["lon"],
            "label": h["Hospital"]
        })

    map_df = pd.DataFrame(map_rows)

    st.map(
        map_df,
        latitude="lat",
        longitude="lon",
        size=80,
        use_container_width=True
    )

    st.success("✅ Map loaded successfully.")
