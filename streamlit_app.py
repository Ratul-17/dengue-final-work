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
st.title("🏥 Dengue Patient Allocation System (Free OSM Map)")

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
# ✅ OFFLINE DHAKA FALLBACK COORDS (NEVER FAILS)
# ===============================
DHAKA_FALLBACK = {
    "Dhanmondi": (23.7465, 90.3760),
    "Gulshan": (23.7806, 90.4170),
    "Uttara": (23.8759, 90.3795),
    "Mirpur": (23.8223, 90.3654),
    "Banani": (23.7936, 90.4066),
    "Mohammadpur": (23.7589, 90.3610),
    "Motijheel": (23.7333, 90.4172),
    "Rampura": (23.7583, 90.4286),
    "Badda": (23.7800, 90.4250),
    "Khilgaon": (23.7461, 90.4322),
    "Shyamoli": (23.7742, 90.3656),
    "Farmgate": (23.7576, 90.3890),
    "Malibagh": (23.7485, 90.4112),
    "Moghbazar": (23.7518, 90.4025),
}

# ===============================
# ✅ HARDENED FREE GEOCODER (CLOUD SAFE)
# ===============================
@lru_cache(maxsize=256)
def geocode(place: str):
    if not place or len(place.strip()) < 3:
        return None

    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": f"{place}, Dhaka, Bangladesh",
        "format": "json",
        "limit": 1
    }

    headers = {
        "User-Agent": "dengue-allocation-app/1.0 (contact: admin@demo.com)"
    }

    try:
        r = requests.get(url, params=params, headers=headers, timeout=20)
        if r.status_code != 200:
            return None
        data = r.json()
        if not data:
            return None
        return float(data[0]["lat"]), float(data[0]["lon"])
    except:
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
# ✅ FAKE AVAILABILITY DATA (REPLACE WITH YOUR EXCEL LOGIC LATER)
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
# PROCESS + MAP
# ===============================
if submit:
    st.subheader("✅ Allocation Result")

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

    # ✅ USER LOCATION (ONLINE + FALLBACK)
    user_ll = geocode(pick_area)
    if not user_ll and pick_area in DHAKA_FALLBACK:
        user_ll = DHAKA_FALLBACK[pick_area]

    if not user_ll:
        st.error("❌ Location detection failed.")
        st.stop()

    # ===============================
    # ✅ NEAREST HOSPITAL CALCULATION
    # ===============================
    nearest_list = []

    for _, row in availability.iterrows():
        h_ll = geocode(row["Hospital"])

        if not h_ll:
            continue

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
        st.error("❌ Could not geocode hospitals.")
        st.stop()

    nearest_list = sorted(nearest_list, key=lambda x: x["dist"])[:3]

    # ===============================
    # TABLE
    # ===============================
    st.dataframe(pd.DataFrame(nearest_list)[["Hospital", "Beds", "ICU", "dist"]])

    # ===============================
    # ✅✅✅ FREE OPENSTREETMAP MAP (FOLIUM)
    # ===============================
    st.subheader("🗺️ Nearest Hospitals (OpenStreetMap)")

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

    st_folium(m, width=1100, height=520)

    st.success("✅ Map loaded using FREE OpenStreetMap.")
