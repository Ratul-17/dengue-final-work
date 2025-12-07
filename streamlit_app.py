# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import math

# ===============================
# APP CONFIG
# ===============================
st.set_page_config(page_title="Dengue Allocation", page_icon="🏥", layout="wide")
st.title("🏥 Integrated Hospital Dengue Patient Allocation System")

# ===============================
# FIXED HOSPITAL LIST
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

# ===============================
# ✅ FIXED DHAKA AREA COORDINATES (NO INTERNET REQUIRED)
# ===============================
DHAKA_COORDS = {
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
}

# ===============================
# DISTANCE FUNCTION
# ===============================
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    p = math.pi / 180
    a = 0.5 - math.cos((lat2-lat1)*p)/2 + math.cos(lat1*p)*math.cos(lat2*p)*(1-math.cos((lon2-lon1)*p))/2
    return 2 * R * math.asin(math.sqrt(a))

# ===============================
# ✅ FAKE HOSPITAL COORDINATES (STABLE)
# ===============================
HOSPITAL_COORDS = {
    "Dhaka Medical College Hospital": (23.7258, 90.3977),
    "SSMC & Mitford Hospital": (23.7077, 90.4073),
    "Bangladesh Shishu Hospital & Institute": (23.7419, 90.3785),
    "Shaheed Suhrawardy Medical College Hospital": (23.7712, 90.3712),
    "Bangabandhu Sheikh Mujib Medical University": (23.7396, 90.3950),
    "Police Hospital, Rajarbagh": (23.7412, 90.4203),
    "Mugda Medical College Hospital": (23.7263, 90.4395),
    "Bangladesh Medical College Hospital": (23.7463, 90.3779),
    "Holy Family Red Crescent Hospital": (23.7456, 90.3893),
    "BIRDEM Hospital": (23.7391, 90.3947),
    "Ibn Sina Hospital": (23.7515, 90.3816),
    "Square Hospital": (23.7535, 90.3842),
    "Samorita Hospital": (23.7510, 90.3754),
    "Central Hospital Dhanmondi": (23.7525, 90.3790),
    "Lab Aid Hospital": (23.7521, 90.3849),
    "Green Life Medical Hospital": (23.7529, 90.3841),
    "Sirajul Islam Medical College Hospital": (23.7911, 90.3667),
    "Ad-Din Medical College Hospital": (23.7239, 90.3923),
}

# ===============================
# ✅ FAKE AVAILABILITY DATA
# ===============================
@st.cache_data
def fake_availability():
    rows = []
    for h in HOSPITALS_UI:
        rows.append({
            "Hospital": h,
            "Beds": np.random.randint(0, 40),
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
        pick_area = st.selectbox("Pick Dhaka Area", list(DHAKA_COORDS.keys()))
    with col2:
        age = st.number_input("Age", 0, 120, 25)
        platelet = st.number_input("Platelet", 0, 300000, 120000)

    submit = st.form_submit_button("🚑 Allocate")

# ===============================
# ALLOCATION + MAP
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

    # ✅ GUARANTEED USER LOCATION
    user_ll = DHAKA_COORDS[pick_area]

    # ===============================
    # ✅ NEAREST HOSPITAL CALCULATION
    # ===============================
    nearest_list = []

    for _, row in availability.iterrows():
        h_ll = HOSPITAL_COORDS[row["Hospital"]]
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

    # ===============================
    # TABLE
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
        size=90,
        use_container_width=True
    )

    st.success("✅ Map is now fully working with NO geocoding & NO API keys.")
