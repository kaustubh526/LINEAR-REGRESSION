"""
Food Delivery Time Predictor — Streamlit App
Coefficients baked in from the trained LinearRegression model (sklearn 1.6.1)

Install:
    pip install streamlit

Run:
    streamlit run food_delivery_streamlit.py

Deploy free:
    https://streamlit.io/cloud  →  push to GitHub → connect repo → deploy
"""

import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Food Delivery Time Predictor",
    page_icon="🛵",
    layout="centered",
)

# ── Model coefficients ────────────────────────────────────────────────────────

INTERCEPT = 18.910872782988115

COEF = {
    "Distance_km":              2.9890812044731208,
    "Preparation_Time_min":     0.9712811930609221,
    "Courier_Experience_yrs":  -0.6643082668488862,
    "Weather_Foggy":            5.751364534522872,
    "Weather_Rainy":            4.690908141715106,
    "Weather_Snowy":            9.233624572803944,
    "Weather_Windy":            1.5870840617971822,
    "Traffic_Level_Low":       -11.04792926986285,
    "Traffic_Level_Medium":    -5.412539677193411,
    "Time_of_Day_Evening":      0.8279377672071604,
    "Time_of_Day_Morning":     -0.2524647258812492,
    "Time_of_Day_Night":       -1.4304313773404096,
    "Vehicle_Type_Car":         0.4990047091472109,
    "Vehicle_Type_Scooter":    -0.9620057699956909,
}

WEATHER_EMOJI   = {"Clear": "☀️", "Foggy": "🌫️", "Rainy": "🌧️", "Snowy": "❄️", "Windy": "💨"}
VEHICLE_EMOJI   = {"Scooter": "🛵", "Bike": "🚲", "Car": "🚗"}

# ── Prediction logic ──────────────────────────────────────────────────────────

def predict(distance, prep_time, experience, weather, traffic, time_of_day, vehicle):
    features = {
        "Distance_km":             distance,
        "Preparation_Time_min":    prep_time,
        "Courier_Experience_yrs":  experience,
        "Weather_Foggy":           int(weather == "Foggy"),
        "Weather_Rainy":           int(weather == "Rainy"),
        "Weather_Snowy":           int(weather == "Snowy"),
        "Weather_Windy":           int(weather == "Windy"),
        "Traffic_Level_Low":       int(traffic == "Low"),
        "Traffic_Level_Medium":    int(traffic == "Medium"),
        "Time_of_Day_Evening":     int(time_of_day == "Evening"),
        "Time_of_Day_Morning":     int(time_of_day == "Morning"),
        "Time_of_Day_Night":       int(time_of_day == "Night"),
        "Vehicle_Type_Car":        int(vehicle == "Car"),
        "Vehicle_Type_Scooter":    int(vehicle == "Scooter"),
    }
    raw = INTERCEPT + sum(COEF[k] * v for k, v in features.items())
    return max(1.0, round(raw, 1)), features

# ── UI ────────────────────────────────────────────────────────────────────────

st.title("🛵 Delivery Time Predictor")
st.caption("Linear regression model · trained on 1,000 food delivery orders")
st.divider()

# Inputs
col1, col2 = st.columns(2)
with col1:
    distance    = st.slider("📍 Distance (km)",         0.5, 20.0, 10.0, step=0.1)
    prep_time   = st.slider("🍳 Preparation time (min)", 1,   40,   15)
with col2:
    experience  = st.slider("👤 Courier experience (yrs)", 0.0, 9.0, 4.5, step=0.5)
    weather     = st.selectbox("🌤️ Weather",
                    ["Clear", "Foggy", "Rainy", "Snowy", "Windy"],
                    format_func=lambda w: f"{WEATHER_EMOJI[w]} {w}")

col3, col4 = st.columns(2)
with col3:
    traffic     = st.radio("🚦 Traffic level", ["Low", "Medium", "High"], horizontal=True)
with col4:
    time_of_day = st.radio("🕐 Time of day",   ["Morning", "Afternoon", "Evening", "Night"], horizontal=True)

vehicle = st.radio("🚗 Vehicle type", ["Scooter", "Bike", "Car"],
                   format_func=lambda v: f"{VEHICLE_EMOJI[v]} {v}", horizontal=True)

st.divider()

# ── Result ────────────────────────────────────────────────────────────────────

result, features = predict(distance, prep_time, experience, weather, traffic, time_of_day, vehicle)

tier_map = [
    (35,  "🟢 Very fast",        "#d4edda"),
    (55,  "🟡 Standard",         "#fff3cd"),
    (75,  "🟠 Slower than avg",  "#ffe5d0"),
    (999, "🔴 Long delivery",    "#f8d7da"),
]
tier_label, tier_bg = next((t[1], t[2]) for t in tier_map if result < t[0])

st.markdown(
    f"""
    <div style="background:{tier_bg};border-radius:12px;padding:1.5rem 2rem;text-align:center;">
        <div style="font-size:14px;color:#555;margin-bottom:4px;">Estimated delivery time</div>
        <div style="font-size:56px;font-weight:800;color:#333;line-height:1;">{result:.0f} min</div>
        <div style="font-size:14px;color:#666;margin-top:6px;">{tier_label}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Factor breakdown ──────────────────────────────────────────────────────────

with st.expander("📊 Factor breakdown", expanded=True):
    contribs = {k: round(COEF[k] * v, 2) for k, v in features.items() if COEF[k] * v != 0}
    sorted_contribs = sorted(contribs.items(), key=lambda x: abs(x[1]), reverse=True)

    labels, vals, colors = [], [], []
    for feat, val in sorted_contribs:
        labels.append(feat.replace("_", " "))
        vals.append(val)
        colors.append("#D85A30" if val > 0 else "#1D9E75")

    import pandas as pd
    df = pd.DataFrame({"Feature": labels, "Contribution (min)": vals})
    df = df.sort_values("Contribution (min)")

    st.bar_chart(df.set_index("Feature"), horizontal=True, color="#D85A30")

# ── Tips ──────────────────────────────────────────────────────────────────────

tips = []
if weather == "Snowy":
    tips.append("❄️ Snow adds ~9 min — alert the customer early.")
if traffic == "High":
    tips.append("🚦 High traffic adds ~5–11 min vs low traffic.")
if experience < 2:
    tips.append("👤 Veteran couriers (9 yrs) are ~6 min faster on average.")
if distance > 15:
    tips.append("📍 Long distance is the biggest driver — consider zone-based routing.")

if tips:
    st.info("  \n".join(tips))

# ── Model info ────────────────────────────────────────────────────────────────

with st.expander("🔍 Model details"):
    st.write("**Model type:** LinearRegression (scikit-learn 1.6.1)")
    st.write(f"**Intercept:** `{INTERCEPT}`")
    st.write("**Coefficients:**")
    coef_df = pd.DataFrame(
        list(COEF.items()), columns=["Feature", "Coefficient"]
    ).sort_values("Coefficient", ascending=False)
    st.dataframe(coef_df, use_container_width=True, hide_index=True)