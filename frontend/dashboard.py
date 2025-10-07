import os
import requests
import streamlit as st

API_BASE = os.getenv("API_BASE") or st.secrets.get("API_BASE", "http://127.0.0.1:8000")

st.set_page_config(page_title="Taxi Price Prediction", layout= "centered")
st.title("Taxi Price Prediction")

SESSION = requests.Session()
TIMEOUT = 10

@st.cache_data(ttl=30)
def get_schema():
    r = SESSION.get(f"{API_BASE}/schema", timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()

try:
    schema = get_schema()
    NUM = schema.get("numeric", [])
    CAT = schema.get("categoric", [])
except Exception as e:
    st.error(f"Failed to load schema from API: ({API_BASE}/schema), Make sure the API is running. \n\n{e}")
    st.stop()

with st.form("Predict_form"):
    st.subheader("Trip_inputs")
    c1, c2 = st.columns(2)
    with c1:
        dist = st.number_input("Trip_Distance_km", min_value=0.0, value=8.0, step=0.1)
        dur = st.number_input("Trip_Duration_Minutes", min_value=0.0, value=15.0, step=0.5)
        pax = st.number_input("Passenger_Count", min_value=0, value=1, step=1)
        base = st.number_input("Base_Fare", min_value=0.0, value=40.0, step=1.0)
    with c2:
        per_km = st.number_input("Per_Km_Rate", min_value=0.0, value=12.0, step=0.1)
        per_min = st.number_input("Per_Minute_Rate", min_value=0.0, value=3.0, step=0.1)
        tod = st.selectbox("Time_of_Day", ["Morning", "Afternoon", "Evening", "Night"], index=2)
        dow = st.selectbox("Day_of_Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], index=4)
        traffic = st.selectbox("Traffic_Conditions", ["Light", "Moderate", "Heavy"], index=1)
        weather = st.selectbox("Weather", ["Clear", "Rain", "Snow", "Fog"], index=0)
    
    submitted = st.form_submit_button("Predict")

rule_estimate = base + per_km * dist + per_min * dur
km_per_min = (dist / dur) if dur > 0 else 0.0
diff_to_rule = 0.0

candidate = {
    "Trip_Distance_km": dist,
    "Trip_Duration_Minutes": dur,
    "Passenger_Count": pax,
    "Base_Fare": base,
    "Per_Km_Rate": per_km,
    "Per_Minute_Rate": per_min,
    "Time_of_Day": tod,
    "Day_of_Week": dow,
    "Traffic_Conditions": traffic,
    "Weather": weather,
    "Rule_Estimate": rule_estimate,
    "Diff_to_Rule": diff_to_rule,
    "Km_per_Min": km_per_min
}

def build_payload():
    payload = {}
    for c in NUM:
        payload[c] = float(candidate.get(c, 0.0))
    for c in CAT:
        v = candidate.get(c, "")
        payload[c] = "" if v is None else str(v)
    return payload

if submitted:
    payload = build_payload()
    with st.spinner("Calling API..."):
        try:
            r = SESSION.post(f"{API_BASE}/predict", json=payload, timeout=TIMEOUT)
            r.raise_for_status()
            pred = r.json().get("predicted_price")
            st.success(f"Predicted Price: {pred:.2f} **")
        except Exception as e:
            st.error(f"Request failed: {e}")
            with st.expander("Debug payload"):
                st.json(payload)

with st.expander("Health / schema"):
    try:
        health = SESSION.get(f"{API_BASE}/health", timeout=TIMEOUT).json()
        st.write("Health:", health)
    except Exception as e:
        st.error(f"Health not available: {e}")
    st.write("Schema:", schema)
