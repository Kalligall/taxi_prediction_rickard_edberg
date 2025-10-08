import os
import requests
import streamlit as st
import uuid
import math

st.set_page_config(page_title="Taxi Price Predictor", layout= "centered")
st.title("Taxi Price Predictor")

def resolve_api_base():
    env = os.getenv("API_BASE")
    if env:
        return env
    try:
        return st.secrets.get("API_BASE")
    except Exception:
        return "http://127.0.0.1:8000"

API_BASE = resolve_api_base()

def get_secret_safe(key: str):
    v = os.getenv(key)
    if v:
        return v
    try:
        return st.secrets[key]
    except Exception:
        return None
    
GOOGLE_MAPS_API_KEY = get_secret_safe("GOOGLE_MAPS_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", None)


SESSION = requests.Session()
TIMEOUT = 15

@st.cache_data(ttl=30)
def get_schema():
    r = SESSION.get(f"{API_BASE}/schema", timeout=TIMEOUT)
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, dict):
        raise RuntimeError(f"Expected /schema response: {data}")
    return data

try:
    schema = get_schema()
    NUM = list(schema.get("numeric", []))
    CAT = list(schema.get("categoric", []))
    if not NUM:
        raise RuntimeError("Schema has no numeric features. Retrain the model to refresh")
except Exception as e:
    st.error(f"Failed to load schema from API: ({API_BASE}/schema), Make sure the API is running. \n\n{e}")
    st.stop()

def post_predict(payload: dict) -> float:
    r = SESSION.post(f"{API_BASE}/predict", json=payload, timeout=TIMEOUT)
    r.raise_for_status()
    return float(r.json().get("predicted_price"))

def gmaps_distance_duration(origin: str, destination: str, api_key: str):
    """Returns (distance_km, duration_min) between origin and destination using Google Maps API for routes and places."""
    url = "https://routes.googleapis.com/directions/v2:computeRoutes"
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": GOOGLE_MAPS_API_KEY,
        "X-Goog-FieldMask": "routes.distanceMeters,routes.duration"
    }
    
    body = {
        "origin": {"address": origin},
        "destination": {"address": destination},
        "travelMode": "DRIVE",
        "routingPreference": "TRAFFIC_AWARE",
        "computeAlternativeRoutes": False,
    }

    resp = SESSION.post(url, headers=headers, json=body, timeout=TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    if not data.get("routes"):
        raise RuntimeError(f"No route found: {data}")
    route = data["routes"][0]
    meters = float(route["distanceMeters"])
    seconds = float(str(route["duration"]).rstrip("s"))
    return meters / 1000.0, seconds / 60.0
   


def _apply_pending_text():
    if "pending_origin_text" in st.session_state:
        st.session_state["origin_text"] = st.session_state.pop("pending_origin_text")
    if "pending_destination_text" in st.session_state:
        st.session_state["destination_text"] = st.session_state.pop("pending_destination_text")

def get_places_token() -> str:
    """One sessions token per user session."""
    if "places_token" not in st.session_state:
        st.session_state["places_token"] = str(uuid.uuid4())
    return st.session_state["places_token"]

def places_autocomplete(query: str, api_key: str, *,
                        language="en", region=None, max_suggestions=5):
    if not query or len(query.strip()) < 3:
        return []
    
    url = "https://places.googleapis.com/v1/places:autocomplete"
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": api_key,
        "X-Goog-FieldMask": "suggestions.placePrediction.placeId,"
                            "suggestions.placePrediction.text",
    }
    body = {
        "input": query,
        "languageCode": language,
        "sessionToken": get_places_token(),
        "includedPrimaryTypes": ["street_address", "premise", "route", "locality", "airport"]
    }
    if region:
        body["regionCode"] = region

    resp = requests.post(url, headers=headers, json=body)
    resp.raise_for_status()
    data = resp.json()
    out = []
    for s in data.get("suggestions", [])[:max_suggestions]:
        pp = s.get("placePrediction", {})
        text = (pp.get("text") or {}).get("text")
        pid = pp.get("placeId")
        if text:
            out.append({"text": text, "place_id": pid})
    return out


## add when gemini key is present     
#def convert_currency(amount: float, from_currency: str, to_currency: str, api_key: str) -> float:
#    """Converts amount from one currency to another using Gemini Currency API."""
#    url = "https://api.gemini.com/v1/price"
#    params = {
#        "from": from_currency,
#        "to": to_currency,
#        "api_key": api_key
#    }
#    try:
#        resp = SESSION.get(url, params=params, timeout=TIMEOUT)
#       resp.raise_for_status()
#        rate = float(resp.json().get("rate"))
#        return amount * rate
#   except Exception as e:
#        st.error(f"Failed to convert currency using Gemini API: {e}")
#        return amount

## remove when gemini key is present
def convert_currency(amount: float, rate: float | None) -> float:
    """Manual conversion using provided rate (if any)."""
    try:
        if rate is None or not math.isfinite(rate) or rate <= 0:
            return float(amount)
        return float(amount) * float(rate)
    except Exception:
        return float(amount)


def build_payload_from_schema(NUM_cols, candidate: dict) -> dict:
    payload = {}
    for c in NUM_cols:
        v = candidate.get(c, 0.0)
        if v is None:
            v = 0.0
        try:
            payload[c] = 0.0 if v is None else float(v)
        except Exception:
            payload[c] = 0.0
    return payload

st.subheader("Enter trip addresses")

_apply_pending_text()

c1, c2 = st.columns(2)

with c1:
    origin = st.text_input("Origin", key="origin_text", placeholder="e.g., Vasagatan 1, Stockholm")
    suggestions_o = places_autocomplete(st.session_state.get("origin_text", ""), GOOGLE_MAPS_API_KEY) if GOOGLE_MAPS_API_KEY else []
    if suggestions_o:
        st.caption("Suggestions (origin)")
        cols = st.columns(min(2, len(suggestions_o)))
        for i, s in enumerate(suggestions_o):
            col = cols[i % len(cols)]
            if col.button(s["text"], key=f"origin_sugg_{i}", use_container_width=True):
                st.session_state["pending_origin_text"] = s["text"]
                st.rerun()

with c2:
    destination = st.text_input("Destination", key="destination_text", placeholder="e.g., Arlanda Airport")
    suggestions_d = places_autocomplete(st.session_state.get("destination_text", ""), GOOGLE_MAPS_API_KEY) if GOOGLE_MAPS_API_KEY else []
    if suggestions_d:
        st.caption("Suggestions (destination)")
        cols = st.columns(min(2, len(suggestions_d)))
        for i, s in enumerate(suggestions_d):
            col = cols[i % len(cols)]
            if col.button(s["text"], key=f"dest_sugg_{i}", use_container_width=True):
                st.session_state["pending_destination_text"] = s["text"]
                st.rerun()


    base = ["Base_Fare"]
    per_km = ["Per_Km_Rate"]
    per_min = ["Per_Minute_Rate"]

go = st.button("Get price")

if go:
    origin= st.session_state.get("origin_text", "")
    destination = st.session_state.get("destination_text", "")

    if not origin or not destination:
        st.error("please enter both origin and destination")
        st.stop()
    if not GOOGLE_MAPS_API_KEY:
        st.error("Google Maps API key missing")
        st.stop()
    else:
        st.caption("Maps key loaded")

    with st.spinner("Looking up distance & duration..."):
        try:
            dist_km, dur_min = gmaps_distance_duration(origin, destination, GOOGLE_MAPS_API_KEY)
        except Exception as e:
            st.error(f"Maps lookup failed: {e}")
            st.stop()

    km_per_min = (dist_km / dur_min) if dur_min > 0 else 0.0
    candidate = {
        "Trip_Distance_km": dist_km,
        "Trip_Duration_Minutes": dur_min,
        "Base_Fare": base,
        "Per_Km_Rate": per_km,
        "Per_Minute_Rate": per_min,
        "Km_per_Min": km_per_min,
    }
    payload = build_payload_from_schema(NUM, candidate)

    with st.spinner("Predicting price..."):
        try:
            price = post_predict(payload)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            with st.expander("debug payload"):
                st.write("API_BASE:", API_BASE)
                st.json({"payload": payload, "schema_numeric": NUM})
            st.stop()

    st.success(f"Estimated price: **{price:.2f}**")
    st.caption(f"Distance: {dist_km:.2f} km * Duration: {dur_min:.1f} min")



with st.expander("Health / schema"):
    try:
        health = SESSION.get(f"{API_BASE}/health", timeout=TIMEOUT).json()
        st.write("Health:", health)
    except Exception as e:
        st.error(f"Health not available: {e}")
    st.write("Schema:", schema)


if GEMINI_API_KEY:
    st.markdown("---")
    st.caption("Optional gemini explanation is available (API key detected).")
    if st.button ("Explain this prediction (Gemini)"):
        st.info("Hook ready - implement when we add gemini")
else:
    pass