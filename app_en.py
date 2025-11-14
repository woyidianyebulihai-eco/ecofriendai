import streamlit as st
import requests, base64, tempfile
import numpy as np
from PIL import Image
import yaml

st.set_page_config(page_title="EcoHome Advisor", layout="wide")
st.title("EcoHome Advisor — Sustainable Recommendations")
st.write("keys:", list(st.secrets.keys()))


# ---------------- CONFIG LOADER ----------------
@st.cache_resource
def load_config():
    try:
        api_key = st.secrets["ROBOFLOW_API_KEY"]
        url = st.secrets["ROBOFLOW_URL"]  # FULL URL from deploy page
        return {"api_key": api_key, "url": url}
    except Exception as e:
        st.error(f"Roboflow API not configured: {e}")
        return None

rf = load_config()

# ---------------- IMAGE ANALYSIS ----------------
def simple_lightness(img_np):
    roi = img_np[: img_np.shape[0] // 2]
    return float((0.299*roi[:,:,0] + 0.587*roi[:,:,1] + 0.114*roi[:,:,2]).mean())

def analyze_segmentation(img: Image.Image):
    if rf is None:
        return None

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        img.save(tmp.name)
        with open(tmp.name, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()

    try:
        resp = requests.post(
            f"{rf['url']}?api_key={rf['api_key']}",
            data=encoded,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=20
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.warning(f"Segmentation failed: {e}")
        return None

# ---------------- ALWAYS-GIVE-RESULT RULE ENGINE ----------------
def generate_recommendations(ctx):
    recs = []

    # ALWAYS give roof advice
    if ctx["roof_brightness"] < 140:
        recs.append("Your roof appears dark; consider cool roof coating or lighter materials.")
    else:
        recs.append("Your roof appears moderately reflective; consider adding insulation under roof deck.")

    # Humidity logic
    if ctx["humidity"] == "high":
        recs.append("High humidity detected — choose vapor-open, breathable wall systems.")
    else:
        recs.append("Humidity is manageable — prioritize insulation and air sealing.")

    # Orientation
    if ctx["orientation"] in ["south", "west"]:
        recs.append("South/west facade → consider shading devices or Low-E films.")
    else:
        recs.append("North/east facade → ensure thermal envelope integrity.")

    # Window ratio
    if ctx["window_ratio"] > 0.3:
        recs.append("High glazing ratio → upgrade window performance to reduce heat loss/gain.")
    else:
        recs.append("Moderate glazing → focus on walls and roof for efficiency gains.")

    return recs

# ---------------- SIDEBAR ----------------
climate = st.sidebar.selectbox("Climate", ["hot", "temperate", "cold"])
humidity = st.sidebar.selectbox("Humidity", ["low", "medium", "high"])
orientation = st.sidebar.selectbox("Orientation", ["north","south","east","west"])
window_ratio = st.sidebar.slider("Window-to-wall ratio", 0.0, 0.6, 0.2)

# ---------------- MAIN ----------------
uploaded = st.file_uploader("Upload your house photo", type=["jpg", "png"])

img_np = None
seg = None

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded image", use_container_width=True)
    img_np = np.array(img)
    seg = analyze_segmentation(img)

run = st.button("Analyze")

if run:
    roof_brightness = simple_lightness(img_np) if img_np is not None else 180

    ctx = {
        "climate": climate,
        "humidity": humidity,
        "orientation": orientation,
        "window_ratio": window_ratio,
        "roof_brightness": roof_brightness
    }

    recs = generate_recommendations(ctx)

    st.subheader("Recommendations")
    for r in recs:
        st.markdown(f"- {r}")

    st.subheader("AI Reasoning")
    st.write(
        f"""
        - Roof brightness score: **{roof_brightness:.1f}**  
        - Climate: **{climate}**  
        - Humidity: **{humidity}**  
        - Orientation: **{orientation}**  
        - Window ratio: **{window_ratio:.2f}**
        """
    )
