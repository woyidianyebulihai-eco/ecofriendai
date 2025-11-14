# app_en.py
# ---------------------------------------------------------
# Requirements (requirements.txt):
# streamlit
# opencv-python
# pillow
# pyyaml
# numpy
# roboflow
#
# ---------------------------------------------------------
# Add your Roboflow credentials to .streamlit/secrets.toml:
#
# ROBOFLOW_API_KEY = "your_api_key"
# ROBOFLOW_WORKSPACE = "your_workspace_slug"
# ROBOFLOW_PROJECT = "your_project_slug"
# ROBOFLOW_VERSION = 1
# ---------------------------------------------------------

import streamlit as st
import cv2
import yaml
import numpy as np
from PIL import Image
import tempfile
from collections import defaultdict

# -------------------- PAGE SETUP --------------------
st.set_page_config(page_title="EcoHome Advisor", layout="wide")
st.title("EcoHome Advisor — Sustainable Renovation Recommender")

st.markdown("""
**How it works**

1. (Optional) Upload a photo of your home's exterior or roof.  
   - A house-segmentation model will estimate materials such as roof, siding, stone, fascia, trim, etc.
2. Fill out the inputs on the left (climate, humidity, orientation, window-to-wall ratio, budget).
3. Click **Analyze & Recommend** to see recommendations and an AI reasoning explanation.
""")

# -------------------- LOAD ROBOFLOW MODEL --------------------
@st.cache_resource
def load_rf_model():
    """Load Roboflow segmentation model. Return None if missing configuration."""
    try:
        from roboflow import Roboflow
        api_key = st.secrets["ROBOFLOW_API_KEY"]
        workspace = st.secrets["ROBOFLOW_WORKSPACE"]
        project_slug = st.secrets["ROBOFLOW_PROJECT"]
        version_num = int(st.secrets.get("ROBOFLOW_VERSION", 1))
    except Exception as e:
        st.sidebar.warning(
            "Roboflow model not configured. Segmentation analysis will be skipped.\n"
            f"Details: {e}"
        )
        return None

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project_slug)
    model = project.version(version_num).model
    return model

rf_model = load_rf_model()

HOUSE_MATERIAL_CLASSES = [
    "Horizontal Siding",
    "Vertical Siding",
    "Shakes",
    "Stone"
]

# -------------------- DEFAULT RULES (editable) --------------------
DEFAULT_RULES = """
climate_zones:
  hot_humid:
    roof:
      - when: "roof_color == 'dark' and roof_share > 0.15"
        recommend:
          - "High-reflectivity (cool) roof coating over existing roof"
          - "Light-colored or cool roof tiles / membrane"
        saving_pct: "8-15%"
        rationale: "Dark roofs in hot-humid climates significantly increase cooling load."
    walls:
      - when: "main_wall_material in ['Stone'] and humidity == 'high'"
        recommend:
          - "Use vapor-open exterior or interior insulation"
          - "Improve moisture management layers"
        saving_pct: "5-12%"
        rationale: "Stone walls can trap moisture; vapor-open layers prevent mold."
      - when: "main_wall_material in ['Horizontal Siding','Vertical Siding','Shakes'] and humidity == 'high'"
        recommend:
          - "Improve ventilated rain-screen behind siding"
          - "Seal joints while keeping drainage paths open"
        saving_pct: "3-8%"
        rationale: "Lightweight siding in humid climates must manage moisture effectively."
    garage:
      - when: "has_garage"
        recommend:
          - "Weatherstrip garage-to-house doorway"
          - "Air-seal penetrations between garage and living areas"
        saving_pct: "2-5%"
        rationale: "Attached garages often introduce heat, moisture, or pollutants indoors."

  temperate:
    roof:
      - when: "roof_color == 'dark'"
        recommend:
          - "High-reflectivity roof coating (SR > 0.7)"
          - "Light-colored shingles or membrane when reroofing"
        saving_pct: "5-10%"
        rationale: "Dark roofs may still cause seasonal overheating in temperate climates."
    windows:
      - when: "window_area_ratio > 0.25 and orientation in ['south','west']"
        recommend:
          - "Low-E window film or double glazing"
          - "External shading (louvers, overhangs, awnings)"
        saving_pct: "4-10%"
        rationale: "South/west glazing benefits from solar control."

  cold_dry:
    windows:
      - when: "window_area_ratio > 0.2 and orientation in ['north','east']"
        recommend:
          - "Triple glazing (triple-silver Low-E)"
          - "High-airtightness window frames"
        saving_pct: "5-12%"
        rationale: "Heat loss through windows dominates in cold, dry climates."
    walls:
      - when: "main_wall_material in ['Stone']"
        recommend:
          - "Add continuous exterior insulation (mineral wool or foam boards)"
          - "Improve air sealing at joints and trims"
        saving_pct: "10-20%"
        rationale: "Stone walls without continuous insulation lose heat rapidly."

budget:
  low:
    cap: 1500
    prefer:
      - "Weatherstrips"
      - "Interior blinds/curtains"
      - "Targeted air sealing"
  mid:
    cap: 6000
    prefer:
      - "Cool roof coating"
      - "Low-E window film"
      - "External shading kits"
  high:
    cap: 20000
    prefer:
      - "BIPV roof"
      - "Continuous exterior insulation"
      - "High-performance windows/doors"
"""

# -------------------- IMAGE ANALYSIS HELPERS --------------------
def simple_roof_lightness_score(img_bgr):
    """Estimate roof brightness from upper half of image using LAB L-channel."""
    h, w = img_bgr.shape[:2]
    roi = img_bgr[:h // 2, :]
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    return float(np.mean(lab[:, :, 0]))

def estimate_glass_reflection_ratio(img_bgr):
    """Rough estimate of bright reflective/glass-like pixels."""
    b, g, r = cv2.split(img_bgr)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, bright = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    diff = cv2.subtract(b, r)
    _, blueish = cv2.threshold(diff, 40, 255, cv2.THRESH_BINARY)
    mask = cv2.bitwise_and(bright, blueish)
    return float(np.sum(mask > 0)) / float(img_bgr.shape[0] * img_bgr.shape[1] + 1e-6)

def analyze_house_segments_with_roboflow(pil_image, model):
    """Use Roboflow segmentation model to extract facade/roof features."""
    if model is None:
        return {
            "main_wall_material": "unknown",
            "wall_material_shares": {m: 0.0 for m in HOUSE_MATERIAL_CLASSES},
            "roof_share": 0.0,
            "has_garage": False,
            "has_entry_door": False,
            "raw_predictions": {}
        }

    w, h = pil_image.size
    total_area = w * h

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        pil_image.save(tmp.name)
        pred = model.predict(tmp.name, confidence=40, overlap=30).json()

    class_areas = defaultdict(float)
    has_garage = False
    has_entry_door = False

    for p in pred.get("predictions", []):
        cls = p.get("class", "")
        area = float(p.get("width", 0) * p.get("height", 0))
        class_areas[cls] += area
        if cls == "Garage Door":
            has_garage = True
        if cls == "Entry Door":
            has_entry_door = True

    wall_shares = {m: class_areas[m] / (total_area + 1e-6) for m in HOUSE_MATERIAL_CLASSES}
    main_wall_material = max(wall_shares.items(), key=lambda kv: kv[1])[0] if wall_shares else "unknown"
    roof_share = class_areas["Roof"] / (total_area + 1e-6)

    return {
        "main_wall_material": main_wall_material,
        "wall_material_shares": {k: float(v) for k, v in wall_shares.items()},
        "roof_share": float(roof_share),
        "has_garage": has_garage,
        "has_entry_door": has_entry_door,
        "raw_predictions": pred
    }

# -------------------- RULE ENGINE --------------------
def eval_condition(cond, ctx):
    try:
        return eval(cond, {}, ctx)
    except Exception:
        return False

def apply_rules(rules, ctx):
    outputs = []
    cz = ctx.get("climate_zone", "temperate")
    zone = rules.get("climate_zones", {}).get(cz, {})

    for part, recs in zone.items():
        if not isinstance(recs, list):
            continue
        for rule in recs:
            if eval_condition(rule.get("when", "True"), ctx):
                outputs.append({
                    "part": part,
                    "recommend": rule.get("recommend", []),
                    "saving_pct": rule.get("saving_pct", "—"),
                    "rationale": rule.get("rationale", "")
                })

    budget_cfg = rules.get("budget", {}).get(ctx.get("budget", "mid"), {})
    return outputs, budget_cfg

def ai_reasoning(ctx, recs, budget_cfg):
    lines = []
    lines.append(
        f"- **Climate**: zone = `{ctx['climate_zone']}`, "
        f"humidity = `{ctx['humidity']}`, orientation = `{ctx['orientation']}`."
    )
    lines.append(
        f"- **Image cues**: roof lightness ≈ {ctx['roof_L']:.1f} (`{ctx['roof_color']}`), "
        f"roof share ≈ {ctx['roof_share']:.2f}, "
        f"dominant wall material = `{ctx['main_wall_material']}`, "
        f"garage = {ctx['has_garage']}, entry door = {ctx['has_entry_door']}."
    )
    lines.append(
        f"- **Window ratio** (user input): {ctx['window_area_ratio']:.2f}"
    )

    lines.append("\n- **Matched Recommendations**:")
    if recs:
        for r in recs:
            lines.append(
                f"  • **{r['part']}**: {', '.join(r['recommend'])} "
                f"(estimated savings {r['saving_pct']}).  \n"
                f"    Reasoning: {r['rationale']}"
            )
    else:
        lines.append("  • No rules matched.")

    if budget_cfg:
        lines.append(
            f"\n- **Budget guidance**: (cap ≈ {budget_cfg.get('cap')} USD) → "
            f"priority upgrades: {', '.join(budget_cfg.get('prefer', []))}."
        )

    lines.append(
        "\n_These recommendations combine material segmentation with simplified "
        "building physics principles for climate-appropriate improvements._"
    )

    return "\n".join(lines)

# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.header("Inputs")
    climate_zone = st.selectbox("Climate zone", ["hot_humid", "temperate", "cold_dry"])
    humidity = st.selectbox("Humidity level", ["low", "medium", "high"], index=1)
    orientation = st.selectbox("Facade orientation", ["north", "south", "east", "west"], index=1)
    window_area_ratio = st.slider("Window-to-wall ratio", 0.0, 0.6, 0.2, 0.01)
    budget = st.selectbox("Budget tier", ["low", "mid", "high"], index=1)
    rules_text = st.text_area("Rules (YAML)", value=DEFAULT_RULES, height=360)

# -------------------- MAIN UI --------------------
left, right = st.columns([1, 1])

img_bgr = None
seg_features = None

with left:
    st.subheader("① Upload Photo (optional)")
    img_file = st.file_uploader("Choose JPG/PNG", type=["jpg", "jpeg", "png"])

    if img_file:
        image = Image.open(img_file).convert("RGB")
        st.image(image, caption="Uploaded image")
        img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        with st.spinner("Running house segmentation..."):
            seg_features = analyze_house_segments_with_roboflow(image, rf_model)

        st.write("**Segmentation summary:**")
        st.json({
            "main_wall_material": seg_features["main_wall_material"],
            "roof_share": seg_features["roof_share"],
            "wall_material_shares": seg_features["wall_material_shares"],
            "has_garage": seg_features["has_garage"],
            "has_entry_door": seg_features["has_entry_door"]
        })

    run = st.button("Analyze & Recommend")

# -------------------- ANALYSIS OUTPUT --------------------
if run:
    try:
        rules = yaml.safe_load(rules_text)
    except Exception as e:
        st.error(f"YAML error: {e}")
        st.stop()

    roof_L = 180.0
    roof_color = "unknown"
    glass_ratio = 0.0

    if img_bgr is not None:
        roof_L = simple_roof_lightness_score(img_bgr)
        roof_color = "dark" if roof_L < 140 else "light"
        glass_ratio = estimate_glass_reflection_ratio(img_bgr)

    main_wall_material = seg_features["main_wall_material"] if seg_features else "unknown"
    roof_share = seg_features["roof_share"] if seg_features else 0.0
    has_garage = seg_features["has_garage"] if seg_features else False
    has_entry_door = seg_features["has_entry_door"] if seg_features else False

    ctx = dict(
        climate_zone=climate_zone,
        humidity=humidity,
        orientation=orientation,
        window_area_ratio=window_area_ratio,
        budget=budget,
        roof_L=roof_L,
        roof_color=roof_color,
        glass_ratio=glass_ratio,
        main_wall_material=main_wall_material,
        roof_share=roof_share,
        has_garage=has_garage,
        has_entry_door=has_entry_door,
    )

    recs, budget_cfg = apply_rules(rules, ctx)

    with right:
        st.subheader("② Recommendations")
        if not recs:
            st.info("No recommendations matched. Try adjusting inputs or editing your YAML rules.")
        else:
            for r in recs:
                with st.container():
                    st.markdown(f"**Part:** {r['part']}")
                    st.markdown(f"**Recommend:** {', '.join(r['recommend'])}")
                    st.markdown(f"**Estimated savings:** {r['saving_pct']}")
                    st.caption(f"Why: {r['rationale']}")

        st.subheader("③ AI Reasoning")
        st.markdown(ai_reasoning(ctx, recs, budget_cfg))

else:
    with right:
        st.info("Upload an image (optional), adjust inputs, then click **Analyze & Recommend**.")
