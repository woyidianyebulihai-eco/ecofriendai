import streamlit as st
import cv2, yaml, numpy as np
from PIL import Image

# -------------------- PAGE SETUP --------------------
st.set_page_config(page_title="EcoHome Advisor (MVP) - EN", page_icon="🌿", layout="wide")
st.title("🌿 EcoHome Advisor — Sustainable Renovation Recommender (MVP)")

st.markdown("""
**How it works**
1) (Optional) Upload a photo of your home's exterior or roof.
2) Fill a few inputs on the left (climate, humidity, orientation, window-to-wall ratio, budget).
3) Click **Analyze & Recommend** to see the recommendations and an **AI Reasoning** explanation.

> You don't need building expertise. Default values work for a quick demo.
""")

# -------------------- DEFAULT RULES (editable) --------------------
DEFAULT_RULES = """
climate_zones:
  hot_humid:
    roof:
      - when: "roof_color == 'dark'"
        recommend: ["High-reflectivity roof coating (SR > 0.8)", "Cool roof membrane / ceramic heat-reflective spray"]
        saving_pct: "8-15%"
        rationale: "High solar radiation plus dark roof → absorbs more heat; cool the roof first."
    walls:
      - when: "humidity == 'high'"
        recommend: ["Mineral wool / wood-fiber insulation (vapor-open)", "Mold-resistant liner / breathable paint"]
        saving_pct: "3-8%"
        rationale: "Hot-humid → prefer vapor-open systems to reduce moisture/mold risk."
  temperate:
    windows:
      - when: "window_area_ratio > 0.25 and orientation in ['south','west']"
        recommend: ["Low-E window film or double glazing", "External shading (louvers / overhangs / awnings)"]
        saving_pct: "4-10%"
        rationale: "Moderate climate with strong sun → reduce solar heat gains first."
  cold_dry:
    windows:
      - when: "window_area_ratio > 0.2 and orientation in ['north','east']"
        recommend: ["Triple glazing (triple-silver Low-E)", "High-airtightness frames", "Improve perimeter sealing"]
        saving_pct: "5-12%"
        rationale: "Cold and dry → main losses through windows; prioritize insulation & airtightness."

budget:
  low: { cap: 1500, prefer: ["Door/window weatherstrips", "Interior shades/curtains", "Localized recycled insulation"] }
  mid:  { cap: 6000, prefer: ["Cool roof coating", "Low-E window film", "External shading kits"] }
  high: { cap: 20000, prefer: ["BIPV roof (building-integrated PV)", "Continuous exterior insulation", "High-performance windows/doors"] }
"""

# -------------------- SIMPLE IMAGE ANALYSIS --------------------
def simple_roof_lightness_score(img_bgr):
    """
    Simplified heuristic: take upper half of image as 'roof candidate area' and compute average lightness.
    Lower lightness = darker; higher = lighter.
    """
    h, w = img_bgr.shape[:2]
    roi = img_bgr[:h//2, :]
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    L = lab[:,:,0]  # 0-255
    return float(np.mean(L))

def estimate_glass_reflection_ratio(img_bgr):
    """
    Simplified heuristic: pixels that are both very bright and more blue than red ≈ potential reflective glass.
    Extremely rough for demo purposes only.
    """
    b,g,r = cv2.split(img_bgr)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, bright = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    diff = cv2.subtract(b, r)  # emphasize bluish reflections
    _, blueish = cv2.threshold(diff, 40, 255, cv2.THRESH_BINARY)
    mask = cv2.bitwise_and(bright, blueish)
    ratio = float(np.sum(mask>0)) / float(img_bgr.shape[0]*img_bgr.shape[1] + 1e-6)
    return ratio

def eval_condition(cond:str, ctx:dict):
    try:
        return eval(cond, {}, ctx)
    except Exception:
        return False

def apply_rules(rules:dict, ctx:dict):
    outputs = []
    cz = ctx.get("climate_zone", "temperate")
    zone = rules.get("climate_zones", {}).get(cz, {})
    for part, recs in zone.items():
        if not isinstance(recs, list):
            continue
        for rule in recs:
            cond = rule.get("when", "True")
            if eval_condition(cond, ctx):
                outputs.append({
                    "part": part,
                    "recommend": rule.get("recommend", []),
                    "saving_pct": rule.get("saving_pct", "—"),
                    "rationale": rule.get("rationale", "")
                })
    budget_cfg = rules.get("budget", {}).get(ctx.get("budget","mid"), {})
    return outputs, budget_cfg

def ai_reasoning(ctx, recs, budget_cfg):
    lines = []
    lines.append(f"- Climate zone: {ctx.get('climate_zone')}; Humidity: {ctx.get('humidity')}; Main orientation: {ctx.get('orientation')}")
    lines.append(f"- Image cues: roof lightness score {ctx.get('roof_L'):.1f} ({ctx.get('roof_color')}), suspected glass/strong-reflection {ctx.get('glass_ratio')*100:.1f}%")
    lines.append("- Rule matches:")
    if recs:
        for r in recs:
            lines.append(f"  • **{r['part']}**: {', '.join(r['recommend'])} (estimated saving {r['saving_pct']}). Why: {r['rationale']}")
    else:
        lines.append("  • No strong recommendations for the current inputs. Try uploading a clearer photo or adjusting inputs.")
    if budget_cfg:
        lines.append(f"- Budget hint: within your tier (cap ≈ {budget_cfg.get('cap','—')} USD), prioritize: {', '.join(budget_cfg.get('prefer', []))}.")
    return "\n".join(lines)

# -------------------- SIDEBAR INPUTS --------------------
with st.sidebar:
    st.header("Inputs (use defaults if unsure)")
    climate_zone = st.selectbox("Climate zone", ["hot_humid", "temperate", "cold_dry"], index=1)
    humidity = st.selectbox("Humidity level", ["low", "medium", "high"], index=1)
    orientation = st.selectbox("Main facade orientation", ["north","south","east","west"], index=1)
    window_area_ratio = st.slider("Window-to-wall ratio (0 = very few windows; 0.6 = many windows)", 0.0, 0.6, 0.2, 0.01)
    budget = st.selectbox("Budget tier", ["low","mid","high"], index=1)
    rules_text = st.text_area("Rules (YAML, editable)", value=DEFAULT_RULES, height=260)

# -------------------- MAIN: UPLOAD + ANALYZE --------------------
left, right = st.columns([1,1])

with left:
    st.subheader("① Upload exterior/roof photo (optional)")
    img_file = st.file_uploader("Choose JPG/PNG (you can skip this)", type=["jpg","jpeg","png"])
    img_bgr = None
    if img_file is not None:
        image = Image.open(img_file).convert("RGB")
        st.image(image, caption="Uploaded image", use_container_width=True)
        img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    run = st.button("Analyze & Recommend", type="primary")

if run:
    # 1) parse rules
    rules = yaml.safe_load(rules_text)

    # 2) quick image analysis
    roof_L = 180.0
    roof_color = "unknown"
    glass_ratio = 0.0
    if img_bgr is not None:
        roof_L = simple_roof_lightness_score(img_bgr)
        roof_color = "dark" if roof_L < 140 else "light"
        glass_ratio = estimate_glass_reflection_ratio(img_bgr)

    # 3) build context
    ctx = dict(
        climate_zone = climate_zone,
        humidity = humidity,
        orientation = orientation,
        window_area_ratio = window_area_ratio,
        budget = budget,
        roof_L = roof_L,
        roof_color = roof_color,
        glass_ratio = glass_ratio
    )

    # 4) apply rules
    recs, budget_cfg = apply_rules(rules, ctx)

    with right:
        st.subheader("② Recommendation List")
        if not recs:
            st.info("No strong recommendations yet. Try uploading a photo or adjust humidity/orientation/window ratio—or edit the rules on the left.")
        else:
            for r in recs:
                with st.container(border=True):
                    st.markdown(f"**Part**: {r['part']}")
                    st.markdown(f"**Recommend**: {', '.join(r['recommend'])}")
                    st.markdown(f"**Estimated energy saving**: {r['saving_pct']}")
                    st.caption(f"Why: {r['rationale']}")

        st.subheader("③ AI Reasoning")
        st.markdown(ai_reasoning(ctx, recs, budget_cfg))

else:
    with right:
        st.info("Select inputs on the left, optionally upload a photo, then click **Analyze & Recommend**.")
