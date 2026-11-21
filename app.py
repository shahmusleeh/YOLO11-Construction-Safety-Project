from pathlib import Path
import streamlit as st
import pandas as pd
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="Smart Construction Safety Monitor", page_icon="🦺", layout="wide")

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "best.pt"

SAFETY_VIOLATIONS = {"NO-Hardhat", "NO-Mask", "NO-Safety Vest"}

@st.cache_resource
def load_model(path: str):
    return YOLO(path)

st.title("🦺 Smart Construction Safety Monitor")
st.write("Upload a construction-site image to detect workers, PPE, machinery, vehicles, and potential PPE violations.")

with st.sidebar:
    st.header("Detection settings")
    confidence = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)
    st.caption("Higher values show only more confident detections.")

if not MODEL_PATH.exists():
    st.error("Trained model file not found. Place your trained best.pt beside app.py.")
    st.code("your-repository/\\n├── app.py\\n├── best.pt\\n├── requirements.txt\\n└── README.md")
    st.stop()

try:
    model = load_model(str(MODEL_PATH))
except Exception as exc:
    st.exception(exc)
    st.stop()

uploaded_file = st.file_uploader("Choose a construction-site image", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
    except Exception as exc:
        st.error(f"Could not open image: {exc}")
        st.stop()

    left, right = st.columns(2)
    with left:
        st.subheader("Uploaded image")
        st.image(image, use_container_width=True)

    with st.spinner("Running YOLO detection..."):
        try:
            result = model.predict(source=image, conf=confidence, verbose=False)[0]
            annotated_rgb = result.plot()[:, :, ::-1]
        except Exception as exc:
            st.error(f"Detection failed: {exc}")
            st.stop()

    with right:
        st.subheader("Detection result")
        st.image(annotated_rgb, channels="RGB", use_container_width=True)

    names = result.names
    rows = []
    violation_count = 0

    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls.item())
            score = float(box.conf.item())
            class_name = names[class_id] if isinstance(names, dict) else names[class_id]
            is_violation = class_name in SAFETY_VIOLATIONS
            violation_count += int(is_violation)
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            rows.append({
                "Detected object": class_name,
                "Confidence": round(score, 3),
                "Potential PPE violation": "Yes" if is_violation else "No",
                "X1": round(x1, 1), "Y1": round(y1, 1),
                "X2": round(x2, 1), "Y2": round(y2, 1),
            })

    st.divider()
    st.subheader("Detection summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Objects detected", len(rows))
    c2.metric("Potential PPE violations", violation_count)
    c3.metric("Confidence threshold", f"{confidence:.0%}")

    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button(
            "Download detection report (CSV)",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="construction_safety_detection_report.csv",
            mime="text/csv",
        )
        if violation_count:
            st.warning(f"{violation_count} detection(s) were labelled NO-Hardhat, NO-Mask, or NO-Safety Vest. Please verify the image manually.")
        else:
            st.success("No configured PPE-violation classes were detected. This does not guarantee the site is safe.")
    else:
        st.info("No objects were detected above the selected confidence threshold.")

    st.caption("AI detections may contain false positives and false negatives. A qualified person must verify safety decisions.")
else:
    st.info("Upload a JPG, JPEG, PNG, or WEBP image to start detection.")
    st.markdown("**Classes:** Hardhat, Mask, NO-Hardhat, NO-Mask, NO-Safety Vest, Person, Safety Cone, Safety Vest, machinery, vehicle.")
