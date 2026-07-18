import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile

st.set_page_config(page_title="EcoSort AI", page_icon="♻️")

st.title("♻️ EcoSort AI")
st.write("Upload a waste image for classification.")

model = YOLO("best.pt")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", use_container_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp.name)

        results = model.predict(tmp.name)

    names = results[0].names
    pred = results[0].probs.top1
    conf = float(results[0].probs.top1conf) * 100

    st.success(f"Prediction: {names[pred]}")
    st.info(f"Confidence: {conf:.2f}%")

    from ultralytics import YOLO

    model = YOLO("best.pt")