
import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# --- Page Configuration ---
st.set_page_config(page_title="Smart Traffic Control", page_icon="🚦")

st.title("🚦 Smart Traffic Management System")
st.write("Prioritizing Emergency Vehicles using Computer Vision")

# --- Load Model ---
# We use @st.cache_resource so the model loads only once, making the app faster
@st.cache_resource
def load_model():
    model = MobileNetV2(weights='imagenet')
    return model

with st.spinner('Loading AI Model...'):
    model = load_model()

# --- Prediction Logic ---
def predict_image(img):
    # Resize image to 224x224 (required by MobileNetV2)
    img = img.resize((224, 224))

    # Convert image to numpy array
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
    img_array = preprocess_input(img_array)

    # Get predictions
    predictions = model.predict(img_array)
    decoded_preds = decode_predictions(predictions, top=1)[0]

    return decoded_preds[0][1], decoded_preds[0][2] # Returns Label and Confidence

# --- Sidebar Input ---
st.sidebar.header("Input Source")
option = st.sidebar.radio("Choose Input:", ("Webcam (Live)", "Upload Image"))

input_image = None

if option == "Webcam (Live)":
    img_file = st.camera_input("Take a picture")
    if img_file is not None:
        input_image = Image.open(img_file)
elif option == "Upload Image":
    img_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])
    if img_file is not None:
        input_image = Image.open(img_file)
        st.image(input_image, caption="Uploaded Image", use_column_width=True)

# --- Results & Traffic Logic ---
if input_image is not None:
    st.divider()
    st.subheader("Analysis Result")

    # Run the prediction
    label, confidence = predict_image(input_image)

    # Check if it is an ambulance
    # We check if the word 'ambulance' is inside the detected label
    is_ambulance = 'ambulance' in label.lower()

    col1, col2 = st.columns([1, 2])

    with col1:
        st.info("Detection Details")
        st.metric("Object", label.replace("_", " ").title())
        st.metric("Confidence", f"{confidence*100:.2f}%")

    with col2:
        if is_ambulance:
            st.success("## 🟢 SIGNAL TURNED GREEN")
            st.write("**PRIORITY ALERT:** Emergency Vehicle Detected.")
            st.write("Traffic lights have been switched to clear the path.")
        else:
            st.error("## 🔴 SIGNAL REMAINS RED")
            st.write("**STATUS:** Routine Traffic Flow.")
            st.write(f"Detected a {label}, which follows standard traffic rules.")
