import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array

# Load your trained model
model = tf.keras.models.load_model('model_weights/vgg19_model_01.h5')

# Image size (match your model input)
IMAGE_SIZE = (128, 128)

# App title and description
st.set_page_config(page_title="Pneumonia Detector", layout="centered")
st.title("🩺 Pneumonia Detection from Chest X-ray")
st.write("Upload a chest X-ray image, and this app will predict whether the person has **Pneumonia** or is **Normal**.")

# Upload image
uploaded_file = st.file_uploader("Choose a Chest X-ray image", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Chest X-ray", use_column_width=True)

    # Preprocess image
    img = image.convert("RGB")
    img = img.resize(IMAGE_SIZE)
    img_array = img_to_array(img) / 255.0  # normalize
    img_array = np.expand_dims(img_array, axis=0)  # batch dimension

    # Predict
    prediction = model.predict(img_array)[0][0]  # Assuming binary classification

    # Show result
    st.markdown("---")
    if prediction > 0.5:
        st.error("🦠 **Pneumonia Detected** (Confidence: {:.2f}%)".format(prediction * 100))
    else:
        st.success("✅ **Normal** (Confidence: {:.2f}%)".format((1 - prediction) * 100))
