import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

model = tf.keras.models.load_model("models/image_model.h5")

class_names = ['cats', 'dogs']

st.title("AI Image Classifier")

uploaded_file = st.file_uploader("Upload Image")

if uploaded_file is not None:

    img = image.load_img(uploaded_file, target_size=(224,224))

    st.image(uploaded_file)

    img_array = image.img_to_array(img) / 255.0

    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    predicted_class = class_names[np.argmax(prediction)]

    confidence = np.max(prediction)

    st.write("Prediction:", predicted_class)

    st.write("Confidence:", confidence)