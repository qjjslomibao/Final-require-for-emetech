# Names:
#    Lomibao, Justin Joshua
#    Genabe, Richmond John
#    Jimenez, Maw
#    Carl Voltair
#    Lance Macamus
# Course & Section: CPE019-CPE32S1
# Instructor: Dr. Jonathan V. Taylar
# Date Perform: December 10, 2023
# Date Submitted: December 11, 2023
pip install -r packages.txt
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('best_model.h5')
    return model

model = load_model()

st.title("Emtech2 - Emotion Prediction App")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    img = img.resize((64, 64))
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = "Happy" if prediction[0, 0] >= 0.5 else "Sad"
    confidence = prediction[0, 0] if predicted_class == "Happy" else 1 - prediction[0, 0]
    confidence_scalar = float(confidence)

    st.image(img, caption=f'Predicted Class: {predicted_class} (Confidence: {confidence_scalar:.2f})', use_column_width=True)
