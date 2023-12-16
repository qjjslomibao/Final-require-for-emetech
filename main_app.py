import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Add the following lines to explicitly set __file__
if __name__ == '__main__':
    import __main__ as main
    setattr(main, '__file__', 'main_app.py')  # Replace with your actual script name

# Define a dummy hash function for builtins.function
def my_hash_func(func):
    return hash(func.__code__)

@st.cache(allow_output_mutation=True, hash_funcs={type(lambda: None): my_hash_func})
def load_model():
    model = tf.keras.models.load_model('/content/drive/MyDrive/EMTECH 2 FINAL REQUIREMENT DATASET/best_model.h5')
    return model

st.title("Emtech2 - Emotion Prediction App")

model = load_model()

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
