import streamlit as st
import cv2
import numpy as np

def process_image(uploaded_file):
    if uploaded_file is not None:
        # Read the image
        image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)

        # Convert the image to grayscale
        grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Display the original and grayscale images
        st.image([image, grayscale_image], caption=['Original Image', 'Grayscale Image'], use_column_width=True)

def main():
    st.title("Happy or Sad Detection")
    st.write("The concept of the project is based on the midterm exam that identifies the weather. We applied a CNN model to train the model for detecting if the face is happy or sad. You can upload a photo reserved from the Google Drive link that we also submitted.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        process_image(uploaded_file)

if __name__ == "__main__":
    main()
