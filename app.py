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
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np

# Define a simple neural network for emotion prediction
class EmotionModel(nn.Module):
    def __init__(self):
        super(EmotionModel, self).__init__()
        # Modify this architecture based on your specific requirements
        self.fc = nn.Linear(64 * 64 * 3, 1)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.sigmoid(self.fc(x))
        return x

# Load the pre-trained PyTorch model
@st.cache(allow_output_mutation=True)
def load_model():
    model = EmotionModel()
    model.load_state_dict(torch.load('best_model.pth', map_location=torch.device('cpu')))
    return model

model = load_model()

st.title("Emtech2 - Emotion Prediction App")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    img = img.resize((64, 64))
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_tensor = torch.from_numpy(np.transpose(img_array, (2, 0, 1))).float()
    img_tensor = img_tensor.unsqueeze(0)

    # Forward pass through the PyTorch model
    prediction = model(img_tensor)
    predicted_class = "Happy" if prediction.item() >= 0.5 else "Sad"
    confidence = prediction.item() if predicted_class == "Happy" else 1 - prediction.item()
    confidence_scalar = float(confidence)

    st.image(img, caption=f'Predicted Class: {predicted_class} (Confidence: {confidence_scalar:.2f})', use_column_width=True)
