import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load the pre-trained model
model = tf.keras.models.load_model('model.h5')

# Function to preprocess the uploaded image
def preprocess_image(image):
    image = image.resize((128, 128))  # Resize to the input size required by the model
    #image = np.array(image) / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to make predictions
def predict(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    return 'Healthy' if prediction < 0.5 else 'Powdery Mildew'

# Streamlit app layout
st.title('Cherry Leaf Mildew Detection')
st.write('Upload an image of a cherry leaf to detect if it is healthy or has powdery mildew.')

# Hypotheses Section
st.header('Hypotheses')
st.write("""
In this section, we outline the key hypotheses related to the detection of powdery mildew on cherry leaves:

1. **Image Quality**: High-quality images with clear visibility of the leaf surface will yield more accurate predictions.
2. **Leaf Texture and Color**: The model distinguishes between healthy leaves and those affected by mildew based on texture and color differences.
3. **Lighting Conditions**: Images taken under uniform lighting conditions improve the model's ability to correctly classify the health status of the leaf.
4. **Environmental Factors**: The appearance of powdery mildew might vary under different environmental conditions, affecting the model's accuracy.

These hypotheses guide our approach in both data collection and model training to improve the detection accuracy.
""")


# Image upload section
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Predict and display the result
    label = predict(image)
    st.write(f'The uploaded cherry leaf is: **{label}**')

# Run the app using the command: streamlit run app.py
