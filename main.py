import streamlit as st
import cv2 as cv
import numpy as np
import keras
from PIL import Image
import webbrowser

# Disease labels
label_name = ['Apple scab', 'Apple Black rot', 'Apple Cedar apple rust', 'Apple healthy', 
              'Cherry Powdery mildew', 'Cherry healthy', 'Corn Cercospora leaf spot Gray leaf spot', 
              'Corn Common rust', 'Corn Northern Leaf Blight', 'Corn healthy', 'Grape Black rot', 
              'Grape Esca', 'Grape Leaf blight', 'Grape healthy', 'Peach Bacterial spot', 'Peach healthy', 
              'Pepper bell Bacterial spot', 'Pepper bell healthy', 'Potato Early blight', 'Potato Late blight', 
              'Potato healthy', 'Strawberry Leaf scorch', 'Strawberry healthy', 'Tomato Bacterial spot', 
              'Tomato Early blight', 'Tomato Late blight', 'Tomato Leaf Mold', 'Tomato Septoria leaf spot', 
              'Tomato Spider mites', 'Tomato Target Spot', 'Tomato Yellow Leaf Curl Virus', 'Tomato mosaic virus', 
              'Tomato healthy']

# Set page style
st.markdown("""
    <style>
        body {
            background-color: #f4f4f4;
        }
        .stApp {
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
            text-align: center;
        }
        h1 {
            color: #4CAF50;
        }
    </style>
""", unsafe_allow_html=True)

# Page title
st.title("\U0001F33F AI-Powered Leaf Disease Detection")

# Load the model
model = keras.models.load_model('Training/model/Leaf Deases(96,88).h5')

def is_leaf_image(image):
    img = np.array(image)
    img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([90, 255, 255])
    mask = cv.inRange(hsv, lower_green, upper_green)
    green_percentage = (np.sum(mask > 0) / mask.size) * 100
    return green_percentage > 30

# Upload or capture options
col1, col2 = st.columns(2)
with col1:
    uploaded_file = st.file_uploader("\U0001F4C2 Upload an Image", type=["jpg", "png", "jpeg"])
with col2:
    open_camera = st.checkbox("\U0001F4F8 Open Camera")
    camera_image = st.camera_input("Take a Photo") if open_camera else None

def process_image(image):
    img = Image.open(image)
    img = np.array(img)
    img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
    normalized_image = np.expand_dims(cv.resize(cv.cvtColor(img, cv.COLOR_BGR2RGB), (150, 150)), axis=0)
    return img, normalized_image

if uploaded_file or camera_image:
    image_source = uploaded_file if uploaded_file else camera_image
    img, normalized_image = process_image(image_source)
    st.image(img, caption="\U0001F4F7 Processed Image", use_column_width=True)

    if is_leaf_image(img):
        st.success("\u2705 Leaf detected. Processing...")

        with st.spinner("\U0001F50D Analyzing Image..."):
            predictions = model.predict(normalized_image)
        
        progress = st.progress(0)
        for i in range(101):
            progress.progress(i)

        confidence = predictions[0][np.argmax(predictions)] * 100
        detected_disease = label_name[np.argmax(predictions)]
        
        if confidence >= 80:
            st.success(f"\u2705 Disease Detected: **{detected_disease}**")
            st.write(f"\U0001F3AF **Confidence:** {confidence:.2f}%")
            
            # Remedy button
            query = f"remedy for {detected_disease}"
            chatgpt_url = f"https://chat.openai.com/?q={query}"
            if st.button("🩺 Get Remedy"):
                webbrowser.open(chatgpt_url)
        else:
            st.error("⚠️ Unable to confidently determine the disease. Try another image.")
    else:
        st.error("❌ This is not a leaf image. Please upload a valid leaf image.")
