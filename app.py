import streamlit as st
import torch
import cv2 as c
import numpy as np
from PIL import Image

# Load custom CSS and Bootstrap
def load_styles():
    st.markdown("""
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
    <style>
    body {
        font-family: 'Arial', sans-serif;
        background-color: #f8f9fa;
    }
    .title {
        font-size: 3em;
        text-align: center;
        color: #007bff;
        margin-top: 20px;
    }
    .main-container {
        text-align: center;
        margin-top: 50px;
    }
    .stButton button {
        background-color: #28a745;
        color: white;
        font-size: 16px;
        padding: 10px 20px;
        border: none;
        border-radius: 5px;
        cursor: pointer;
    }
    .stButton button:hover {
        background-color: #218838;
    }
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        text-align: center;
        background-color: #343a40;
        color: white;
        padding: 10px;
        font-size: 14px;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model(model_path):
    return torch.hub.load("ultralytics/yolov5", "custom", path=model_path)

def main():
    load_styles()  # Load styles
    st.markdown('<div class="title">YOLOv5 Detection App</div>', unsafe_allow_html=True)

    # Main content
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    if st.button("Live Detection"):
        st.subheader("Live Detection")
        model = load_model("yolov5s.pt")
        st.write("Click **Start Live Detection** to begin.")
        if st.button("Start Live Detection"):
            live_detection(model)
    
    elif st.button("Upload Image"):
        st.subheader("Upload an Image for Detection")
        model = load_model("yolov5m.pt")
        uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            detect_image(model, image)
    
    st.markdown('</div>', unsafe_allow_html=True)

def live_detection(model):
    cap = c.VideoCapture(0)
    if not cap.isOpened():
        st.error("Unable to access the webcam.")
        return

    stframe = st.empty()  # Placeholder for video stream

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to grab frame.")
            break

        results = model(frame, size=640)  # Run detection on frame
        results.render()  # Modify the image with detections

        # Extract the rendered frame from results
        frame_rgb = results.ims[0]  # Use `ims` to get the modified image

        stframe.image(frame_rgb, channels="RGB", use_column_width=True)

        # Exit logic for quitting the loop
        if c.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    c.destroyAllWindows()

def detect_image(model, image):
    np_image = np.array(image)
    np_image = cv2_to_rgb(np_image)

    results = model(np_image, size=640)
    results.render()  # Render detections on the image

    rendered_image = results.ims[0]  # Extract the rendered image from results
    if rendered_image is not None:
        rendered_image = cv2_to_image(rendered_image)
        st.image(rendered_image, caption="Detected Objects", use_column_width=True)
    else:
        st.error("Error processing the image. Please try again.")

def cv2_to_image(cv2_image):
    return cv2_image[:, :, ::-1]  # Convert BGR to RGB for Streamlit

def cv2_to_rgb(np_image):
    return c.cvtColor(np_image, c.COLOR_RGB2BGR)  # Convert RGB to BGR for YOLOv5

if __name__ == "__main__":
    main()

    # Footer
    st.markdown("""
    <div class="footer">
        Developed by <a href="https://rauf-psi.vercel.app/" style="color: #007bff;">Rauf</a> | YOLOv5 Detection App
    </div>
    """, unsafe_allow_html=True)
