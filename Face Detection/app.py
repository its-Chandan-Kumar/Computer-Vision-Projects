# --------------------Import Libraries--------------------
import streamlit as st
import cv2
import numpy as np
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from PIL import Image


# --------------------Page Config--------------------
st.set_page_config(
    page_title="Face Detection App",
    page_icon="😎",
    layout="centered"
)

st.title("😎 Face Detection")
st.write("Detect faces from Images or Live webcam using Haar Cascade")


# --------------------Load Haar Cascade--------------------
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


# --------------------Sidebar--------------------
st.sidebar.header("Select Mode")
mode = st.sidebar.radio("Choose Input Type", ["Image Upload", "Webcam"])


# --------------------Image Upload Mode--------------------
if mode == "Image Upload":

    uploaded_file = st.file_uploader(
        "Upload an Image", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=6
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        st.success(f"Detected {len(faces)} face(s)")
        st.image(image, channels="BGR", caption="Processed Image")


# --------------------Webcam Mode--------------------
elif mode == "Webcam":

    st.info("Allow camera access to start Face Detection")

    def video_frame_callback(frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img,1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30,30),
        flags=cv2.CASCADE_SCALE_IMAGE
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

    webrtc_streamer(
        key="face-detection",
        mode=WebRtcMode.SENDRECV,
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )
