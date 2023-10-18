from streamlit_webrtc import webrtc_streamer
import av
import streamlit as st
from streamlit.logger import get_logger
from helper import process_frame
logger = get_logger(__name__)

st.header('Gender, Age, and Ethnicity Prediction')
def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    new_frame= process_frame(img)
    return av.VideoFrame.from_ndarray(new_frame, format="bgr24")


webrtc_streamer(key="example", video_frame_callback=video_frame_callback)