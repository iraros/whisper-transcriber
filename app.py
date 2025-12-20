import os

import streamlit as st

from logic import Transcriber
from video_translate_pipeline import run_full_pipeline


def simple_transcriber():
    # Streamlit Interface
    st.title("Video to SRT Subtitle Generator 🎥")

    # File uploader
    uploaded_file = st.file_uploader("Upload your video file (MP4/MOV)", type=["mp4", "mov"])

    # Check if a file is uploaded
    if uploaded_file is not None:
        # Save the uploaded file to disk temporarily
        video_path = uploaded_file.name
        with open(video_path, "wb") as f:
            f.write(uploaded_file.read())

        # Add a button to start processing
        if st.button("Start Transcription"):
            st.info("Processing your file. Please wait...")
            srt_path = None
            try:
                # Create the Transcriber instance and process the video
                t = Transcriber()
                srt_path = t.transcribe_video(video_path)

                # Display success message and download button
                st.success("Transcription complete!")
                with open(srt_path, "rb") as f:
                    st.download_button(label="Download SRT File", data=f, file_name=os.path.basename(srt_path),
                                       mime="text/plain")
            except Exception as e:
                st.error(f"Error: {e}")
            finally:
                # Clean up temporary files
                if os.path.exists(video_path):
                    os.remove(video_path)
                if srt_path and os.path.exists(srt_path):
                    os.remove(srt_path)


def video_translator():

    st.title("video_translator 🎥")
    uploaded_file = st.file_uploader("Upload your video file (MP4/MOV)", type=["mp4", "mov"])

    # Check if a file is uploaded
    if uploaded_file is not None:
        # Save the uploaded file to disk temporarily
        video_path = uploaded_file.name
        with open(video_path, "wb") as f:
            f.write(uploaded_file.read())

        # Add a button to start processing
        if st.button("Start Processing"):
            with st.spinner("Processing video. This may take several minutes..."):
                run_full_pipeline(
                    video_path=video_path,
                    email=st.secrets["SMTP_USER"],
                    smtp_user=st.secrets["SMTP_USER"],
                    smtp_pass=st.secrets["SMTP_PASS"]
                )
            st.success("Done. Video sent by email.")


mode = st.sidebar.selectbox("Mode", ["Translate & subtitle video", "Transcribe only"])
if mode == "Transcribe only":
    simple_transcriber()
else:
    video_translator()
