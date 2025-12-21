import os

import streamlit as st

from modules.send_email import send_video_email
from modules.sub_embedding import embed_subtitles
from modules.transcriber import Transcriber
from modules.translation import translate_srt_with_gpt
from modules.utils import get_tmp_path


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
        # Use your new utility function to put it in the /tmp folder
        video_path = get_tmp_path(uploaded_file.name)

        with open(video_path, "wb") as f:
            f.write(uploaded_file.read())

        with st.spinner("Processing video. This may take several minutes..."):

            st.info("transcribing...")
            t = Transcriber()
            original_srt_path = t.transcribe_video(video_path)

            st.info("translating...")
            english_srt_path = translate_srt_with_gpt(original_srt_path)

            st.info('embedding...')
            subtitled_video_path = embed_subtitles(video_path, english_srt_path)

            st.info('sending result...')
            send_video_email(
                video_path,
                subtitled_video_path,
                original_srt_path,
                english_srt_path,
                st.secrets["SMTP_USER"],
                st.secrets["SMTP_USER"],
                st.secrets["SMTP_PASS"]
            )

            tmp_dir = get_tmp_path('')
            for file_name in os.listdir(tmp_dir):
                path = os.path.join(tmp_dir, file_name)
                if os.path.exists(path):
                    os.remove(path)
        st.success("Done. Video sent by email.")


mode = st.sidebar.selectbox("Mode", ["Translate & subtitle video", "Transcribe only"])
if mode == "Transcribe only":
    simple_transcriber()
else:
    video_translator()
