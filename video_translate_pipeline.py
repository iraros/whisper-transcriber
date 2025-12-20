import os

import streamlit as st

from logic import Transcriber
from modules.send_email import send_video_email
from modules.sub_embedding import embed_subtitles
from modules.translation import translate_srt_with_gpt


def run_full_pipeline(
    video_path: str,
    email: str,
    smtp_user: str,
    smtp_pass: str
):
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
        email,
        smtp_user,
        smtp_pass
    )
    for path in [video_path, original_srt_path, english_srt_path, subtitled_video_path]:
        if os.path.exists(path):
            os.remove(path)

    return subtitled_video_path

