import os
import smtplib
from email.message import EmailMessage
from pathlib import Path

import pysrt
import streamlit as st
from openai import OpenAI
from tqdm import tqdm

# IMPORT YOUR EXISTING FUNCTION
from logic import Transcriber

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
def translate_srt_with_gpt(original_srt_path: Path, target_lang="English") -> Path:
    """
    Translates an existing SRT file to English using GPT models.
    Handles dialects like Levantine Arabic.
    """
    # Load original SRT
    subs = pysrt.open(original_srt_path, encoding="utf-8")
    translated_subs = pysrt.SubRipFile()

    for i, sub in tqdm(enumerate(subs, start=1), total=len(subs)):
        # Prompt GPT to translate each subtitle
        prompt = (
            f"Translate the following text into natural {target_lang}, "
            f"keeping the spoken/dialect style. Preserve meaning:\n{sub.text}"
        )

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a translator to English."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
        )

        translated_text = response.choices[0].message.content.strip()

        translated_subs.append(
            pysrt.SubRipItem(
                index=i,
                start=sub.start,
                end=sub.end,
                text=translated_text
            )
        )

    # Save the translated SRT
    english_srt_path = original_srt_path.with_suffix(".en.srt")
    translated_subs.save(english_srt_path, encoding="utf-8")
    return english_srt_path._str


# ---------- EMBED SUBTITLES ----------
from moviepy.config import change_settings
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
from moviepy.video.tools.subtitles import SubtitlesClip

# Update this path to where you actually installed ImageMagick
# It is usually in Program Files
change_settings({"IMAGEMAGICK_BINARY": r"C:\Program Files\ImageMagick-7.1.2-Q16\magick.exe"})

def embed_subtitles(video_path, srt_path, output_path="output_video.mp4"):
    # 1. Load the video
    video = VideoFileClip(video_path)

    # 2. Define the look of the subtitles
    # Customize 'fontsize', 'color', and 'font' as needed
    generator = lambda txt: TextClip(
        txt,
        font='Arial-Bold',
        fontsize=70,
        color='white',
        stroke_color='black',
        stroke_width=2,
        method='caption',
        size=(video.w * 0.8, None)  # Wrap text at 80% of video width
    )

    # 3. Initialize the SubtitlesClip
    subtitles = SubtitlesClip(srt_path, generator)

    # 4. Set the position to the middle of the screen
    # 'center' handles both horizontal and vertical alignment
    subtitles = subtitles.set_position('center')

    # 5. Overlay the subtitles on the original video
    final_video = CompositeVideoClip([video, subtitles])

    # 6. Write the result to a file
    # 'libx264' is the standard codec for high compatibility
    final_video.write_videofile(output_path, fps=video.fps, codec="libx264", audio_codec="aac")
    print(f'saved video in {output_path}')
    return output_path



# ---------- EMAIL ----------

def send_video_email(
    video_path: Path,
    recipient: str,
    smtp_user: str,
    smtp_pass: str
):
    msg = EmailMessage()
    msg["Subject"] = "Your translated video"
    msg["From"] = smtp_user
    msg["To"] = recipient
    msg.set_content("Attached is your video with translated subtitles.")

    with open(video_path, "rb") as f:
        msg.add_attachment(
            f.read(),
            maintype="video",
            subtype="mp4",
            filename=video_path.name
        )

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
        s.login(smtp_user, smtp_pass)
        s.send_message(msg)


# ---------- FULL PIPELINE (CORRECT ORDER) ----------

def run_full_pipeline(
    video_path: str,
    email: str,
    smtp_user: str,
    smtp_pass: str
):
    """
    1. Use EXISTING transcribe_video() → original SRT
    2. Whisper translate → English SRT
    3. Merge
    4. Burn subtitles
    5. Email video
    """

    st.info("transcribing...")
    # 1. EXISTING transcription (DO NOT TOUCH)
    t = Transcriber()
    original_srt_path = t.transcribe_video(video_path)

    st.info("translating...")
    # 2. Translation only
    english_srt = translate_srt_with_gpt(Path(original_srt_path))

    st.info('embedding...')
    # 3. Embed into video
    subtitled_video_path = embed_subtitles(video_path, english_srt)

    st.info('sending result...')
    # 4. Email result
    send_video_email(
        Path(subtitled_video_path),
        email,
        smtp_user,
        smtp_pass
    )
    for path in [video_path, original_srt_path, english_srt, subtitled_video_path]:
        if os.path.exists(path):
            os.remove(path)

    return subtitled_video_path

