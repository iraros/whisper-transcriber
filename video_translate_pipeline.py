import os
import smtplib
from email.message import EmailMessage
from pathlib import Path

import pysrt
import streamlit as st
from openai import OpenAI
from tqdm import tqdm

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
import os
import textwrap
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from moviepy import VideoFileClip, ImageClip, CompositeVideoClip
from moviepy.video.tools.subtitles import SubtitlesClip


def create_text_image(text, video_width, font_size=50):
    # Try to load a font that exists on Streamlit Cloud (Linux)
    try:
        # Streamlit Cloud (Debian) usually has these paths
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if not os.path.exists(font_path):
            font_path = "arial.ttf"  # Fallback for local Windows
        font = ImageFont.truetype(font_path, font_size)
    except:
        font = ImageFont.load_default()

    # Wrap text to 80% of width
    chars_per_line = max(1, int((video_width * 0.8) / (font_size * 0.5)))
    lines = textwrap.wrap(text, width=chars_per_line)
    wrapped_text = "\n".join(lines)

    # Calculate size
    temp_img = Image.new('RGBA', (video_width, 1000), (0, 0, 0, 0))
    draw = ImageDraw.Draw(temp_img)
    bbox = draw.multiline_textbbox((0, 0), wrapped_text, font=font, align="center")

    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    img = Image.new('RGBA', (video_width, h + 40), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Position
    pos = ((video_width - w) // 2, 10)

    # Simple outline for visibility
    for o in range(-2, 3):
        draw.multiline_text((pos[0] + o, pos[1]), wrapped_text, font=font, fill="black", align="center")
        draw.multiline_text((pos[0], pos[1] + o), wrapped_text, font=font, fill="black", align="center")

    draw.multiline_text(pos, wrapped_text, font=font, fill="white", align="center")
    return np.array(img)


def embed_subtitles(video_path, srt_path, output_path="output_video.mp4"):
    video = VideoFileClip(str(video_path))

    # Define the generator explicitly
    def make_text(txt):
        img_data = create_text_image(txt, video.w)
        return ImageClip(img_data, transparent=True)

    # CRITICAL: Use 'make_textclip=' as a keyword argument
    # This prevents MoviePy from misinterpreting the function as a font path
    subtitles = SubtitlesClip(str(srt_path), make_textclip=make_text)

    # In MoviePy 2.x, use with_position instead of set_position
    subtitles = subtitles.with_position('center')

    final_video = CompositeVideoClip([video, subtitles])

    final_video.write_videofile(
        output_path,
        fps=video.fps,
        codec="libx264",
        audio_codec="aac",
        threads=4  # Faster encoding on cloud
    )

    video.close()
    return output_path
# ---------- EMAIL ----------

def send_video_email(
    original_file_name: str,
    video_path: str,
    recipient: str,
    smtp_user: str,
    smtp_pass: str
):
    msg = EmailMessage()
    msg["Subject"] = f"Translated video from {original_file_name}"
    msg["From"] = smtp_user
    msg["To"] = recipient
    msg.set_content("Attached is your video with translated subtitles.")

    with open(video_path, "rb") as f:
        msg.add_attachment(
            f.read(),
            maintype="video",
            subtype="mp4",
            filename=video_path
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
        video_path,
        subtitled_video_path,
        email,
        smtp_user,
        smtp_pass
    )
    for path in [video_path, original_srt_path, english_srt, subtitled_video_path]:
        if os.path.exists(path):
            os.remove(path)

    return subtitled_video_path

