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
import numpy as np
import textwrap
from PIL import Image, ImageDraw, ImageFont
from moviepy import VideoFileClip, ImageClip, CompositeVideoClip
from moviepy.video.tools.subtitles import SubtitlesClip


def create_text_image(text, video_width, font_size=60, color=(255, 255, 255)):
    # 1. Load Font
    try:
        # Windows: "arial.ttf", Linux: "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()

    # 2. Wrap Text: Determine how many characters fit in ~80% of the video width
    # Average character width is roughly font_size * 0.5 for sans-serif fonts
    chars_per_line = int((video_width * 0.8) / (font_size * 0.45))
    lines = textwrap.wrap(text, width=chars_per_line)
    wrapped_text = "\n".join(lines)

    # 3. Calculate canvas size needed for wrapped text
    # We use a dummy image to calculate the bounding box of the whole block
    temp_img = Image.new('RGBA', (video_width, 1000), (0, 0, 0, 0))
    temp_draw = ImageDraw.Draw(temp_img)
    bbox = temp_draw.multiline_textbbox((0, 0), wrapped_text, font=font, align="center")

    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    # Create final canvas (adding padding)
    canvas_h = text_h + 40
    img = Image.new('RGBA', (video_width, canvas_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # 4. Draw Text centered on canvas
    position = ((video_width - text_w) // 2, 10)

    # Draw outline (Stroke)
    stroke_width = 3
    for x_offset in range(-stroke_width, stroke_width + 1):
        for y_offset in range(-stroke_width, stroke_width + 1):
            draw.multiline_text((position[0] + x_offset, position[1] + y_offset),
                                wrapped_text, font=font, fill=(0, 0, 0, 255), align="center")

    # Draw main text
    draw.multiline_text(position, wrapped_text, font=font, fill=color, align="center")

    return np.array(img)


def embed_subtitles(video_path, srt_path, output_path="output_video.mp4"):
    video = VideoFileClip(str(video_path))

    def generator(txt):
        # We pass the video width so the function knows where to wrap
        img_array = create_text_image(txt, video.w, font_size=55)
        return ImageClip(img_array, transparent=True)

    subtitles = SubtitlesClip(str(srt_path), make_textclip=generator)

    # Center the subtitle block on the screen
    subtitles = subtitles.with_position('center')

    final_video = CompositeVideoClip([video, subtitles])

    final_video.write_videofile(
        output_path,
        fps=video.fps,
        codec="libx264",
        audio_codec="aac"
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

