import os

# WORKAROUND for libiomp5md.dll error on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import logging
import asyncio
import sys
import subprocess
import textwrap
import cv2
import numpy as np
from pathlib import Path

# Handle MoviePy v1 and v2 import differences
try:
    from moviepy.editor import VideoFileClip
except ImportError:
    try:
        from moviepy import VideoFileClip
    except ImportError:
        print("❌ CRITICAL: 'moviepy' not found. Please run: pip install moviepy")

# -----------------------------------------------------------------------------
# DEPENDENCY CHECK
# -----------------------------------------------------------------------------
try:
    from telegram import Update
    from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters
except ImportError:
    print("\n" + "!" * 60)
    print("CRITICAL IMPORT ERROR")
    print("The 'telegram' module was not found or is the wrong version.")
    print("Please run: pip uninstall telegram && pip install python-telegram-bot")
    print("!" * 60 + "\n")
    sys.exit(1)

try:
    from faster_whisper import WhisperModel
    from openai import OpenAI
except ImportError as e:
    print(f"\nCRITICAL IMPORT ERROR: Missing dependency {e.name}")
    print(f"Please run: pip install faster-whisper openai")
    sys.exit(1)

# -----------------------------------------------------------------------------
# CONFIGURATION & PATH SETUP
# -----------------------------------------------------------------------------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_TELEGRAM_BOT_TOKEN_HERE")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY_HERE")

REPO_DIR = Path(__file__).parent.absolute()
LOCAL_TMP_DIR = REPO_DIR / "tmp"
LOCAL_TMP_DIR.mkdir(exist_ok=True)

FILES_FROM_PREVIOUS_RUN = []

LANGUAGE_DATA = {
    'ar': ('🇸🇦', 'Arabic'), 'he': ('🇮🇱', 'Hebrew'), 'ru': ('🇷🇺', 'Russian'),
    'es': ('🇪🇸', 'Spanish'), 'fr': ('🇫🇷', 'French'), 'de': ('🇩🇪', 'German'),
    'it': ('🇮🇹', 'Italian'), 'pt': ('🇵🇹', 'Portuguese'), 'ja': ('🇯🇵', 'Japanese'),
    'ko': ('🇰🇷', 'Korean'), 'zh': ('🇨🇳', 'Chinese'), 'tr': ('🇹🇷', 'Turkish'),
    'nl': ('🇳🇱', 'Dutch'), 'pl': ('🇵🇱', 'Polish'), 'uk': ('🇺🇦', 'Ukrainian'),
    'hi': ('🇮🇳', 'Hindi'), 'fa': ('🇮🇷', 'Persian'), 'en': ('🇺🇸', 'English'),
    'vi': ('🇻🇳', 'Vietnamese'), 'th': ('🇹🇭', 'Thai')
}

WHISPER_MODEL_SIZE = "medium"
DEVICE_TYPE = "cpu"
COMPUTE_TYPE = "int8"

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

openai_client = OpenAI(api_key=OPENAI_API_KEY)

logger.info(f"Loading Whisper model: {WHISPER_MODEL_SIZE}...")
try:
    whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device=DEVICE_TYPE, compute_type=COMPUTE_TYPE)
except Exception as e:
    logger.error(f"Failed to load Whisper model: {e}")
    whisper_model = None


# -----------------------------------------------------------------------------
# UTILS
# -----------------------------------------------------------------------------
def format_timestamp(seconds: float):
    """Converts seconds to SRT timestamp format (HH:MM:SS,mmm)"""
    milliseconds = int((seconds - int(seconds)) * 1000)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02}:{minutes:02}:{secs:02},{milliseconds:03}"


# -----------------------------------------------------------------------------
# SUBTITLE RENDERING LOGIC (Manual OpenCV + MoviePy)
# -----------------------------------------------------------------------------
def draw_text_with_outline(img, text, font=cv2.FONT_HERSHEY_DUPLEX, font_scale=1.2, thickness=2):
    """Draws Arial-style bold subtitles with a thick black outline."""
    h, w, _ = img.shape
    max_width_chars = max(15, int(w / 30))
    wrapped_lines = textwrap.wrap(text, width=max_width_chars)

    line_height = int(50 * font_scale)
    bottom_margin = 80

    total_text_height = len(wrapped_lines) * line_height
    start_y = h - total_text_height - bottom_margin

    for i, line in enumerate(wrapped_lines):
        text_size = cv2.getTextSize(line, font, font_scale, thickness)[0]
        text_x = (w - text_size[0]) // 2
        text_y = start_y + (i * line_height) + text_size[1]

        cv2.putText(img, line, (text_x, text_y), font, font_scale, (0, 0, 0), thickness + 4, cv2.LINE_AA)
        cv2.putText(img, line, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


def create_subtitled_video(input_video: str, timed_segments: list, output_video: str):
    """Manually renders subtitles and merges audio."""
    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    temp_silent = str(LOCAL_TMP_DIR / f"silent_{os.path.basename(input_video)}")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_silent, fourcc, fps, (width, height))

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        current_time = frame_count / fps
        active_text = next((s['text'] for s in timed_segments if s['start'] <= current_time <= s['end']), None)

        if active_text:
            draw_text_with_outline(frame, active_text)

        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()

    try:
        orig_clip = VideoFileClip(input_video)
        rendered_clip = VideoFileClip(temp_silent)

        if hasattr(rendered_clip, 'with_audio'):
            final_clip = rendered_clip.with_audio(orig_clip.audio)
        else:
            final_clip = rendered_clip.set_audio(orig_clip.audio)

        final_clip.write_videofile(output_video, codec="libx264", audio_codec="aac", logger=None)

        orig_clip.close()
        rendered_clip.close()
        if os.path.exists(temp_silent):
            os.remove(temp_silent)
        return True
    except Exception as e:
        logger.error(f"Media merge failed: {e}")
        return False


# -----------------------------------------------------------------------------
# TRANSCRIPTION & TRANSLATION LOGIC
# -----------------------------------------------------------------------------
def transcribe_and_translate(video_path: str):
    segments, info = whisper_model.transcribe(video_path, beam_size=5)

    segments_list = list(segments)
    # Joining with a delimiter so GPT can translate while keeping segments distinct
    full_transcript_str = " || ".join([s.text.strip() for s in segments_list])

    source_name = LANGUAGE_DATA.get(info.language, ('🌍', info.language.upper()))[1]

    # Improved GPT Translation Prompt to maintain segments
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system",
             "content": f"Translate to natural English subtitles. Maintain the ' || ' delimiters to keep segment timing. Source: {source_name}. Output ONLY translation."},
            {"role": "user", "content": full_transcript_str}
        ],
        temperature=0.3
    )
    translation_blob = response.choices[0].message.content
    translated_parts = [p.strip() for p in translation_blob.split("||")]

    timed_segments = []
    for i, s in enumerate(segments_list):
        # Fallback to original text if GPT messed up the segment count
        text = translated_parts[i] if i < len(translated_parts) else s.text
        timed_segments.append({'start': s.start, 'end': s.end, 'text': text})

    full_orig = " ".join([s.text for s in segments_list])
    full_trans = " ".join(translated_parts)

    return full_orig, full_trans, timed_segments, info.language, info.language_probability


# -----------------------------------------------------------------------------
# TELEGRAM HANDLERS
# -----------------------------------------------------------------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("👋 Send me a video for high-quality English subtitles!")


async def handle_video(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global FILES_FROM_PREVIOUS_RUN

    if FILES_FROM_PREVIOUS_RUN:
        for old_path in FILES_FROM_PREVIOUS_RUN:
            try:
                if os.path.exists(old_path): os.remove(old_path)
            except:
                pass
        FILES_FROM_PREVIOUS_RUN = []

    status_msg = await update.message.reply_text("📥 Processing...")
    media = update.message.video or update.message.video_note
    if not media: return

    file_id = media.file_id
    unique_id = f"proc_{file_id[:10]}"
    temp_in = str(LOCAL_TMP_DIR / f"{unique_id}_in.mp4")
    temp_out = str(LOCAL_TMP_DIR / f"{unique_id}_out.mp4")
    temp_srt = str(LOCAL_TMP_DIR / f"{unique_id}.srt")

    try:
        new_file = await context.bot.get_file(file_id)
        await new_file.download_to_drive(custom_path=temp_in)

        await status_msg.edit_text("⚙️ Transcribing & Translating...")
        loop = asyncio.get_running_loop()
        orig, trans, segments, l_code, prob = await loop.run_in_executor(None, transcribe_and_translate, temp_in)

        # Save SRT with proper timings
        with open(temp_srt, "w", encoding="utf-8") as f:
            for i, seg in enumerate(segments):
                f.write(
                    f"{i + 1}\n{format_timestamp(seg['start'])} --> {format_timestamp(seg['end'])}\n{seg['text']}\n\n")

        await status_msg.edit_text("🎬 Rendering video with synced subtitles...")
        # CRITICAL FIX: Pass the 'segments' list (with individual start/end) instead of a single giant block
        success = await loop.run_in_executor(None, create_subtitled_video, temp_in, segments, temp_out)

        caption = f"🇺🇸 **Translation:**\n{trans}"
        if success:
            await status_msg.edit_text("✅ Done! Uploading...")
            await update.message.reply_video(video=open(temp_out, 'rb'), caption=caption[:1024], parse_mode="Markdown")
        else:
            await status_msg.edit_text("⚠️ Render failed. Sending text.")
            await update.message.reply_text(caption)

    except Exception as e:
        logger.error(f"Error: {e}")
        await status_msg.edit_text(f"❌ Error: {str(e)}")

    FILES_FROM_PREVIOUS_RUN.extend([temp_in, temp_out, temp_srt])


if __name__ == '__main__':
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.VIDEO | filters.VIDEO_NOTE, handle_video))
    app.run_polling()