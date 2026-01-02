import os
import logging
import asyncio
import sys
import textwrap
import cv2
import shutil
import socket
from pathlib import Path
from logging.handlers import RotatingFileHandler

# Fix for library conflicts
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Handle MoviePy v1/v2 differences
try:
    from moviepy.editor import VideoFileClip
except ImportError:
    try:
        from moviepy import VideoFileClip
    except ImportError:
        print("❌ CRITICAL: 'moviepy' not found.")

try:
    from telegram import Update
    from telegram.constants import ParseMode
    from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters
    from telegram.request import HTTPXRequest
    from openai import OpenAI
except ImportError as e:
    print(f"CRITICAL: Missing dependency. Run: pip install python-telegram-bot openai opencv-python-headless moviepy")
    sys.exit(1)

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Environment Detection
IS_LOCAL = os.getenv("IS_LOCAL", "false").lower() == "true"

BASE_DIR = Path(__file__).parent.absolute()
LOCAL_TMP_DIR = BASE_DIR / "tmp"
LOCAL_TMP_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# LOGGING SETUP
# -----------------------------------------------------------------------------
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log_file = BASE_DIR / "bot_log.log"

file_handler = RotatingFileHandler(log_file, maxBytes=5 * 1024 * 1024, backupCount=3, encoding='utf-8')
file_handler.setFormatter(log_formatter)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(log_formatter)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

if IS_LOCAL:
    TRANSLATION_MODEL = "gpt-3.5-turbo"
    USE_LOCAL_WHISPER = True
    try:
        from faster_whisper import WhisperModel

        whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
        logger.info("🛠 DEBUG MODE: Local Whisper + GPT-3.5")
    except ImportError:
        USE_LOCAL_WHISPER = False
else:
    TRANSLATION_MODEL = os.getenv("TRANSLATION_MODEL", "gpt-4o-mini")
    USE_LOCAL_WHISPER = False
    logger.info("🚀 PROD MODE: Cloud Whisper + GPT-4o-Mini")

LANGUAGE_DATA = {
    'ar': ('🇸🇦', 'Arabic'), 'he': ('🇮🇱', 'Hebrew'), 'ru': ('🇷🇺', 'Russian'),
    'es': ('🇪🇸', 'Spanish'), 'fr': ('🇫🇷', 'French'), 'de': ('🇩🇪', 'German'),
    'it': ('🇮🇹', 'Italian'), 'pt': ('🇵🇹', 'Portuguese'), 'ja': ('🇯🇵', 'Japanese'),
    'ko': ('🇰🇷', 'Korean'), 'zh': ('🇨🇳', 'Chinese'), 'tr': ('🇹🇷', 'Turkish'),
    'nl': ('🇳🇱', 'Dutch'), 'pl': ('🇵🇱', 'Polish'), 'uk': ('🇺🇦', 'Ukrainian'),
    'hi': ('🇮🇳', 'Hindi'), 'fa': ('🇮🇷', 'Persian'), 'en': ('🇺🇸', 'English'),
    'vi': ('🇻🇳', 'Vietnamese'), 'th': ('🇹🇭', 'Thai'), 'id': ('🇮🇩', 'Indonesian')
}

openai_client = OpenAI(api_key=OPENAI_API_KEY)


def cleanup_temp_folder():
    for file_path in LOCAL_TMP_DIR.glob('*'):
        try:
            if file_path.is_file():
                file_path.unlink()
            elif file_path.is_dir():
                shutil.rmtree(file_path)
        except:
            pass


def draw_text_with_outline(img, text):
    h, w, _ = img.shape
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = w / 600.0 if w > 300 else 0.5
    thickness = 1
    wrapped_lines = textwrap.wrap(text, width=max(12, int(w / 18)))
    line_height = int(35 * font_scale)
    start_y = h - (len(wrapped_lines) * line_height) - 30

    for i, line in enumerate(wrapped_lines):
        text_size = cv2.getTextSize(line, font, font_scale, thickness)[0]
        tx, ty = (w - text_size[0]) // 2, start_y + (i * line_height) + text_size[1]
        cv2.putText(img, line, (tx, ty), font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(img, line, (tx, ty), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


def create_subtitled_video(input_video: str, segments: list, output_video: str, progress_callback=None):
    logger.info(">>> STAGE: Visual Rendering Started")

    # NORMALIZATION STEP: Convert source to H264 to avoid AV1/Header issues
    processed_input = str(LOCAL_TMP_DIR / "normalized_source.mp4")
    try:
        if progress_callback: progress_callback("🔄 Preparing video format...")
        with VideoFileClip(input_video) as clip:
            clip.write_videofile(processed_input, codec="libx264", preset="ultrafast", audio=False, logger=None)
    except Exception as e:
        logger.error(f"Preprocessing/AV1 Normalization failed: {e}")
        return False, f"Codec error: Your platform cannot decode this video format (AV1/HEVC)."

    cap = cv2.VideoCapture(processed_input)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    temp_silent = str(LOCAL_TMP_DIR / "silent_tmp.mp4")
    out = cv2.VideoWriter(temp_silent, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    fc = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        t = fc / fps
        txt = next((s['text'] for s in segments if s['start'] <= t <= s['end']), None)
        if txt: draw_text_with_outline(frame, txt)
        out.write(frame)
        if fc % 30 == 0 and progress_callback:
            percent = int((fc / total_frames) * 100) if total_frames > 0 else 0
            progress_callback(f"🎬 Rendering subtitles... {percent}%")
        fc += 1

    cap.release()
    out.release()

    logger.info(">>> STAGE: Audio Merging Started")
    try:
        with VideoFileClip(input_video) as orig_clip:
            with VideoFileClip(temp_silent) as rendered_clip:
                final = rendered_clip.set_audio(orig_clip.audio)
                final.write_videofile(output_video, codec="libx264", audio_codec="aac", preset="ultrafast", logger=None)
        return True, None
    except Exception as e:
        logger.error(f"Merge error: {e}")
        return False, f"Audio merge failed: {str(e)}"
    finally:
        if os.path.exists(temp_silent): os.remove(temp_silent)
        if os.path.exists(processed_input): os.remove(processed_input)


async def transcribe_and_translate(video_path: str, duration: float, update_func):
    loop = asyncio.get_running_loop()
    audio_path = str(LOCAL_TMP_DIR / "audio.m4a")

    def ext_audio():
        with VideoFileClip(video_path) as v:
            if v.audio: v.audio.write_audiofile(audio_path, codec='aac', logger=None)

    await loop.run_in_executor(None, ext_audio)
    if not os.path.exists(audio_path): return None

    if USE_LOCAL_WHISPER:
        await update_func("⚙️ Local Transcription...")
        segs_gen, info = await loop.run_in_executor(None, lambda: whisper_model.transcribe(audio_path))
        raw_segs, lang_raw = list(segs_gen), info.language
    else:
        await update_func("☁️ Cloud Transcription...")
        res = await loop.run_in_executor(None, lambda: openai_client.audio.transcriptions.create(
            model="whisper-1", file=open(audio_path, "rb"), response_format="verbose_json",
            timestamp_granularities=["segment"]
        ))
        raw_segs, lang_raw = res.segments, res.language

    if not raw_segs: return None

    full_text = " || ".join([s.text.strip() if hasattr(s, 'text') else s['text'].strip() for s in raw_segs])
    l_code = lang_raw.lower()[:2]
    flag, name = LANGUAGE_DATA.get(l_code, ('🌍', lang_raw.upper()))

    await update_func(f"✨ Translating from {flag} {name}...")
    gpt = await loop.run_in_executor(None, lambda: openai_client.chat.completions.create(
        model=TRANSLATION_MODEL, messages=[
            {"role": "system", "content": "Translate to English. Keep ' || ' separators."},
            {"role": "user", "content": full_text}
        ]
    ))

    trans_parts = [p.strip() for p in gpt.choices[0].message.content.split("||")]
    timed = []
    for i, s in enumerate(raw_segs):
        st = s.start if hasattr(s, 'start') else s['start']
        en = s.end if hasattr(s, 'end') else s['end']
        txt = trans_parts[i] if i < len(trans_parts) else (s.text if hasattr(s, 'text') else s['text'])
        timed.append({'start': st, 'end': en, 'text': txt})

    return " ".join([s.text if hasattr(s, 'text') else s['text'] for s in raw_segs]), " ".join(
        trans_parts), timed, lang_raw


async def handle_video(update: Update, context: ContextTypes.DEFAULT_TYPE):
    media = update.message.video or update.message.video_note
    msg = await update.message.reply_text("📥 Receiving video...")
    fid = media.file_id
    t_in, t_out = str(LOCAL_TMP_DIR / f"i_{fid[:8]}.mp4"), str(LOCAL_TMP_DIR / f"o_{fid[:8]}.mp4")

    try:
        f = await context.bot.get_file(fid)
        await f.download_to_drive(custom_path=t_in)

        cap_info = cv2.VideoCapture(t_in)
        duration = cap_info.get(cv2.CAP_PROP_FRAME_COUNT) / (cap_info.get(cv2.CAP_PROP_FPS) or 30.0)
        cap_info.release()

        result = await transcribe_and_translate(t_in, duration, lambda t: msg.edit_text(t))
        if not result:
            await msg.edit_text("⚠️ No speech detected.")
            return

        orig, trans, segments, lang_raw = result
        loop = asyncio.get_running_loop()

        def prog(t):
            asyncio.run_coroutine_threadsafe(msg.edit_text(t), loop)

        success, err_msg = await loop.run_in_executor(None, create_subtitled_video, t_in, segments, t_out, prog)

        if success:
            l_code = lang_raw.lower()[:2]
            flag, name = LANGUAGE_DATA.get(l_code, ('🌍', lang_raw.upper()))
            cap = f"{flag} **{name}:**\n{orig}\n\n🇺🇸 **English:**\n{trans}"
            await update.message.reply_video(video=open(t_out, 'rb'), caption=cap[:1024], parse_mode=ParseMode.MARKDOWN)
            await msg.delete()
        else:
            await msg.edit_text(
                f"❌ **Error:** {err_msg}\n\n(This usually happens with unsupported codecs like AV1 on some servers)")

    except Exception as e:
        logger.error(f"Handler Error: {e}")
        await msg.edit_text(f"❌ **System Error:** {str(e)}")
    finally:
        for p in [t_in, t_out]:
            if os.path.exists(p): os.remove(p)


if __name__ == '__main__':
    if not TELEGRAM_BOT_TOKEN: sys.exit("Missing BOT_TOKEN")
    cleanup_temp_folder()
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", lambda u, c: u.message.reply_text("Send me a video!")))
    app.add_handler(MessageHandler(filters.VIDEO | filters.VIDEO_NOTE, handle_video))
    app.run_polling()