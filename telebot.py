import os
import logging
import asyncio
import sys
import textwrap
import cv2
import shutil
import socket
from pathlib import Path

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
    from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters
    from openai import OpenAI
except ImportError as e:
    print(f"CRITICAL: Missing dependency. Run: pip install python-telegram-bot openai opencv-python-headless moviepy")
    sys.exit(1)

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Check if running on GCP VM (usually has 'google' in hostname or specific metadata)
# You can also manually set IS_LOCAL="true" in your local environment variables
hostname = socket.gethostname()
is_local_env = os.getenv("IS_LOCAL", "false").lower() == "true"
IS_LOCAL = is_local_env or "google" not in hostname

# Explicit Logging for Environment Debugging
print("-" * 30)
print(f"DEBUG: IS_LOCAL status: {IS_LOCAL}")
print(f"DEBUG: Hostname detected: {hostname}")
print(f"DEBUG: IS_LOCAL Env Var: {is_local_env}")
print("-" * 30)

if IS_LOCAL:
    # Debug/Local settings: Cheaper/Faster for testing
    TRANSLATION_MODEL = "gpt-3.5-turbo"
    USE_LOCAL_WHISPER = True
    # If using local whisper, we need faster-whisper installed
    try:
        from faster_whisper import WhisperModel

        whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
        print("🛠 DEBUG MODE: Using local 'base' Whisper and GPT-3.5-Turbo")
    except ImportError:
        print("⚠️ faster-whisper not found, falling back to cloud whisper for debug.")
        USE_LOCAL_WHISPER = False
else:
    # Production/VM settings: High quality
    TRANSLATION_MODEL = os.getenv("TRANSLATION_MODEL", "gpt-4o-mini")
    USE_LOCAL_WHISPER = False
    print("🚀 PROD MODE: Using Cloud Whisper and GPT-4o-Mini")

BASE_DIR = Path(__file__).parent.absolute()
LOCAL_TMP_DIR = BASE_DIR / "tmp"
LOCAL_TMP_DIR.mkdir(parents=True, exist_ok=True)

LANGUAGE_DATA = {
    'ar': ('🇸🇦', 'Arabic'), 'he': ('🇮🇱', 'Hebrew'), 'ru': ('🇷🇺', 'Russian'),
    'es': ('🇪🇸', 'Spanish'), 'fr': ('🇫🇷', 'French'), 'de': ('🇩🇪', 'German'),
    'it': ('🇮🇹', 'Italian'), 'pt': ('🇵🇹', 'Portuguese'), 'ja': ('🇯🇵', 'Japanese'),
    'ko': ('🇰🇷', 'Korean'), 'zh': ('🇨🇳', 'Chinese'), 'tr': ('🇹🇷', 'Turkish'),
    'nl': ('🇳🇱', 'Dutch'), 'pl': ('🇵🇱', 'Polish'), 'uk': ('🇺🇦', 'Ukrainian'),
    'hi': ('🇮🇳', 'Hindi'), 'fa': ('🇮🇷', 'Persian'), 'en': ('🇺🇸', 'English'),
    'vi': ('🇻🇳', 'Vietnamese'), 'th': ('🇹🇭', 'Thai')
}

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

openai_client = OpenAI(api_key=OPENAI_API_KEY)


# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------
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
    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    target_w = 480
    if orig_w > target_w:
        ratio = target_w / float(orig_w)
        w, h = target_w, int(orig_h * ratio)
        logger.info(f"Aggressive Resize: {orig_w}x{orig_h} -> {w}x{h}")
    else:
        w, h = orig_w, orig_h

    temp_silent = str(LOCAL_TMP_DIR / "silent_tmp.mp4")
    out = cv2.VideoWriter(temp_silent, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    milestones = {int(total_frames * (i / 20)): i * 5 for i in range(1, 20)}

    fc = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        if w != orig_w:
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)

        t = fc / fps
        txt = next((s['text'] for s in segments if s['start'] <= t <= s['end']), None)
        if txt: draw_text_with_outline(frame, txt)
        out.write(frame)

        if fc in milestones:
            pct = milestones[fc]
            logger.info(f"Progress: {pct}%")
            if progress_callback:
                progress_callback(f"🎬 Rendering... {pct}%")
        fc += 1

    cap.release();
    out.release()

    try:
        with VideoFileClip(input_video) as orig_clip:
            with VideoFileClip(temp_silent) as rendered_clip:
                if hasattr(rendered_clip, 'with_audio'):
                    final = rendered_clip.with_audio(orig_clip.audio)
                else:
                    final = rendered_clip.set_audio(orig_clip.audio)

                final.write_videofile(output_video, codec="libx264", audio_codec="aac", fps=fps, preset="ultrafast",
                                      logger=None)
        return True
    except Exception as e:
        logger.error(f"Merge error: {e}")
        return False
    finally:
        if os.path.exists(temp_silent): os.remove(temp_silent)


async def transcribe_and_translate(video_path: str, update_func):
    loop = asyncio.get_running_loop()

    if USE_LOCAL_WHISPER:
        await update_func("⚙️ Local transcription (Fast)...")
        # Run local whisper in executor
        segs_gen, info = await loop.run_in_executor(None, lambda: whisper_model.transcribe(video_path))
        raw_segs, lang_raw = list(segs_gen), info.language
    else:
        await update_func("☁️ Cloud extraction...")
        audio_path = str(LOCAL_TMP_DIR / "audio.m4a")

        def ext():
            with VideoFileClip(video_path) as v:
                v.audio.write_audiofile(audio_path, codec='aac', logger=None)

        await loop.run_in_executor(None, ext)

        res = await loop.run_in_executor(None, lambda: openai_client.audio.transcriptions.create(
            model="whisper-1", file=open(audio_path, "rb"), response_format="verbose_json",
            timestamp_granularities=["segment"]
        ))
        raw_segs, lang_raw = res.segments, res.language

    if not raw_segs: return None

    full_text = " || ".join([s.text.strip() if hasattr(s, 'text') else s['text'].strip() for s in raw_segs])
    await update_func(f"✨ Translating ({TRANSLATION_MODEL})...")

    gpt = await loop.run_in_executor(None, lambda: openai_client.chat.completions.create(
        model=TRANSLATION_MODEL, messages=[
            {"role": "system", "content": "Translate to English. Keep ' || ' separators. Output ONLY translated text."},
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

    orig_text = " ".join([s.text if hasattr(s, 'text') else s['text'] for s in raw_segs])
    return orig_text, " ".join(trans_parts), timed, lang_raw


async def handle_video(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = await update.message.reply_text("📥 Downloading...")
    media = update.message.video or update.message.video_note
    fid = media.file_id
    t_in, t_out = str(LOCAL_TMP_DIR / f"i_{fid[:8]}.mp4"), str(LOCAL_TMP_DIR / f"o_{fid[:8]}.mp4")
    loop = asyncio.get_running_loop()

    try:
        f = await context.bot.get_file(fid)
        await f.download_to_drive(custom_path=t_in)

        def thread_safe_update(text):
            asyncio.run_coroutine_threadsafe(msg.edit_text(text), loop)

        async def async_update(text):
            try:
                await msg.edit_text(text)
            except:
                pass

        result = await transcribe_and_translate(t_in, async_update)
        if not result:
            await msg.edit_text("⚠️ No speech found.");
            return

        orig, trans, segments, lang_raw = result
        await async_update("🎬 Rendering subtitles... 0%")

        success = await loop.run_in_executor(
            None, create_subtitled_video, t_in, segments, t_out, thread_safe_update
        )

        if success:
            l_code = lang_raw.lower()[:2]
            flag, name = LANGUAGE_DATA.get(l_code, ('🌍', l_code.upper()))
            mode_tag = " [DEBUG]" if IS_LOCAL else ""
            cap = f"{flag} **Original:**\n{orig}\n\n🇺🇸 **English:**\n{trans}{mode_tag}"
            await msg.delete()
            await update.message.reply_video(video=open(t_out, 'rb'), caption=cap[:1024], parse_mode="Markdown")
        else:
            await msg.edit_text("⚠️ Render failed.")
    except Exception as e:
        logger.error(f"Error: {e}")
        try:
            await msg.edit_text(f"❌ Error: {str(e)}")
        except:
            pass
    finally:
        for p in [t_in, t_out]:
            if os.path.exists(p): os.remove(p)


if __name__ == '__main__':
    if not TELEGRAM_BOT_TOKEN:
        print("Error: TELEGRAM_BOT_TOKEN not set.")
        sys.exit(1)
    cleanup_temp_folder()
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", lambda u, c: u.message.reply_text("Send a video!")))
    app.add_handler(MessageHandler(filters.VIDEO | filters.VIDEO_NOTE, handle_video))
    logger.info(f"🚀 Bot started. Mode: {'DEBUG (Local)' if IS_LOCAL else 'PROD (VM)'}")
    app.run_polling()