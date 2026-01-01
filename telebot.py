import os
import logging
import asyncio
import sys
import textwrap
import cv2
import shutil
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
TRANSLATION_MODEL = os.getenv("TRANSLATION_MODEL", "gpt-4o-mini")

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

# Standard logging setup - this will go to nohup.out
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
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
    font_scale = 1.0
    thickness = 2
    wrapped_lines = textwrap.wrap(text, width=max(15, int(w / 25)))
    line_height = int(40 * font_scale)
    start_y = h - (len(wrapped_lines) * line_height) - 60

    for i, line in enumerate(wrapped_lines):
        text_size = cv2.getTextSize(line, font, font_scale, thickness)[0]
        tx, ty = (w - text_size[0]) // 2, start_y + (i * line_height) + text_size[1]
        cv2.putText(img, line, (tx, ty), font, font_scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
        cv2.putText(img, line, (tx, ty), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


def create_subtitled_video(input_video: str, segments: list, output_video: str, progress_callback=None):
    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    temp_silent = str(LOCAL_TMP_DIR / "silent_tmp.mp4")
    out = cv2.VideoWriter(temp_silent, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    # Milestones for every 5%
    milestones = {int(total_frames * (i / 20)): i * 5 for i in range(1, 20)}

    logger.info(f"Starting render for {input_video} ({total_frames} frames)")
    fc = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        t = fc / fps
        txt = next((s['text'] for s in segments if s['start'] <= t <= s['end']), None)
        if txt: draw_text_with_outline(frame, txt)
        out.write(frame)

        if fc in milestones:
            pct = milestones[fc]
            status_text = f"🎬 Rendering video... {pct}%"
            # Log to server console/nohup.out
            logger.info(f"Progress: {pct}% (Frame {fc}/{total_frames})")
            # Update Telegram user via callback
            if progress_callback:
                progress_callback(status_text)

        fc += 1

    cap.release();
    out.release()
    logger.info("Visual rendering complete. Merging audio...")
    progress_callback("Visual rendering complete. Merging audio...")

    try:
        with VideoFileClip(input_video) as orig_clip:
            with VideoFileClip(temp_silent) as rendered_clip:
                # API compatibility check for MoviePy v1 and v2
                if hasattr(rendered_clip, 'with_audio'):
                    # MoviePy v2.x
                    final = rendered_clip.with_audio(orig_clip.audio)
                else:
                    # MoviePy v1.x
                    final = rendered_clip.set_audio(orig_clip.audio)

                final.write_videofile(output_video, codec="libx264", audio_codec="aac", logger=None)
        logger.info(f"Final video saved: {output_video}")
        return True
    except Exception as e:
        logger.error(f"Merge error: {e}")
        return False
    finally:
        if os.path.exists(temp_silent): os.remove(temp_silent)


async def transcribe_and_translate(video_path: str, update_func):
    loop = asyncio.get_running_loop()
    await update_func("☁️ Processing Audio...")
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
    await update_func(f"✨ Translating from {lang_raw.upper()}...")
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

    return " ".join([s.text if hasattr(s, 'text') else s['text'] for s in raw_segs]), " ".join(
        trans_parts), timed, lang_raw


async def handle_video(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = await update.message.reply_text("📥 Downloading...")
    media = update.message.video or update.message.video_note
    fid = media.file_id
    t_in, t_out = str(LOCAL_TMP_DIR / f"i_{fid[:8]}.mp4"), str(LOCAL_TMP_DIR / f"o_{fid[:8]}.mp4")

    # Capture the current event loop to use for thread-safe communication
    loop = asyncio.get_running_loop()

    try:
        f = await context.bot.get_file(fid)
        await f.download_to_drive(custom_path=t_in)
        logger.info(f"Video downloaded for user {update.effective_user.id}")

        # Thread-safe update function for use inside the background executor
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
            cap = f"{flag} **Original:**\n{orig}\n\n🇺🇸 **English:**\n{trans}"
            await msg.delete()
            await update.message.reply_video(video=open(t_out, 'rb'), caption=cap[:1024], parse_mode="Markdown")
            logger.info(f"Video sent successfully to user {update.effective_user.id}")
        else:
            await msg.edit_text("⚠️ Render failed.")
    except Exception as e:
        logger.error(f"Handle video error: {str(e)}")
        try:
            await msg.edit_text(f"❌ Error: {str(e)}")
        except:
            pass
    finally:
        for p in [t_in, t_out]:
            if os.path.exists(p):
                try:
                    os.remove(p)
                except:
                    pass


if __name__ == '__main__':
    if not TELEGRAM_BOT_TOKEN:
        print("Error: TELEGRAM_BOT_TOKEN not set.")
        sys.exit(1)
    cleanup_temp_folder()
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", lambda u, c: u.message.reply_text("Send a video!")))
    app.add_handler(MessageHandler(filters.VIDEO | filters.VIDEO_NOTE, handle_video))
    logger.info("🚀 Bot started on GCP. Logging to console enabled.")
    app.run_polling()