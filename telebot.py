import os
import logging
import asyncio
import sys
import textwrap
import cv2
import shutil
from pathlib import Path

# WORKAROUND for libiomp5md.dll error on some Windows environments
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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
    print("\n" + "!" * 60 + "\nCRITICAL: 'python-telegram-bot' not found.\n" + "!" * 60 + "\n")
    sys.exit(1)

try:
    from faster_whisper import WhisperModel
    from openai import OpenAI
except ImportError as e:
    print(f"\nCRITICAL: Missing dependency {e.name}. Run: pip install faster-whisper openai")
    sys.exit(1)

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_TELEGRAM_BOT_TOKEN_HERE")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY_HERE")
TRANSLATION_MODEL = os.getenv("TRANSLATION_MODEL", "gpt-4o-mini")

USE_CLOUD_WHISPER = True

BASE_DIR = Path(__file__).parent.absolute()
LOCAL_TMP_DIR = BASE_DIR / "tmp"
LOCAL_TMP_DIR.mkdir(parents=True, exist_ok=True)

# Dictionary keyed by ISO 639-1 language codes
LANGUAGE_DATA = {
    'ar': ('🇸🇦', 'Arabic'), 'he': ('🇮🇱', 'Hebrew'), 'ru': ('🇷🇺', 'Russian'),
    'es': ('🇪🇸', 'Spanish'), 'fr': ('🇫🇷', 'French'), 'de': ('🇩🇪', 'German'),
    'it': ('🇮🇹', 'Italian'), 'pt': ('🇵🇹', 'Portuguese'), 'ja': ('🇯🇵', 'Japanese'),
    'ko': ('🇰🇷', 'Korean'), 'zh': ('🇨🇳', 'Chinese'), 'tr': ('🇹🇷', 'Turkish'),
    'nl': ('🇳🇱', 'Dutch'), 'pl': ('🇵🇱', 'Polish'), 'uk': ('🇺🇦', 'Ukrainian'),
    'hi': ('🇮🇳', 'Hindi'), 'fa': ('🇮🇷', 'Persian'), 'en': ('🇺🇸', 'English'),
    'vi': ('🇻🇳', 'Vietnamese'), 'th': ('🇹🇭', 'Thai')
}

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

openai_client = OpenAI(api_key=OPENAI_API_KEY)
whisper_model = None if USE_CLOUD_WHISPER else WhisperModel("medium", device="cpu", compute_type="int8")


# -----------------------------------------------------------------------------
# UTILS
# -----------------------------------------------------------------------------
def cleanup_temp_folder():
    for file_path in LOCAL_TMP_DIR.glob('*'):
        try:
            if file_path.is_file():
                file_path.unlink()
            elif file_path.is_dir():
                shutil.rmtree(file_path)
        except Exception as e:
            logger.error(f'Cleanup error: {e}')


def draw_text_with_outline(img, text):
    h, w, _ = img.shape
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 1.1
    thickness = 2
    wrapped_lines = textwrap.wrap(text, width=max(15, int(w / 28)))
    line_height = int(45 * font_scale)
    start_y = h - (len(wrapped_lines) * line_height) - 80

    for i, line in enumerate(wrapped_lines):
        text_size = cv2.getTextSize(line, font, font_scale, thickness)[0]
        tx, ty = (w - text_size[0]) // 2, start_y + (i * line_height) + text_size[1]
        cv2.putText(img, line, (tx, ty), font, font_scale, (0, 0, 0), thickness + 4, cv2.LINE_AA)
        cv2.putText(img, line, (tx, ty), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


def create_subtitled_video(input_video: str, segments: list, output_video: str):
    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps < 1: fps = 30.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
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
                final.write_videofile(output_video, codec="libx264", audio_codec="aac", logger=None)
        return True
    except Exception as e:
        logger.error(f"Merge error: {e}")
        return False
    finally:
        if os.path.exists(temp_silent): os.remove(temp_silent)


async def transcribe_and_translate(video_path: str, update_func):
    loop = asyncio.get_running_loop()
    if USE_CLOUD_WHISPER:
        await update_func("☁️ Extracting audio...")
        audio_path = str(LOCAL_TMP_DIR / "audio.m4a")

        def ext():
            with VideoFileClip(video_path) as v:
                v.audio.write_audiofile(audio_path, codec='aac', logger=None)

        await loop.run_in_executor(None, ext)

        res = await loop.run_in_executor(None, lambda: openai_client.audio.transcriptions.create(
            model="whisper-1", file=open(audio_path, "rb"), response_format="verbose_json",
            timestamp_granularities=["segment"]
        ))
        raw_segs, lang = res.segments, res.language
    else:
        await update_func("⚙️ Local transcription...")
        segs_gen, info = whisper_model.transcribe(video_path)
        raw_segs, lang = list(segs_gen), info.language

    if not raw_segs: return None

    full_text = " || ".join([s.text.strip() if hasattr(s, 'text') else s['text'].strip() for s in raw_segs])
    await update_func(f"✨ Translating from {lang.upper()}...")

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

    return " ".join([s.text if hasattr(s, 'text') else s['text'] for s in raw_segs]), " ".join(trans_parts), timed, lang


async def handle_video(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = await update.message.reply_text("📥 Downloading...")
    media = update.message.video or update.message.video_note
    fid = media.file_id
    t_in, t_out = str(LOCAL_TMP_DIR / f"i_{fid[:8]}.mp4"), str(LOCAL_TMP_DIR / f"o_{fid[:8]}.mp4")

    try:
        f = await context.bot.get_file(fid)
        await f.download_to_drive(custom_path=t_in)

        async def up(t):
            try:
                await msg.edit_text(t)
            except:
                pass

        result = await transcribe_and_translate(t_in, up)
        if not result:
            await msg.edit_text("⚠️ No speech found.");
            return

        orig, trans, segments, l_code = result
        await up("🎬 Rendering subtitles...")
        success = await asyncio.get_running_loop().run_in_executor(None, create_subtitled_video, t_in, segments, t_out)

        if success:
            # Normalize l_code to lower case to match dictionary keys
            clean_code = l_code.lower()[:2]
            flag, name = LANGUAGE_DATA.get(clean_code, ('🌍', clean_code.upper()))

            cap = f"{flag} **Original:**\n{orig}\n\n🇺🇸 **English:**\n{trans}"
            await msg.delete()
            await update.message.reply_video(video=open(t_out, 'rb'), caption=cap[:1024], parse_mode="Markdown")
        else:
            await msg.edit_text("⚠️ Render failed.")
    except Exception as e:
        await msg.edit_text(f"❌ Error: {str(e)}")
    finally:
        for p in [t_in, t_out]:
            if os.path.exists(p): os.remove(p)


if __name__ == '__main__':
    cleanup_temp_folder()
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", lambda u, c: u.message.reply_text("Send a video!")))
    app.add_handler(MessageHandler(filters.VIDEO | filters.VIDEO_NOTE, handle_video))
    print("🚀 Bot is starting...")
    app.run_polling()