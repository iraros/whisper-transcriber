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

# Environment Detection - Strictly based on IS_LOCAL env var
IS_LOCAL = os.getenv("IS_LOCAL", "false").lower() == "true"

BASE_DIR = Path(__file__).parent.absolute()
LOCAL_TMP_DIR = BASE_DIR / "tmp"
LOCAL_TMP_DIR.mkdir(parents=True, exist_ok=True)

# Pricing Constants
WHISPER_USD_PER_MIN = 0.006
USD_TO_ILS = 3.7
AGOROT_PER_ILS = 100

# Translation chunking — max segments per GPT call (prevents token limit failures on long videos)
TRANSLATION_CHUNK_SIZE = 25

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

print("\n" + "=" * 40)
print("       ENVIRONMENT DETECTION")
print("=" * 40)
print(f" IS_LOCAL:     {IS_LOCAL}")
print(f" LOG FILE:     {log_file}")
print("=" * 40 + "\n")

if IS_LOCAL:
    TRANSLATION_MODEL = "gpt-3.5-turbo"
    WHISPER_LABEL = "Local Faster-Whisper (Base)"
    USE_LOCAL_WHISPER = True
    try:
        from faster_whisper import WhisperModel

        whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
        print(f"[DEBUG] DEBUG MODE: {WHISPER_LABEL} + {TRANSLATION_MODEL}")
    except ImportError:
        USE_LOCAL_WHISPER = False
else:
    TRANSLATION_MODEL = os.getenv("TRANSLATION_MODEL", "gpt-4o-mini")
    WHISPER_LABEL = "OpenAI Whisper-1 (Cloud)"
    USE_LOCAL_WHISPER = False
    print(f"🚀 PROD MODE: {WHISPER_LABEL} + {TRANSLATION_MODEL}")

# Comprehensive Language Map for Flags & Full Names
LANGUAGE_DATA = {
    'ar': ('🇸🇦', 'Arabic'), 'he': ('🇮🇱', 'Hebrew'), 'ru': ('🇷🇺', 'Russian'),
    'es': ('🇪🇸', 'Spanish'), 'fr': ('🇫🇷', 'French'), 'de': ('🇩🇪', 'German'),
    'it': ('🇮🇹', 'Italian'), 'pt': ('🇵🇹', 'Portuguese'), 'ja': ('🇯🇵', 'Japanese'),
    'ko': ('🇰🇷', 'Korean'), 'zh': ('🇨🇳', 'Chinese'), 'tr': ('🇹🇷', 'Turkish'),
    'nl': ('🇳🇱', 'Dutch'), 'pl': ('🇵🇱', 'Polish'), 'uk': ('🇺🇦', 'Ukrainian'),
    'hi': ('🇮🇳', 'Hindi'), 'fa': ('🇮🇷', 'Persian'), 'en': ('🇺🇸', 'English'),
    'vi': ('🇻🇳', 'Vietnamese'), 'th': ('🇹🇭', 'Thai'), 'id': ('🇮🇩', 'Indonesian'),
    'el': ('🇬🇷', 'Greek'), 'cs': ('🇨🇿', 'Czech'), 'hu': ('🇭🇺', 'Hungarian')
}

openai_client = OpenAI(api_key=OPENAI_API_KEY)


def format_text_for_telegram(parts: list[str]) -> str:
    """Group translated segments into readable paragraphs for Telegram display."""
    sentence_endings = ('.', '!', '?', '...', '؟', '。', '！', '？')
    paragraphs = []
    group = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        group.append(part)
        if part.endswith(sentence_endings) or len(group) >= 4:
            paragraphs.append(' '.join(group))
            group = []
    if group:
        paragraphs.append(' '.join(group))
    return '\n\n'.join(paragraphs)


def split_long_segments(segments: list, max_sentences: int = 2) -> list:
    """Split subtitle segments that contain more than max_sentences into smaller ones.
    Time is distributed proportionally by character count."""
    import re
    result = []
    for seg in segments:
        text = seg['text'].strip()
        # Split on sentence-ending punctuation followed by whitespace or end of string
        sentences = re.split(r'(?<=[.!?؟。！？])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) <= max_sentences:
            result.append(seg)
            continue

        duration = seg['end'] - seg['start']
        total_chars = sum(len(s) for s in sentences) or 1
        current_time = seg['start']

        for i in range(0, len(sentences), max_sentences):
            group = sentences[i:i + max_sentences]
            group_text = ' '.join(group)
            group_chars = sum(len(s) for s in group)
            group_duration = duration * (group_chars / total_chars)
            result.append({
                'start': current_time,
                'end': current_time + group_duration,
                'text': group_text
            })
            current_time += group_duration

    return result


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
    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_w, orig_h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    target_w = 480
    if orig_w > target_w:
        ratio = target_w / float(orig_w)
        w, h = target_w, int(orig_h * ratio)
    else:
        w, h = orig_w, orig_h

    temp_silent = str(LOCAL_TMP_DIR / "silent_tmp.mp4")
    out = cv2.VideoWriter(temp_silent, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    # Track percentages for logging
    last_logged_pct = -1

    fc = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        if w != orig_w: frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        t = fc / fps
        txt = next((s['text'] for s in segments if s['start'] <= t <= s['end']), None)
        if txt: draw_text_with_outline(frame, txt)
        out.write(frame)

        # Percentage calculation for both UI and Local Logs
        pct = int((fc / total_frames) * 100) if total_frames > 0 else 0
        if pct % 5 == 0 and pct != last_logged_pct:
            logger.info(f"Rendering Progress: {pct}%")
            if progress_callback: progress_callback(f"🎬 Rendering... {pct}%")
            last_logged_pct = pct

        fc += 1
    cap.release()
    out.release()

    logger.info(">>> STAGE: Visual Rendering Finished")
    if progress_callback: progress_callback("✅ Finished rendering frames.")

    logger.info(">>> STAGE: Audio Merging Started")
    if progress_callback: progress_callback("🎵 Merging with original sound...")

    try:
        with VideoFileClip(input_video) as orig_clip:
            with VideoFileClip(temp_silent) as rendered_clip:
                final = rendered_clip.set_audio(orig_clip.audio) if not hasattr(rendered_clip,
                                                                                'with_audio') else rendered_clip.with_audio(
                    orig_clip.audio)
                final.write_videofile(output_video, codec="libx264", audio_codec="aac", fps=fps, preset="ultrafast",
                                      logger=None)
        logger.info(">>> STAGE: Audio Merging Finished")
        return True
    except Exception as e:
        logger.error(f"Merge error: {e}")
        return False
    finally:
        if os.path.exists(temp_silent): os.remove(temp_silent)


async def transcribe_and_translate(video_path: str, duration: float, update_func):
    loop = asyncio.get_running_loop()

    # Calculate video length and cost
    duration_str = f"{int(duration // 60)}m {int(duration % 60)}s"
    cost_usd = (duration / 60.0) * WHISPER_USD_PER_MIN
    cost_agorot = cost_usd * USD_TO_ILS * AGOROT_PER_ILS

    if USE_LOCAL_WHISPER:
        cost_str = "Free (Local)"
    else:
        cost_str = f"~{cost_agorot:.2f} Agorot"

    status_header = f"⏱ Video: {duration_str}\n💰 Cost: {cost_str}\n\n"

    if USE_LOCAL_WHISPER:
        logger.info(f">>> STAGE: Transcription Started ({WHISPER_LABEL})")
        await update_func(f"{status_header}⚙️ Transcribing with {WHISPER_LABEL}...")
        segs_gen, info = await loop.run_in_executor(None, lambda: whisper_model.transcribe(video_path))
        raw_segs, lang_raw = list(segs_gen), info.language
    else:
        logger.info(f">>> STAGE: Transcription Started ({WHISPER_LABEL})")
        await update_func(f"{status_header}☁️ Transcribing with {WHISPER_LABEL}...")
        audio_path = str(LOCAL_TMP_DIR / "audio.m4a")

        def ext():
            with VideoFileClip(video_path) as v: v.audio.write_audiofile(audio_path, codec='aac', logger=None)

        await loop.run_in_executor(None, ext)
        res = await loop.run_in_executor(None, lambda: openai_client.audio.transcriptions.create(
            model="whisper-1", file=open(audio_path, "rb"), response_format="verbose_json",
            timestamp_granularities=["segment"]
        ))
        raw_segs, lang_raw = res.segments, res.language

    logger.info(f">>> STAGE: Transcription Finished (Lang: {lang_raw})")

    if not raw_segs: return None

    # Get language details for progress message
    l_code = lang_raw.lower()[:2]
    flag, name = LANGUAGE_DATA.get(l_code, ('🌍', lang_raw.upper()))

    logger.info(f">>> STAGE: Translation Started ({TRANSLATION_MODEL})")
    total_chunks = (len(raw_segs) + TRANSLATION_CHUNK_SIZE - 1) // TRANSLATION_CHUNK_SIZE
    trans_parts = []

    for chunk_idx in range(0, len(raw_segs), TRANSLATION_CHUNK_SIZE):
        chunk = raw_segs[chunk_idx:chunk_idx + TRANSLATION_CHUNK_SIZE]
        chunk_num = chunk_idx // TRANSLATION_CHUNK_SIZE + 1
        await update_func(
            f"{status_header}✨ Translating {flag} {name} → EN ({chunk_num}/{total_chunks})..."
        )
        chunk_text = " || ".join(
            [s.text.strip() if hasattr(s, 'text') else s['text'].strip() for s in chunk]
        )
        gpt = await loop.run_in_executor(None, lambda ct=chunk_text: openai_client.chat.completions.create(
            model=TRANSLATION_MODEL, messages=[
                {"role": "system", "content": (
                    "Translate to English. Preserve ALL ' || ' separators exactly. "
                    "Output the same number of segments as the input."
                )},
                {"role": "user", "content": ct}
            ]
        ))
        parts = [p.strip() for p in gpt.choices[0].message.content.split("||")]
        # Guard against GPT returning fewer parts than expected
        while len(parts) < len(chunk):
            parts.append(parts[-1] if parts else "")
        trans_parts.extend(parts[:len(chunk)])

    logger.info(">>> STAGE: Translation Finished")

    timed = []
    for i, s in enumerate(raw_segs):
        st = s.start if hasattr(s, 'start') else s['start']
        en = s.end if hasattr(s, 'end') else s['end']
        txt = trans_parts[i] if i < len(trans_parts) else (s.text if hasattr(s, 'text') else s['text'])
        timed.append({'start': st, 'end': en, 'text': txt})

    orig_text = " ".join([s.text if hasattr(s, 'text') else s['text'] for s in raw_segs])
    trans_formatted = format_text_for_telegram(trans_parts)
    timed = split_long_segments(timed, max_sentences=2)
    return orig_text, trans_formatted, timed, lang_raw


async def handle_video(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    logger.info("-" * 20)
    logger.info(f"New Request from User ID: {user_id}")

    msg = await update.message.reply_text("📥 Downloading...")
    media = update.message.video or update.message.video_note
    fid = media.file_id
    t_in, t_out = str(LOCAL_TMP_DIR / f"i_{fid[:8]}.mp4"), str(LOCAL_TMP_DIR / f"o_{fid[:8]}.mp4")
    loop = asyncio.get_running_loop()

    try:
        logger.info(">>> STAGE: Downloading File")
        f = await context.bot.get_file(fid)
        await f.download_to_drive(custom_path=t_in)

        cap_info = cv2.VideoCapture(t_in)
        fps = cap_info.get(cv2.CAP_PROP_FPS) or 30.0
        duration = cap_info.get(cv2.CAP_PROP_FRAME_COUNT) / fps
        cap_info.release()

        logger.info(f"Video Info: {duration:.2f}s, {fps} FPS")

        def thread_safe_update(text):
            asyncio.run_coroutine_threadsafe(msg.edit_text(text), loop)

        result = await transcribe_and_translate(t_in, duration, lambda t: msg.edit_text(t))
        if not result:
            await msg.edit_text("⚠️ No speech found.")
            return

        orig, trans, segments, lang_raw = result
        await msg.edit_text("🎬 Rendering... 0%")
        success = await loop.run_in_executor(None, create_subtitled_video, t_in, segments, t_out, thread_safe_update)

        if success:
            logger.info(f">>> STAGE: Sending Finished Video")
            l_code = lang_raw.lower()[:2]
            flag, name = LANGUAGE_DATA.get(l_code, ('🌍', lang_raw.upper()))

            mode_tag = "\n[DEBUG]" if IS_LOCAL else ""
            full_text = f"{flag} *{name}:*\n\n{orig}\n\n🇺🇸 *English:*\n\n{trans}{mode_tag}"

            # Send video with a short caption; text goes in a separate message to avoid truncation
            short_cap = f"{flag} {name} → 🇺🇸 English (subtitled)"
            await update.message.reply_video(
                video=open(t_out, 'rb'),
                caption=short_cap,
                read_timeout=600,
                write_timeout=600
            )

            # Send full formatted transcript — split into chunks if over Telegram's 4096-char limit
            MAX_MSG_LEN = 4096
            for i in range(0, len(full_text), MAX_MSG_LEN):
                await update.message.reply_text(
                    full_text[i:i + MAX_MSG_LEN],
                    parse_mode=ParseMode.MARKDOWN
                )

            await msg.delete()
            logger.info(">>> STAGE: COMPLETE - SUCCESS")
        else:
            await msg.edit_text("⚠️ Render failed.")
            logger.error(">>> STAGE: COMPLETE - FAILED AT RENDER")
    except Exception as e:
        logger.error(f">>> STAGE: ERROR: {e}", exc_info=True)
        try:
            await msg.edit_text(f"❌ Error: {str(e)}")
        except:
            pass
    finally:
        for p in [t_in, t_out]:
            if os.path.exists(p): os.remove(p)
        logger.info("-" * 20)


if __name__ == '__main__':
    if not TELEGRAM_BOT_TOKEN:
        print("Error: TELEGRAM_BOT_TOKEN not set.")
        sys.exit(1)
    cleanup_temp_folder()

    # Using the higher timeout settings from your latest file
    request = HTTPXRequest(connect_timeout=60.0, read_timeout=600.0, write_timeout=600.0)
    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).request(request).build()

    app.add_handler(CommandHandler("start", lambda u, c: u.message.reply_text("Send a video!")))
    app.add_handler(MessageHandler(filters.VIDEO | filters.VIDEO_NOTE, handle_video))

    logger.info("🚀 Bot started with Stage Logging and Cost Tracking.")
    app.run_polling()