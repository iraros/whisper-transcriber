import os

# WORKAROUND for libiomp5md.dll error on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import logging
import asyncio
import sys
import subprocess
import textwrap
from pathlib import Path

# -----------------------------------------------------------------------------
# DEPENDENCY CHECK
# -----------------------------------------------------------------------------
try:
    from telegram import Update
    from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters
except ImportError:
    print("\n" + "!" * 60)
    print("CRITICAL IMPORT ERROR: 'python-telegram-bot' not found.")
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
# CONFIGURATION & LOCAL PATH SETUP
# -----------------------------------------------------------------------------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_TELEGRAM_BOT_TOKEN_HERE")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY_HERE")

REPO_DIR = Path(__file__).parent.absolute()
LOCAL_TMP_DIR = REPO_DIR / "tmp"
LOCAL_TMP_DIR.mkdir(exist_ok=True)

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
    logger.info("Whisper model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load Whisper model: {e}")
    whisper_model = None


# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------
def format_timestamp(seconds: float) -> str:
    """Converts seconds to SRT timestamp format: HH:MM:SS,mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def split_line_balanced(words: list) -> list:
    """
    Splits a list of words into two strings such that the length
    difference between the two lines is minimized.
    """
    lens = [len(word) for word in words]
    for i in range(1, len(words)):
        current_diff = abs(sum(lens[:i]) - sum(lens[i:]))
        previous_diff = abs(sum(lens[:i - 1]) - sum(lens[i - 1:]))
        if current_diff >= previous_diff:
            line1 = ' '.join(words[:i - 1])
            line2 = ' '.join(words[i - 1:])
            return [line1, line2] if line2 else [line1]

    return [' '.join(words)]


# -----------------------------------------------------------------------------
# VIDEO & SUBTITLE PROCESSING
# -----------------------------------------------------------------------------
def create_subtitled_video(input_video_path: str, segments: list, output_video_path: str, srt_debug_path: str):
    """
    Burns timed subtitles into pixels using 'drawtext'.
    Also saves a valid .srt file for local debug.
    """
    # 1. Save SRT file for local debug
    try:
        with open(srt_debug_path, "w", encoding="utf-8") as f:
            for i, seg in enumerate(segments):
                start_srt = format_timestamp(seg['start'])
                end_srt = format_timestamp(seg['end'])
                f.write(f"{i + 1}\n{start_srt} --> {end_srt}\n{seg['text']}\n\n")
        logger.info(f"Debug SRT saved to: {srt_debug_path}")
    except Exception as e:
        logger.error(f"Failed to save debug SRT: {e}")

    # 2. Prepare FFmpeg Drawtext Filter
    filter_chains = []
    MAX_LINE_LENGTH = 25  # Threshold to trigger balanced splitting

    for seg in segments:
        words = seg['text'].split()

        # Use balanced splitting logic if text is long enough
        if len(seg['text']) > MAX_LINE_LENGTH:
            lines = split_line_balanced(words)
            wrapped_text = "\n".join(lines)
        else:
            wrapped_text = seg['text']

        safe_text = wrapped_text.replace("'", "'\\''").replace(":", "\\:")

        draw_filter = (
            f"drawtext=text='{safe_text}':fontcolor=white:fontsize=32:"
            f"box=1:boxcolor=black@0.6:boxborderw=10:"
            f"line_spacing=5:x=(w-text_w)/2:y=h-th-40:"
            f"enable='between(t,{seg['start']:.3f},{seg['end']:.3f})'"
        )
        filter_chains.append(draw_filter)

    full_vf = ",".join(filter_chains)

    try:
        cmd = [
            'ffmpeg', '-y',
            '-i', input_video_path,
            '-vf', full_vf,
            '-c:v', 'libx264',
            '-preset', 'veryfast',
            '-crf', '26',
            '-c:a', 'copy',
            output_video_path
        ]

        logger.info(f"Running FFmpeg render...")
        process = subprocess.run(cmd, capture_output=True, text=True)

        if process.returncode != 0:
            logger.error(f"FFmpeg failed: {process.stderr}")
            return False
        return True
    except Exception as e:
        logger.error(f"Exception during subtitle rendering: {str(e)}")
        return False


# -----------------------------------------------------------------------------
# TRANSCRIPTION & TRANSLATION LOGIC
# -----------------------------------------------------------------------------
def transcribe_and_translate_segments(video_path: str):
    if whisper_model is None:
        raise RuntimeError("Whisper model failed to load.")

    logger.info(f"Starting timed transcription: {video_path}")
    segments_gen, info = whisper_model.transcribe(video_path, beam_size=5)

    raw_segments = list(segments_gen)
    if not raw_segments:
        return [], info.language, 0, "", ""

    full_text = " || ".join([s.text.strip() for s in raw_segments])
    _, source_name = LANGUAGE_DATA.get(info.language, ('🌍', info.language.upper()))

    system_prompt = (
        "You are an expert movie subtitle translator. Translate the text into natural English. "
        "The input segments are separated by ' || '. Maintain the same number of segments in your output, "
        "separated by ' || '. Output ONLY the translated segments."
    )

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": full_text}
            ],
            temperature=0.3
        )
        translated_blob = response.choices[0].message.content
        translated_list = [t.strip() for t in translated_blob.split("||")]
    except Exception as e:
        logger.error(f"OpenAI Error: {e}")
        translated_list = [s.text for s in raw_segments]

    final_segments = []
    full_transcript_parts = []
    full_translation_parts = []

    for i, orig in enumerate(raw_segments):
        text = translated_list[i] if i < len(translated_list) else orig.text
        final_segments.append({
            'start': orig.start,
            'end': orig.end,
            'text': text
        })
        full_transcript_parts.append(orig.text.strip())
        full_translation_parts.append(text.strip())

    return final_segments, info.language, info.language_probability, " ".join(full_transcript_parts), " ".join(
        full_translation_parts)


# -----------------------------------------------------------------------------
# TELEGRAM HANDLERS
# -----------------------------------------------------------------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("👋 Hi! Send me a video for movie-style English subtitles!")


async def handle_video(update: Update, context: ContextTypes.DEFAULT_TYPE):
    status_msg = await update.message.reply_text("📥 Downloading media...")

    file_id = None
    if update.message.video:
        file_id = update.message.video.file_id
    elif update.message.video_note:
        file_id = update.message.video_note.file_id

    if not file_id: return

    # Persistence for local debug
    temp_input = str(LOCAL_TMP_DIR / f"in_{file_id}.mp4")
    temp_output = str(LOCAL_TMP_DIR / f"out_{file_id}.mp4")
    temp_srt = str(LOCAL_TMP_DIR / f"sub_{file_id}.srt")

    try:
        new_file = await context.bot.get_file(file_id)
        await new_file.download_to_drive(custom_path=temp_input)

        await status_msg.edit_text("⚙️ Transcribing & Syncing timestamps...")
        loop = asyncio.get_running_loop()
        timed_segments, lang_code, prob, orig_text, trans_text = await loop.run_in_executor(None,
                                                                                            transcribe_and_translate_segments,
                                                                                            temp_input)

        if not timed_segments:
            await status_msg.edit_text("⚠️ No speech detected in the video.")
            return

        source_flag, source_name = LANGUAGE_DATA.get(lang_code, ('🌍', lang_code.upper()))

        await status_msg.edit_text(
            f"{source_flag} Detected {source_name} ({prob:.2%})\n🎬 Rendering movie-style subtitles...")
        success = await loop.run_in_executor(None, create_subtitled_video, temp_input, timed_segments, temp_output,
                                             temp_srt)

        caption = (
            f"{source_flag} **Original ({source_name}):**\n{orig_text[:400]}...\n\n"
            f"🇺🇸 **English Translation:**\n{trans_text[:400]}..."
        )

        if success:
            await status_msg.edit_text("✅ Done! Sending video...")
            await update.message.reply_video(
                video=open(temp_output, 'rb'),
                caption=caption[:1024],
                parse_mode="Markdown"
            )
        else:
            await status_msg.edit_text("⚠️ Render failed. Check logs.")

    except Exception as e:
        logger.error(f"Error: {e}")
        await status_msg.edit_text(f"❌ Error: {str(e)}")
    finally:
        # All files (mp4 and srt) are kept in /tmp/ for debugging
        pass


if __name__ == '__main__':
    if "YOUR_TELEGRAM_BOT_TOKEN_HERE" in TELEGRAM_BOT_TOKEN:
        print("CRITICAL: Set your TELEGRAM_BOT_TOKEN.")
        sys.exit(1)

    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.VIDEO | filters.VIDEO_NOTE, handle_video))
    print("--- Bot is running ---")
    app.run_polling()