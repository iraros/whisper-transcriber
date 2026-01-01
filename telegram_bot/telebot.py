import os

# WORKAROUND for libiomp5md.dll error on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import logging
import asyncio
import tempfile
import sys
import subprocess
from pathlib import Path

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
# CONFIGURATION & MAPPING
# -----------------------------------------------------------------------------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_TELEGRAM_BOT_TOKEN_HERE")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY_HERE")

# Mapping language codes to (Flag, Full Name)
LANGUAGE_DATA = {
    'ar': ('🇸🇦', 'Arabic'),
    'he': ('🇮🇱', 'Hebrew'),
    'ru': ('🇷🇺', 'Russian'),
    'es': ('🇪🇸', 'Spanish'),
    'fr': ('🇫🇷', 'French'),
    'de': ('🇩🇪', 'German'),
    'it': ('🇮🇹', 'Italian'),
    'pt': ('🇵🇹', 'Portuguese'),
    'ja': ('🇯🇵', 'Japanese'),
    'ko': ('🇰🇷', 'Korean'),
    'zh': ('🇨🇳', 'Chinese'),
    'tr': ('🇹🇷', 'Turkish'),
    'nl': ('🇳🇱', 'Dutch'),
    'pl': ('🇵🇱', 'Polish'),
    'uk': ('🇺🇦', 'Ukrainian'),
    'hi': ('🇮🇳', 'Hindi'),
    'fa': ('🇮🇷', 'Persian'),
    'en': ('🇺🇸', 'English'),
    'vi': ('🇻🇳', 'Vietnamese'),
    'th': ('🇹🇭', 'Thai')
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

logger.info(f"Loading Whisper model: {WHISPER_MODEL_SIZE} on {DEVICE_TYPE}...")
try:
    whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device=DEVICE_TYPE, compute_type=COMPUTE_TYPE)
    logger.info("Whisper model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load Whisper model: {e}")
    whisper_model = None


# -----------------------------------------------------------------------------
# VIDEO & SUBTITLE PROCESSING
# -----------------------------------------------------------------------------
def create_subtitled_video(input_video: str, translation_text: str, output_video: str):
    # simple SRT block for the whole translation
    srt_content = f"1\n00:00:00,000 --> 00:00:59,000\n{translation_text}"

    with tempfile.NamedTemporaryFile(delete=False, suffix=".srt", mode='w', encoding='utf-8') as srt_file:
        srt_file.write(srt_content)
        srt_path = srt_file.name

    try:
        # FFmpeg filter paths on Windows are tricky because of the colon in drive letters.
        # We need to escape the backslashes and wrap the path correctly.
        clean_path = srt_path.replace("\\", "/")
        # For Windows, we escape the colon (e.g., C\: ) and wrap in single quotes
        escaped_srt_path = f"'{clean_path.replace(':', r'\:')}'"

        cmd = [
            'ffmpeg', '-y',
            '-i', input_video,
            '-vf', f'scale=-2:480,subtitles={escaped_srt_path}',
            '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '28',
            '-c:a', 'copy',
            output_video
        ]

        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg Error: {e.stderr.decode()}")
        return False
    finally:
        if os.path.exists(srt_path):
            os.remove(srt_path)


# -----------------------------------------------------------------------------
# TRANSCRIPTION & TRANSLATION LOGIC
# -----------------------------------------------------------------------------
def transcribe_locally(video_path: str):
    if whisper_model is None:
        raise RuntimeError("Whisper model failed to load at startup.")

    logger.info(f"Starting local transcription for: {video_path}")
    segments, info = whisper_model.transcribe(video_path, beam_size=5)

    transcribed_text = ""
    for segment in segments:
        transcribed_text += segment.text + " "

    return transcribed_text.strip(), info.language, info.language_probability


def translate_with_gpt(text: str, source_lang_name: str) -> str:
    logger.info("Sending text to OpenAI for translation...")

    system_prompt = (
        "You are an expert translator specializing in spoken dialects. "
        "Translate the following text into natural English. "
        f"Source language: {source_lang_name}. "
        "Keep the translation concise as it will be used for subtitles. "
        "Output ONLY the translated text."
    )

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"OpenAI Translation Error: {e}")
        return f"[Translation Error] {str(e)}"


# -----------------------------------------------------------------------------
# TELEGRAM HANDLERS
# -----------------------------------------------------------------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Hi! I'm your Subtitle Bot.\n\n"
        "Send me a video, and I'll transcribe it, translate it, "
        "and send you back a version with English subtitles!"
    )


async def handle_video(update: Update, context: ContextTypes.DEFAULT_TYPE):
    status_msg = await update.message.reply_text("📥 Downloading media...")

    if update.message.video:
        file_id = update.message.video.file_id
        ext = ".mp4"
    elif update.message.video_note:
        file_id = update.message.video_note.file_id
        ext = ".mp4"
    else:
        return

    temp_input = None
    temp_output = None

    try:
        new_file = await context.bot.get_file(file_id)
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as f:
            temp_input = f.name
        await new_file.download_to_drive(custom_path=temp_input)

        # Step 1: Transcribe
        await status_msg.edit_text("⚙️ Transcribing locally...")
        loop = asyncio.get_running_loop()
        transcript, lang_code, prob = await loop.run_in_executor(None, transcribe_locally, temp_input)

        if not transcript:
            await status_msg.edit_text("⚠️ No speech detected.")
            return

        # Get flag and full name from mapping
        source_flag, source_name = LANGUAGE_DATA.get(lang_code, ('🌍', lang_code.upper()))

        # Step 2: Translate
        await status_msg.edit_text(f"{source_flag} {source_name} ({prob:.2f}) -> Translating...")
        translation = await loop.run_in_executor(None, translate_with_gpt, transcript, source_name)

        # Step 3: Subtitle Video
        await status_msg.edit_text("🎬 Rendering subtitled video (480p)...")
        temp_output = temp_input.replace(ext, f"_subtitled{ext}")
        success = await loop.run_in_executor(None, create_subtitled_video, temp_input, translation, temp_output)

        # Step 4: Finalize
        # Combine everything into a single caption with full names
        caption_text = (
            f"{source_flag} **Original ({source_name}):**\n{transcript}\n\n"
            f"🇺🇸 **Translation (English):**\n{translation}"
        )

        if success:
            await status_msg.edit_text("✅ Done! Sending video...")
            if len(caption_text) > 1024:
                await update.message.reply_video(video=open(temp_output, 'rb'))
                for i in range(0, len(caption_text), 4000):
                    await update.message.reply_text(caption_text[i:i + 4000], parse_mode="Markdown")
            else:
                await update.message.reply_video(
                    video=open(temp_output, 'rb'),
                    caption=caption_text,
                    parse_mode="Markdown"
                )
        else:
            # Fallback for render failure
            await status_msg.edit_text(f"⚠️ Render failed. Sending text only:")
            for i in range(0, len(caption_text), 4000):
                await update.message.reply_text(caption_text[i:i + 4000], parse_mode="Markdown")

    except Exception as e:
        logger.error(f"Error: {e}")
        await status_msg.edit_text(f"❌ Error: {str(e)}")

    finally:
        for path in [temp_input, temp_output]:
            if path and os.path.exists(path):
                os.remove(path)


# -----------------------------------------------------------------------------
# MAIN APP
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    if "YOUR_TELEGRAM_BOT_TOKEN_HERE" in TELEGRAM_BOT_TOKEN:
        print("CRITICAL: Set your TELEGRAM_BOT_TOKEN.")
        sys.exit(1)

    app = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.VIDEO | filters.VIDEO_NOTE, handle_video))

    print("--- Bot is now running ---")
    app.run_polling()