import pysrt
import streamlit as st
from openai import OpenAI
from tqdm import tqdm

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


def translate_srt_with_gpt(original_srt_path: str, target_lang="English") -> str:
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
    english_srt_path = original_srt_path.replace('.srt', '.en.srt')
    translated_subs.save(english_srt_path, encoding="utf-8")
    return english_srt_path
