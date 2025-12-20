import os
import textwrap

import numpy as np
from PIL import ImageFont, Image, ImageDraw
from moviepy import VideoFileClip, ImageClip, CompositeVideoClip
from moviepy.video.tools.subtitles import SubtitlesClip


FONT_SIZE = 50

def define_font():
    try:
        # Streamlit Cloud (Debian) usually has these paths
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if not os.path.exists(font_path):
            font_path = "arial.ttf"  # Fallback for local Windows
        font = ImageFont.truetype(font_path, FONT_SIZE)
    except:
        font = ImageFont.load_default()
    return font

FONT = define_font()


def create_text_image(text, video_width):

    # Wrap text to 80% of width
    chars_per_line = max(1, int((video_width * 0.8) / (FONT_SIZE * 0.5)))
    lines = textwrap.wrap(text, width=chars_per_line)
    wrapped_text = "\n".join(lines)

    # Calculate size
    temp_img = Image.new('RGBA', (video_width, 1000), (0, 0, 0, 0))
    draw = ImageDraw.Draw(temp_img)
    bbox = draw.multiline_textbbox((0, 0), wrapped_text, font=FONT, align="center")

    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    img = Image.new('RGBA', (video_width, h + 40), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Position
    pos = ((video_width - w) // 2, 10)

    # Simple outline for visibility
    for o in range(-2, 3):
        draw.multiline_text((pos[0] + o, pos[1]), wrapped_text, font=FONT, fill="black", align="center")
        draw.multiline_text((pos[0], pos[1] + o), wrapped_text, font=FONT, fill="black", align="center")

    draw.multiline_text(pos, wrapped_text, font=FONT, fill="white", align="center")
    return np.array(img)


def embed_subtitles(video_path, srt_path, output_path="output_video.mp4"):
    video = VideoFileClip(str(video_path))

    # Define the generator explicitly
    def make_text(txt):
        img_data = create_text_image(txt, video.w)
        return ImageClip(img_data, transparent=True)

    # CRITICAL: Use 'make_textclip=' as a keyword argument
    # This prevents MoviePy from misinterpreting the function as a font path
    subtitles = SubtitlesClip(str(srt_path), make_textclip=make_text)

    # This centers horizontally ('center') and places it 80% down the screen (0.8)
    subtitles = subtitles.with_position(('center', video.h * 0.6))

    final_video = CompositeVideoClip([video, subtitles])

    final_video.write_videofile(
        output_path,
        fps=video.fps,
        codec="libx264",
        audio_codec="aac",
        threads=4  # Faster encoding on cloud
    )

    video.close()
    return output_path


if __name__ == '__main__':
    # local run example
    embed_subtitles(video_path=r'/document_5920171835495816504.mp4',
                    srt_path=r'/document_5920171835495816504_subs.en.srt')
