import smtplib
from email.message import EmailMessage


def send_video_email(
        original_file_name: str,
        video_path: str,
        original_srt_path: str,
        translated_srt_path: str,
        recipient: str,
        smtp_user: str,
        smtp_pass: str
):
    msg = EmailMessage()
    msg["Subject"] = f"Translated video from {original_file_name}"
    msg["From"] = smtp_user
    msg["To"] = recipient
    msg.set_content("Attached is your video with translated subtitles.")

    with open(video_path, "rb") as f:
        msg.add_attachment(
            f.read(),
            maintype="video",
            subtype="mp4",
            filename=video_path
        )

    with open(original_srt_path, "rb") as f:
        msg.add_attachment(
            f.read(),
            maintype="text",
            subtype="srt",
            filename=original_srt_path
        )

    with open(translated_srt_path, "rb") as f:
        msg.add_attachment(
            f.read(),
            maintype="text",
            subtype="srt",
            filename=translated_srt_path
        )

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
        s.login(smtp_user, smtp_pass)
        s.send_message(msg)
