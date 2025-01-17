import pickle

import whisper
import os
import pydub
from datetime import timedelta


def extract_audio(video_file, output_wav):
    """ Extracts audio from a .mov or .mp4 file using plain Python."""
    audio = pydub.AudioSegment.from_file(video_file)
    audio = audio.set_channels(1).set_frame_rate(32000)  # Convert to mono, 32kHz
    audio.export(output_wav, format="wav")

def split_line_balanced(words):
    lens = [len(word) for word in words]
    for i in range(1, len(words)):
        current_diff = abs(sum(lens[:i]) - sum(lens[i:]))
        previous_diff = abs(sum(lens[:i - 1]) - sum(lens[i - 1:]))
        if current_diff >= previous_diff:
            return ' '.join(words[:i - 1]), ' '.join(words[i - 1:])
    return ' '.join(words), ''

def get_next_word_start(phrases, phrase_index, word_count):
    current_phrase = phrases[phrase_index]
    if word_count < len(current_phrase) - 1:
        next_word = current_phrase[word_count]
    elif phrase_index < len(phrases) - 1:
        next_word = phrases[phrase_index + 1][0]
    else:
        last_word = phrases[-1][-1]
        _, last_word_end = get_word_times(last_word)
        return last_word_end
    next_word_start, _ = get_word_times(next_word)
    return next_word_start

def write_srt(transcription, save_name="output.srt", max_chars=45, max_line_length=25):
    srt_content = ""
    phrases = get_phrases(transcription)

    index = 1
    sub_text = ""
    start_time = None

    for phrase_index, phrase in enumerate(phrases):
        word_phrase_count = 0
        for word_index, word in enumerate(phrase):
            word_text = word["word"]

            if not start_time:
                start_time, _ = get_word_times(word)

            if len(sub_text) + len(word_text) + 1 > max_chars or word_index == len(phrase) - 1:
                if word_index == len(phrase) - 1:
                    sub_text += word_text + " "
                    word_phrase_count += 1

                words_in_sub = sub_text.strip().split()
                if len(sub_text) > max_line_length:
                    formatted_text = split_line_balanced(words_in_sub)
                else:
                    formatted_text = [sub_text.strip()]
                next_word_start = get_next_word_start(phrases, phrase_index, word_phrase_count)

                formatted_text_str = '\n'.join(formatted_text)  # Create the formatted text as a separate string
                segment_addition = f"{index}\n{start_time} --> {next_word_start}\n{formatted_text_str}\n\n"
                srt_content += segment_addition
                index += 1
                sub_text = ""
                start_time = next_word_start  # Update start time for the next subtitle
                if word_index == len(phrase) - 1:
                    continue

            sub_text += word_text + " "
            word_phrase_count += 1

    # Save the transcript as an SRT file
    with open(save_name, "w", encoding="utf-8") as file:
        file.write(srt_content)

    print(f"SRT file saved as {save_name}")


def get_word_times(word):
    word_start = str(timedelta(seconds=word["start"])).split(".")[0] + "," + str(int(word["start"] * 1000) % 1000).zfill(3)
    word_end = str(timedelta(seconds=word["end"])).split(".")[0] + "," + str(int(word["end"] * 1000) % 1000).zfill(3)
    return word_start, word_end


def get_phrases(transcription):
    words = [word for segment in transcription["segments"] for word in segment.get("words", [])]
    phrases = []
    current_phrase = []
    for i in range(len(words)):
        word = words[i]
        word_text = word['word'].strip().strip('-')
        if word_text and word_text[-1] in ",.?!":
            word['word'] = word_text.strip(",.")
            current_phrase.append(word)
            phrases.append(current_phrase)
            current_phrase = []
        else:
            word['word'] = word_text
            current_phrase.append(word)
    if current_phrase:
        phrases.append(current_phrase)
    return phrases


class Transcriber:
    def __init__(self, model_name="small"):
        self.model = whisper.load_model(model_name)

    def transcribe_video(self, video_file_name):
        pkl_path = os.path.splitext(video_file_name)[0] + "_subs.pkl"
        if os.path.exists(pkl_path):
            with open(pkl_path, "rb") as file:
                result = pickle.load(file)
        else:
            output_wav = os.path.splitext(video_file_name)[0] + ".wav"
            extract_audio(video_file_name, output_wav)
            result = self.model.transcribe(output_wav, word_timestamps=True)
            with open(pkl_path, "wb") as file:
                pickle.dump(result, file)
            os.remove(output_wav)

        save_name = os.path.splitext(video_file_name)[0] + "_subs.srt"
        write_srt(result, save_name)
        return save_name


if __name__ == '__main__':
    t = Transcriber()

    video_file_name = r"C:\Users\Ira\Videos\Timeline 1.mp4"
    t.transcribe_video(video_file_name)
