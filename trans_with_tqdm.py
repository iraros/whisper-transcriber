import pickle

import whisper
import os

from tqdm import tqdm

from logic import Transcriber, extract_audio, write_srt


class TranscriberTQDM(Transcriber):
    def transcribe_video(self, video_file_name):
        output_wav = os.path.splitext(video_file_name)[0] + ".wav"
        extract_audio(video_file_name, output_wav)
        result = self.transcribe_with_progress(output_wav)

        save_name = os.path.splitext(video_file_name)[0] + "_subs.srt"
        write_srt(result, save_name)
        return save_name

    def transcribe_with_progress(self, audio_file):
        """ Transcribe audio with a progress bar. """
        model = self.model
        output_wav = os.path.splitext(video_file_name)[0] + ".wav"
        if not os.path.exists(output_wav):
            extract_audio(video_file_name, output_wav)

        # Load audio and calculate duration
        audio = whisper.load_audio(output_wav)
        audio_duration = len(audio) / 16000  # Length of audio in seconds (assuming 16kHz sampling rate)

        chunk_duration = 120  # seconds
        chunks = int(audio_duration // chunk_duration) + (1 if audio_duration % chunk_duration > 0 else 0)
        print(f"Transcribing {audio_duration:.2f} seconds of audio in {chunks} chunks.")

        cache_dir = os.path.join(os.path.dirname(audio_file), f'transcriptions_{os.path.basename(audio_file)[:-4]}')
        os.makedirs(cache_dir, exist_ok=True)

        # Progress bar
        with tqdm(total=audio_duration, desc="Transcribing", unit="sec") as pbar:
            transcription_result = []
            for i in range(chunks):
                pkl_save_path = os.path.join(cache_dir, f"{i}.pkl")
                if os.path.exists(pkl_save_path):
                    with open(pkl_save_path, "rb") as file:
                        result = pickle.load(file)
                else:
                    # Process the chunk
                    start = i * chunk_duration
                    end = min((i + 1) * chunk_duration, audio_duration)
                    audio_chunk = audio[int(start * 16000):int(end * 16000)]  # Convert seconds to samples
                    result = model.transcribe(audio_chunk, word_timestamps=True)
                    with open(pkl_save_path, "wb") as f:
                        pickle.dump(result, f)

                transcription_result.append(result)

                # Update progress bar
                pbar.update(chunk_duration)

        # Combine all chunks
        final_transcription = {
            "segments": [segment for chunk in transcription_result for segment in chunk["segments"]],
        }
        return final_transcription

if __name__ == '__main__':
    t = TranscriberTQDM()

    video_file_name = r"C:\Users\Ira\Videos\Moana.2.2024.1080p.WEBRIP.H264.DD2.0.COLLECTiVE.mp4"
    t.transcribe_video(video_file_name)
