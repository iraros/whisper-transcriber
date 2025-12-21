import pickle

from modules.transcriber import get_word_times

pkl_save_path = r''

with open(pkl_save_path, "rb") as file:
    result = pickle.load(file)

for segment in result["segments"]:
    start, end = get_word_times(segment)
    print(start, segment["text"])
