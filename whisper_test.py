# add implementation for unique prompting rather than fixed seconds
import numpy as np
import sounddevice as sd
import torch
from transformers import pipeline
import warnings
import re

warnings.filterwarnings("ignore", category=FutureWarning)

whisper = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3", torch_dtype=torch.float32, device="cpu")

sample_rate = 16000
record_duration = 5 #seconds

def record_audio():
    print("Recording... Please speak.")
    audio = sd.rec(int(record_duration * sample_rate), samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()
    print("Recording complete.")
    return audio.flatten()

def extract_object(text):
    match = re.search(r'help me find my (.+)', text.lower())
    if match:
        return match.group(1).strip()
    return None

def transcribe_audio(audio_data):
    print("Transcribing audio...")
    transcription = whisper({"array": audio_data, "sampling_rate": sample_rate, "language": "en"})
    transcribed_text = transcription["text"]
    print("You said:", transcribed_text)

    object_name = extract_object(transcribed_text)
    if object_name:
        print("Object to detect:", object_name)
    else:
        print("No object found in the transcription.")

audio_data = record_audio()
transcribe_audio(audio_data)