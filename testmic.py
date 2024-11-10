import sounddevice as sd

# List all available audio devices
devices = sd.query_devices()
print("Available audio devices:")
for i, device in enumerate(devices):
    print(f"{i}: {device['name']} - {'Input' if device['max_input_channels'] > 0 else 'Output'}")


sample_rate = 44100  # CD-quality audio
duration = 5  # seconds

# Record audio from the microphone
print("Recording...")
audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, device=0, dtype='float32')
sd.wait()  # Wait until the recording is finished
print("Recording complete.")

# Play back the recorded audio
print("Playing back the audio...")
sd.play(audio_data, samplerate=sample_rate)
sd.wait()  # Wait until playback is done
print("Playback complete.")
