import ffmpeg
import pyaudio

audio_url = "http://10.25.255.189:8080/audio.wav"

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=2, rate=44100, output=True)

process = (
    ffmpeg
    .input(audio_url)
    .output('pipe:', format='wav')
    .run_async(pipe_stdout=True, pipe_stderr=True, quiet=True)
)

try:
    while True:
        in_bytes = process.stdout.read(2048)
        if not in_bytes:
            break
        stream.write(in_bytes)
except KeyboardInterrupt:
    print("Audio streaming stopped.")

stream.stop_stream()
stream.close()
p.terminate()
process.terminate()
