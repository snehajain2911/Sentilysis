import sounddevice as sd
import numpy as np
import time


def calculate_loudness(audio_data):
    rms = np.sqrt(np.mean(audio_data**2))
    loudness_db = 20 * np.log10(rms)
    print(loudness_db)


def audio_callback(indata, frame, rate, status):
    if status:
        print(f"Error in audio input: {status}")
    if any(indata):
        loudness = calculate_loudness(indata)
        print(f"Loudness: {100+loudness:.2f} dB ")
        time.sleep(0.5)


# Set the sampling parameters
sample_rate = 44100  # You can adjust this based on your preferences
duration = 10  # Recording duration in seconds

# Record audio from the microphone
with sd.InputStream(callback=audio_callback, channels=1, samplerate=sample_rate):
    print(f"Recording... Please speak.")
    sd.sleep(int(duration * 1000))

print("Recording complete.")
