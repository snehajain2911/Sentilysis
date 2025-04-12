import pyaudio
import librosa
import numpy as np


def calculate_tempo(audio_data, sample_rate):
    # Use librosa to calculate tempo
    onset_env = librosa.onset.onset_strength(y=audio_data, sr=sample_rate)
    tempo, _ = librosa.beat.beat_track(
        onset_envelope=onset_env, sr=sample_rate)
    print(tempo)


def record_audio(duration=5, sample_rate=44100):
    # Record audio from microphone using PyAudio
    p = pyaudio.PyAudio()

    stream = p.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=sample_rate,
                    input=True,
                    frames_per_buffer=1024)

    print("Recording...")

    frames = []
    for i in range(0, int(sample_rate / 1024 * duration)):
        data = stream.read(1024)
        frames.append(np.frombuffer(data, dtype=np.float32))

    print("Finished recording")

    stream.stop_stream()
    stream.close()
    p.terminate()

    audio_data = np.concatenate(frames, axis=0)
    return audio_data, sample_rate


# if __name__ == "__main__":
#     # Record audio from the microphone
#     audio_data, sample_rate = record_audio()

#     # Calculate and print the tempo of speech
#     tempo = calculate_tempo(audio_data, sample_rate)
#     print(f"The tempo of the speech is {tempo:.2f} beats per minute.")
