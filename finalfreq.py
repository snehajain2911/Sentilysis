# import pyaudio
# import numpy as np


# def get_audio_input(sample_rate=44100, chunk_size=2048):
#     p = pyaudio.PyAudio()

#     stream = p.open(format=pyaudio.paInt16,
#                     channels=1,
#                     rate=sample_rate,
#                     input=True,
#                     frames_per_buffer=chunk_size)

#     return p, stream


# def calculate_frequency(data, sample_rate):
#     fft_result = np.fft.fft(data)
#     frequency = np.fft.fftfreq(len(fft_result), 1 / sample_rate)
#     magnitude = np.abs(fft_result)

#     # Find the peak frequency
#     peak_index = np.argmax(magnitude)
#     peak_frequency = abs(frequency[peak_index])

#     return peak_frequency


# def main():
#     sample_rate = 44100
#     chunk_size = 2048

#     p, stream = get_audio_input(sample_rate, chunk_size)

#     try:
#         print("Frequency Checker App - Press Ctrl+C to exit")
#         while True:
#             audio_data = np.frombuffer(stream.read(chunk_size), dtype=np.int16)
#             frequency = calculate_frequency(audio_data, sample_rate)
#             print(f"Current Frequency: {frequency:.2f} Hz", end='\r')

#     except KeyboardInterrupt:
#         print("\nExiting the program.")

#     finally:
#         stream.stop_stream()
#         stream.close()
#         p.terminate()


# if __name__ == "__main__":
#     main()

import pyaudio
import numpy as np
from scipy.signal import butter, lfilter
import time


def get_audio_input(sample_rate=44100, chunk_size=2048):
    p = pyaudio.PyAudio()

    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=sample_rate,
                    input=True,
                    frames_per_buffer=chunk_size)

    return p, stream


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def calculate_frequency(data, sample_rate):
    fft_result = np.fft.fft(data)
    frequency = np.fft.fftfreq(len(fft_result), 1 / sample_rate)
    magnitude = np.abs(fft_result)

    # Find the peak frequency
    peak_index = np.argmax(magnitude)
    peak_frequency = abs(frequency[peak_index])

    return peak_frequency


def calculate_median(count):
    median = 0
    if (count % 2 == 0):
        median = int(((count/2)+(count/2+1))/2)
    else:
        median = int((count+1)/2)
    return median


def main():
    # sample_rate = 44100
    # chunk_size = 2048
    sample_rate = 44100
    chunk_size = 2048
    lowcut = 85  # Adjust these values based on the expected human voice frequency range
    highcut = 255
    count = 0
    freqledge = []
    midfreq = 0
    threshold = 1000
    # sum

    p, stream = get_audio_input(sample_rate, chunk_size)

    try:
        print("Frequency Checker App - Press Ctrl+C to exit")
        while True:
            # count = count+1
            audio_data = np.frombuffer(stream.read(chunk_size), dtype=np.int16)

            # Apply bandpass filter to focus on human voice frequencies
            filtered_audio = butter_bandpass_filter(
                audio_data, lowcut, highcut, sample_rate)

            frequency = calculate_frequency(filtered_audio, sample_rate)
            # print(f"Current Frequency: {frequency:.2f} Hz", end='\r')
            # if (85 < frequency < 255) :
            # Check if the magnitude of the frequency is above the threshold
            if np.max(np.abs(filtered_audio)) > threshold:
                if (frequency > 85 and frequency < 255):
                    count = count + 1
                    freqledge.append(frequency)
                    if (count != 0):
                        midfreq = freqledge[int(calculate_median(count))-1]
                if (freqledge[0] < midfreq and midfreq < frequency):
                    print("uptone")
                elif (freqledge[0] < midfreq and midfreq > frequency):
                    print("mid uptone")
                elif (freqledge[0] > midfreq and midfreq < frequency):
                    print("mid down tone")
                elif (freqledge[0] > midfreq and midfreq > frequency):
                    print("downtone")
                else:
                    print("neutral")
                    # print(f"Current Frequency: {frequency:.2f} Hz and Mid: {midfreq: .2f} Hz")
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nExiting the program.")

    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()


if __name__ == "__main__":
    main()
