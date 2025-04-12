# from nltk.sentiment.vader import SentimentIntensityAnalyzer
# import speech_recognition as sr
# from flask import Flask, render_template, request
# import nltk
# nltk.download('vader_lexicon')


# app = Flask(__name__)


# # Initialize the sentiment analyzer
# sia = SentimentIntensityAnalyzer()


# def get_voice_input():
#     recognizer = sr.Recognizer()

#     with sr.Microphone() as source:
#         print("Say something...")
#         recognizer.adjust_for_ambient_noise(source)
#         # audio = recognizer.listen(source)
#         audio = recognizer.listen(source, phrase_time_limit=None)

#     try:
#         text = recognizer.recognize_google(audio)
#         return text
#     except sr.UnknownValueError:
#         return None
#     except sr.RequestError as e:
#         print(f"Could not request results: {e}")
#         return None


# def analyze_sentiment(text):
#     sentiment_scores = sia.polarity_scores(text)
#     compound_score = sentiment_scores['compound']

#     if compound_score >= 0.05:
#         return "Positive"
#     elif compound_score <= -0.05:
#         return "Negative"
#     else:
#         return "Neutral"


# @app.route('/')
# def index():
#     return render_template('index.html')


# @app.route('/analyze', methods=['POST'])
# def analyze():
#     voice_input = get_voice_input()

#     if voice_input:
#         sentiment = analyze_sentiment(voice_input)
#         return render_template('result.html', sentiment=sentiment, voice_input=voice_input)
#     else:
#         return render_template('result.html', sentiment="No voice input detected", voice_input="")


# if __name__ == '__main__':
#     app.run(debug=True)

# import loud
# import frequency
# import tempo
# import speech_recognition as sr
# import numpy as np
# import librosa


# def calculate_tempo(audio_data, sample_rate):
#     # Use librosa to calculate tempo
#     onset_env = librosa.onset.onset_strength(y=audio_data, sr=sample_rate)
#     tempo, _ = librosa.beat.beat_track(
#         onset_envelope=onset_env, sr=sample_rate)
#     return tempo


# def calculate_loudness(audio_data):
#     rms = np.sqrt(np.mean(audio_data**2))
#     loudness_db = 20 * np.log10(rms)
#     return loudness_db


# recognizer = sr.Recognizer()
# with sr.Microphone() as source:
#     print("Say something...")
#     # Adjust for ambient noise before starting
#     recognizer.adjust_for_ambient_noise(source)
#     # Continuous listening loop
#     while True:
#         try:
#             # Capture audio in real-time
#             audio = recognizer.listen(source, phrase_time_limit=None)
#             # Convert the audio data to a NumPy array
#             audio_data = np.frombuffer(audio.frame_data, dtype=np.int16)
#             # Convert integer audio data to floating-point
#             audio_data_float = audio_data.astype(
#                 np.float32) / np.iinfo(np.int16).max
#             audio_data = np.concatenate(frames, axis=0)
#             # Transcribe the audio
#             text = recognizer.recognize_google(audio)
#             tempo = calculate_tempo(audio_data_float, sample_rate=44100)
#             loud = calculate_loudness(audio)
#             frequency.calculate_frequency(audio)
#             # chat.append(f"User: {text}")
#             # Print the real-time transcription
#             print(text)
#             print("___________________________________________________")
#             # return text
#         except sr.UnknownValueError:
#             # Handle cases where no speech is detected
#             print("None")
#             # return None
#         except sr.RequestError as e:
#             # Handle errors with the transcription service
#             print(f"Could not request results: {e}")
#             break
