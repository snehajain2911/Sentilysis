import time
import queue
import re
import sys
# from google.cloud import speech
import pyaudio
import speech_recognition as sr
from flask import Flask, render_template, request
from google.cloud import speech_v1p1beta1 as speech
from google.cloud import language_v1
import os
import openai
from twilio.twiml.voice_response import Gather, VoiceResponse

API_KEY = open("C:/Users/dipta/Desktop/Diptam/Projects/Professional-Project/Smart-India-Hackathon/Sentilysis/SENTI CODE/gaurav/API_KEY", "r").read()
openai.api_key = API_KEY

app = Flask(__name__)

chat = []


# Replace the path with actual path of the .json file to set up Google Cloud credentials
# Enable Speech-to-Text and Natural Language APIs in Google Cloud Console.
os.environ[
    "GOOGLE_APPLICATION_CREDENTIALS"
] = "C:/Users/dipta/Desktop/Diptam/Projects/Professional-Project/Smart-India-Hackathon/Sentilysis/SENTI CODE/secrets/exalted-legacy-407013-3b4aaf95b683.json"


# Add a global variable to track the time of the last speech input
last_speech_time = time.time()

# Audio recording parameters
RATE = 16000
CHUNK = int(RATE / 10)  # 100ms


class MicrophoneStream:
    """Opens a recording stream as a generator yielding the audio chunks."""

    def __init__(self: object, rate: int = RATE, chunk: int = CHUNK) -> None:
        """The audio -- and generator -- is guaranteed to be on the main thread."""
        self._rate = rate
        self._chunk = chunk

        # Create a thread-safe buffer of audio data
        self._buff = queue.Queue()
        self.closed = True

    def __enter__(self: object) -> object:
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            # The API currently only supports 1-channel (mono) audio
            # https://goo.gl/z757pE
            channels=1,
            rate=self._rate,
            input=True,
            frames_per_buffer=self._chunk,
            # Run the audio stream asynchronously to fill the buffer object.
            # This is necessary so that the input device's buffer doesn't
            # overflow while the calling thread makes network requests, etc.
            stream_callback=self._fill_buffer,
        )

        self.closed = False

        return self

    def __exit__(
        self: object,
        type: object,
        value: object,
        traceback: object,
    ) -> None:
        """Closes the stream, regardless of whether the connection was lost or not."""
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        # Signal the generator to terminate so that the client's
        # streaming_recognize method will not block the process termination.
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(
        self: object,
        in_data: object,
        frame_count: int,
        time_info: object,
        status_flags: object,
    ) -> object:
        """Continuously collect data from the audio stream, into the buffer.

        Args:
            in_data: The audio data as a bytes object
            frame_count: The number of frames captured
            time_info: The time information
            status_flags: The status flags

        Returns:
            The audio data as a bytes object
        """
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self: object) -> object:
        """Generates audio chunks from the stream of audio data in chunks.

        Args:
            self: The MicrophoneStream object

        Returns:
            A generator that outputs audio chunks.
        """
        while not self.closed:
            # Use a blocking get() to ensure there's at least one chunk of
            # data, and stop iteration if the chunk is None, indicating the
            # end of the audio stream.
            chunk = self._buff.get()
            if chunk is None:
                return
            data = [chunk]

            # Now consume whatever other data's still buffered.
            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break

            yield b"".join(data)


def listen_print_loop(responses: object) -> str:
    """Iterates through server responses and prints them.

    The responses passed is a generator that will block until a response
    is provided by the server.

    Each response may contain multiple results, and each result may contain
    multiple alternatives; for details, see https://goo.gl/tjCPAU.  Here we
    print only the transcription for the top alternative of the top result.

    In this case, responses are provided for interim results as well. If the
    response is an interim one, print a line feed at the end of it, to allow
    the next result to overwrite it, until the response is a final one. For the
    final one, print a newline to preserve the finalized transcription.

    Args:
        responses: List of server responses

    Returns:
        The transcribed text.
    """
    num_chars_printed = 0
    for response in responses:
        if not response.results:
            continue

        # The `results` list is consecutive. For streaming, we only care about
        # the first result being considered, since once it's `is_final`, it
        # moves on to considering the next utterance.
        result = response.results[0]
        if not result.alternatives:
            continue

        # Display the transcription of the top alternative.
        transcript = result.alternatives[0].transcript

        # Display interim results, but with a carriage return at the end of the
        # line, so subsequent lines will overwrite them.
        #
        # If the previous result was longer than this one, we need to print
        # some extra spaces to overwrite the previous result
        overwrite_chars = " " * (num_chars_printed - len(transcript))

        if not result.is_final:
            sys.stdout.write(transcript + overwrite_chars + "\r")
            sys.stdout.flush()

            num_chars_printed = len(transcript)

        else:
            print(transcript + overwrite_chars)

            # Exit recognition if any of the transcribed phrases could be
            # one of our keywords.
            if re.search(r"\b(exit|quit)\b", transcript, re.I):
                print("Exiting..")
                break

            num_chars_printed = 0

        return transcript

# GCP Audio Transcribe with Binary Audio Data


def transcribe_audio(audio_content):
    client = speech.SpeechClient()

    audio = speech.RecognitionAudio(content=audio_content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en-US",
    )

    response = client.recognize(config=config, audio=audio)

    return response.results[0].alternatives[0].transcript if response.results else None


# GCP Sentiment Analysis with Text
def analyze_sentiment_google(text_content):
    client = language_v1.LanguageServiceClient()

    document = language_v1.Document(
        content=text_content, type_=language_v1.Document.Type.PLAIN_TEXT
    )
    sentiment = client.analyze_sentiment(
        request={"document": document}
    ).document_sentiment

    return sentiment.score, sentiment.magnitude


def get_voice_input():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Say something...")
        # Adjust for ambient noise before starting
        recognizer.adjust_for_ambient_noise(source)
        # Continuous listening loop
        while True:
            try:
                # Capture audio in real-time
                audio = recognizer.listen(source, phrase_time_limit=None)
                # Transcribe the audio
                text = recognizer.recognize_google(audio)
                chat.append(f"User: {text}")
                # Print the real-time transcription
                print(text)
                print("___________________________________________________")
                return text
            except sr.UnknownValueError:
                # Handle cases where no speech is detected
                return None
            except sr.RequestError as e:
                # Handle errors with the transcription service
                print(f"Could not request results: {e}")
                break


def analyze_sentiment(text):
    sentiment_score, sentiment_magnitude = analyze_sentiment_google(text)

    if sentiment_score >= 0.05:
        chat.append("(Positive)")
        return "Positive"
    elif sentiment_score <= -0.05:
        chat.append("(negetive)")
        return "Negative"
    else:
        chat.append("(neutral)")
        return "Neutral"


def get_response(user_input):
    # Read information from the text file
    with open('C:/Users/dipta/Desktop/Diptam/Projects/Professional-Project/Smart-India-Hackathon/Sentilysis/SENTI CODE/gaurav/demo_text.txt', 'r') as file:
        file_content = file.read()

    # Check if the text file contains relevant information
    if file_content.strip():
        # Use information from the text file to generate a response
        response = generate_openai_response(file_content, user_input)
    else:
        # Fallback to a general response from OpenAI
        response = generate_general_openai_response(user_input)

    return response


def generate_openai_response(context, user_input):
    # Limit the context to fit within the model's maximum context length
    context = context[-4096:]

    # Use OpenAI to generate a response based on the context and user input
    prompt = f"{context}\nUser: {user_input}\nAI:"
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        temperature=0.7,
        max_tokens=150,
        n=1,
        stop=None,
    )

    return response['choices'][0]['text']


def generate_general_openai_response(user_input):
    # Fallback to a general response from OpenAI
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=f"User: {user_input}\nAI:",
        temperature=0.7,
        max_tokens=150,
        n=1,
        stop=None,
    )

    return response['choices'][0]['text']


@app.route("/")
def index():
    return render_template("index.html")


# @app.route("/analyze", methods=["POST"])
# def analyze():
#     language_code = "en-US"
#     client = speech.SpeechClient()
#     config = speech.RecognitionConfig(
#         encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
#         sample_rate_hertz=RATE,
#         language_code=language_code,
#     )

#     streaming_config = speech.StreamingRecognitionConfig(
#         config=config, interim_results=True
#     )

#     with MicrophoneStream(RATE, CHUNK) as stream:
#         audio_generator = stream.generator()
#         requests = (
#             speech.StreamingRecognizeRequest(audio_content=content)
#             for content in audio_generator
#         )

#         responses = client.streaming_recognize(streaming_config, requests)

#         transcript = listen_print_loop(responses)

#     if transcript:
#         sentiment = analyze_sentiment(transcript)
#         user_input = f"{transcript}."
#         response = get_response(user_input)
#         chat.append(f"User: {transcript}")
#         chat.append(f"AI: {response}")
#         return render_template("result.html", sentiment=sentiment, voice_input=chat)
#     else:
#         return render_template("result.html", sentiment="No voice input detected", voice_input="")


@app.route("/analyze", methods=["POST"])
def analyze():
    language_code = "en-US"
    client = speech.SpeechClient()
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code=language_code,
    )

    streaming_config = speech.StreamingRecognitionConfig(
        config=config, interim_results=False
    )

    with MicrophoneStream(RATE, CHUNK) as stream:
        audio_generator = stream.generator()
        requests = (
            speech.StreamingRecognizeRequest(audio_content=content)
            for content in audio_generator
        )

        responses = client.streaming_recognize(streaming_config, requests)

        transcript = listen_print_loop(responses)

    if transcript:
        global last_speech_time
        sentiment = analyze_sentiment(transcript)
        user_input = f"{transcript}."
        response = get_response(user_input)
        chat.append(f"User: {transcript}")
        chat.append(f"AI: {response}")

        # Reset the last speech time when speech is detected
        last_speech_time = time.time()

        return render_template("result.html", sentiment=sentiment, voice_input=chat)
    else:
        # Check if there's a long pause in speech (e.g., 5 seconds)
        current_time = time.time()
        if current_time - last_speech_time > 5:
            # Perform speech-to-text with the recorded audio
            sentiment = analyze_sentiment(' '.join(chat))
            user_input = ' '.join(chat) + "."
            response = get_response(user_input)
            chat.append(f"AI: {response}")

            # Reset the last speech time after processing the pause
            last_speech_time = current_time

            return render_template("result.html", sentiment=sentiment, voice_input=chat)
        else:
            return render_template("result.html", sentiment="No voice input detected", voice_input="")


# @app.route("/voice", methods=["POST"])
# def voice():
#     response = VoiceResponse()

#     # Set timeout and speechTimeout values accordingly
#     with response.gather(action="/analyze_twilio", method="POST", timeout=60, speechTimeout=10) as gather:
#         # gather.say("Please start speaking after the beep.")
#         gather.say(
#             "Welcome to sentilysis. We are working on it. Please mail us at sentilysis@gmail.com for further details.")
#     # If no input is received, redirect to /analyze_twilio immediately
#     response.redirect("/analyze_twilio")

#     return str(response)


# @app.route("/analyze_twilio", methods=["POST"])
# def analyze_twilio():
#     # Extract speech content from Twilio request
#     voice_input = request.form["SpeechResult"]

#     if voice_input:
#         sentiment = analyze_sentiment(voice_input)

#         # Get the response based on the input and text file content
#         response = get_response(voice_input)
#         chat.append(f"AI: {response}")

#         return render_template("result.html", sentiment=sentiment, voice_input=chat)
#     else:
#         return render_template("result.html", sentiment="No voice input detected", voice_input="")

# @app.route("/analyze_twilio", methods=["POST"])
# def analyze_twilio():
#     print("Request Headers:", request.headers)  # Print the request headers
#     print("Request Form Data:", request.form)  # Print the form data
#     # Extract speech content from Twilio request query parameters
#     voice_input = request.args.get("SpeechResult")
#     print("Voice Input:", voice_input)  # Print the extracted voice input

#     if voice_input:
#         sentiment = analyze_sentiment(voice_input)

#         # Get the response based on the input and text file content
#         response = get_response(voice_input)
#         chat.append(f"AI: {response}")

#         return render_template("result.html", sentiment=sentiment, voice_input=chat)
#         # return render_template("result.html", sentiment=sentiment, voice_input=voice_input)
#     else:
#         return render_template("result.html", sentiment="No voice input detected", voice_input="")

# test


if __name__ == "__main__":
    app.run(debug=True)


# import speech_recognition as sr
# from flask import Flask, render_template, request
# from google.cloud import speech_v1p1beta1 as speech
# from google.cloud import language_v1
# import os
# import openai
# from twilio.twiml.voice_response import Gather, VoiceResponse

# API_KEY = open("C:/Users/dipta/Desktop/Diptam/Projects/Professional-Project/Smart-India-Hackathon/Sentilysis/SENTI CODE/gaurav/API_KEY", "r").read()
# openai.api_key = API_KEY

# app = Flask(__name__)

# chat = []


# # Replace the path with actual path of the .json file to set up Google Cloud credentials
# # Enable Speech-to-Text and Natural Language APIs in Google Cloud Console.
# os.environ[
#     "GOOGLE_APPLICATION_CREDENTIALS"
# ] = "C:/Users/dipta/Desktop/Diptam/Projects/Professional-Project/Smart-India-Hackathon/Sentilysis/SENTI CODE/secrets/exalted-legacy-407013-3b4aaf95b683.json"

# # GCP Audio Transcribe with Binary Audio Data


# def transcribe_audio(audio_content):
#     client = speech.SpeechClient()

#     audio = speech.RecognitionAudio(content=audio_content)
#     config = speech.RecognitionConfig(
#         encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
#         sample_rate_hertz=16000,
#         language_code="en-US",
#     )

#     response = client.recognize(config=config, audio=audio)

#     return response.results[0].alternatives[0].transcript if response.results else None


# # GCP Sentiment Analysis with Text
# def analyze_sentiment_google(text_content):
#     client = language_v1.LanguageServiceClient()

#     document = language_v1.Document(
#         content=text_content, type_=language_v1.Document.Type.PLAIN_TEXT
#     )
#     sentiment = client.analyze_sentiment(
#         request={"document": document}
#     ).document_sentiment

#     return sentiment.score, sentiment.magnitude


# # def get_voice_input():
# #     recognizer = sr.Recognizer()

# #     with sr.Microphone() as source:
# #         print("Say something...")
# #         recognizer.adjust_for_ambient_noise(source)
# #         # audio = recognizer.listen(source)
# #         audio = recognizer.listen(source, phrase_time_limit=None)

# #     try:
# #         text = recognizer.recognize_google(audio)
# #         return text
# #     except sr.UnknownValueError:
# #         return None
# #     except sr.RequestError as e:
# #         print(f"Could not request results: {e}")
# #         return None


# def get_voice_input():
#     recognizer = sr.Recognizer()
#     with sr.Microphone() as source:
#         print("Say something...")
#         # Adjust for ambient noise before starting
#         recognizer.adjust_for_ambient_noise(source)
#         # Continuous listening loop
#         while True:
#             try:
#                 # Capture audio in real-time
#                 audio = recognizer.listen(source, phrase_time_limit=None)
#                 # Transcribe the audio
#                 text = recognizer.recognize_google(audio)
#                 chat.append(f"User: {text}")
#                 # Print the real-time transcription
#                 print(text)
#                 print("___________________________________________________")
#                 return text
#             except sr.UnknownValueError:
#                 # Handle cases where no speech is detected
#                 return None
#             except sr.RequestError as e:
#                 # Handle errors with the transcription service
#                 print(f"Could not request results: {e}")
#                 break


# def analyze_sentiment(text):
#     sentiment_score, sentiment_magnitude = analyze_sentiment_google(text)

#     if sentiment_score >= 0.05:
#         chat.append("(Positive)")
#         return "Positive"
#     elif sentiment_score <= -0.05:
#         chat.append("(negetive)")
#         return "Negative"
#     else:
#         chat.append("(neutral)")
#         return "Neutral"


# def get_response(user_input):
#     # Read information from the text file
#     with open('C:/Users/dipta/Desktop/Diptam/Projects/Professional-Project/Smart-India-Hackathon/Sentilysis/SENTI CODE/gaurav/demo_text.txt', 'r') as file:
#         file_content = file.read()

#     # Check if the text file contains relevant information
#     if file_content.strip():
#         # Use information from the text file to generate a response
#         response = generate_openai_response(file_content, user_input)
#     else:
#         # Fallback to a general response from OpenAI
#         response = generate_general_openai_response(user_input)

#     return response


# def generate_openai_response(context, user_input):
#     # Limit the context to fit within the model's maximum context length
#     context = context[-4096:]

#     # Use OpenAI to generate a response based on the context and user input
#     prompt = f"{context}\nUser: {user_input}\nAI:"
#     response = openai.Completion.create(
#         engine="text-davinci-002",
#         prompt=prompt,
#         temperature=0.7,
#         max_tokens=150,
#         n=1,
#         stop=None,
#     )

#     return response['choices'][0]['text']


# def generate_general_openai_response(user_input):
#     # Fallback to a general response from OpenAI
#     response = openai.Completion.create(
#         engine="text-davinci-002",
#         prompt=f"User: {user_input}\nAI:",
#         temperature=0.7,
#         max_tokens=150,
#         n=1,
#         stop=None,
#     )

#     return response['choices'][0]['text']


# @app.route("/")
# def index():
#     return render_template("index.html")


# @app.route("/analyze", methods=["POST"])
# def analyze():
#     voice_input = get_voice_input()
#     if voice_input:
#         sentiment = analyze_sentiment(voice_input)

#         # Get user input
#         user_input = f"{voice_input}."

#         # Get the response based on the input and text file content
#         response = get_response(user_input)
#         chat.append(f"AI: {response}")
#         # Display the generated response
#         # print(f"AI: {response}")
#         print("__________________________________________________")
#         print(chat)
#         print("__________________________________________________")
#         return render_template(
#             "result.html", sentiment=sentiment, voice_input=chat
#         )
#     else:
#         return render_template(
#             "result.html", sentiment="No voice input detected", voice_input=""
#         )


# @app.route("/voice", methods=["POST"])
# def voice():
#     response = VoiceResponse()

#     # Set timeout and speechTimeout values accordingly
#     with response.gather(action="/analyze_twilio", method="POST", timeout=60, speechTimeout=10) as gather:
#         # gather.say("Please start speaking after the beep.")
#         gather.say(
#             "Welcome to sentilysis. We are working on it. Please mail us at sentilysis@gmail.com for further details.")
#     # If no input is received, redirect to /analyze_twilio immediately
#     response.redirect("/analyze_twilio")

#     return str(response)


# # @app.route("/analyze_twilio", methods=["POST"])
# # def analyze_twilio():
# #     # Extract speech content from Twilio request
# #     voice_input = request.form["SpeechResult"]

# #     if voice_input:
# #         sentiment = analyze_sentiment(voice_input)

# #         # Get the response based on the input and text file content
# #         response = get_response(voice_input)
# #         chat.append(f"AI: {response}")

# #         return render_template("result.html", sentiment=sentiment, voice_input=chat)
# #     else:
# #         return render_template("result.html", sentiment="No voice input detected", voice_input="")

# @app.route("/analyze_twilio", methods=["POST"])
# def analyze_twilio():
#     print("Request Headers:", request.headers)  # Print the request headers
#     print("Request Form Data:", request.form)  # Print the form data
#     # Extract speech content from Twilio request query parameters
#     voice_input = request.args.get("SpeechResult")
#     print("Voice Input:", voice_input)  # Print the extracted voice input

#     if voice_input:
#         sentiment = analyze_sentiment(voice_input)

#         # Get the response based on the input and text file content
#         response = get_response(voice_input)
#         chat.append(f"AI: {response}")

#         return render_template("result.html", sentiment=sentiment, voice_input=chat)
#         # return render_template("result.html", sentiment=sentiment, voice_input=voice_input)
#     else:
#         return render_template("result.html", sentiment="No voice input detected", voice_input="")


# if __name__ == "__main__":
#     app.run(debug=True)
