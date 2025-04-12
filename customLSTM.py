import speech_recognition as sr
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from flask import Flask, render_template, request
import openai
import nltk
# import pickle
import tensorflow as tf
# from textblob import TextBlob

# Initialize the sentiment analyzer
sia = SentimentIntensityAnalyzer()

API_KEY = open("C:/Users/dipta/Desktop/Diptam/Projects/Professional-Project/Smart-India-Hackathon/Sentilysis/SENTI CODE/gaurav/API_KEY", "r").read()
openai.api_key = API_KEY

app = Flask(__name__)

chat = []


# with open('model_lstm.pkl', 'rb') as f:
#     loaded_model = pickle.load(f)


# # # Load the trained model from the pickle file
# with open('model_lstm.pkl', 'rb') as file:
#     trained_model = pickle.load(file)


# # Load the tokenizer used during training
# with open('tokenizer.pkl', 'rb') as file:
#     tokenizer = pickle.load(file)

# max_len = 274  # Replace with the actual value used during training
# max_words = 60192

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


# Function to perform sentiment analysis


def analyze_sentiment(text):
    sentiment_scores = sia.polarity_scores(text)
    compound_score = sentiment_scores['compound']

    if compound_score >= 0.05:
        chat.append("(Positive)")
        return "Positive"
    elif compound_score <= -0.05:
        chat.append("(negetive)")
        return "Negative"
    else:
        chat.append("(neutral)")
        return "Neutral"
    # return sentiment


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


@app.route("/analyze", methods=["POST"])
def analyze():
    user_text = get_voice_input()
    if user_text:
        sentiment = analyze_sentiment(user_text)

        # Get user input
        user_input_chat = f"{user_text}."

        # Get the response based on the input and text file content
        response = get_response(user_input_chat)
        chat.append(f"AI: {response}")
        # Display the generated response
        # print(f"AI: {response}")
        print("__________________________________________________")
        print(chat)
        print("__________________________________________________")
        return render_template(
            "result.html", sentiment=sentiment, voice_input=chat
        )
    else:
        return render_template(
            "result.html", sentiment="No voice input detected", voice_input=""
        )


if __name__ == "__main__":
    app.run(debug=True)
