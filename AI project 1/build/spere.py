
# This file was generated by the Tkinter Designer by Parth Jadhav
# https://github.com/ParthJadhav/Tkinter-Designer


from pathlib import Path
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import speech_recognition as sr
from tkinter import filedialog
import tkinter.font as tkFont
import tkinter as tk
import numpy as np
from tkinter import Tk, Canvas, Text, Button, PhotoImage
from nltk.sentiment.vader import SentimentIntensityAnalyzer

OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path(r"C:\Users\Lenovo\PycharmProjects\AI project 1\build\assets\frame0")


def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)


window = Tk()

window.geometry("1514x781")
window.configure(bg = "#000000")


canvas = Canvas(
    window,
    bg = "#000000",
    height = 781,
    width = 1514,
    bd = 0,
    highlightthickness = 0,
    relief = "ridge"
)

canvas.place(x = 0, y = 0)
image_image_1 = PhotoImage(
    file=relative_to_assets("image_1.png"))
image_1 = canvas.create_image(
    760.0,
    392.0,
    image=image_image_1
)
font_style = tkFont.Font(family="Helvetica", size=20, weight="bold")
text_box = Text(window, height=8, width=44, bg="#FFF9F9", fg="purple", borderwidth=0, font=font_style)
text_box.place(x=253, y=318)
import pyttsx3
import threading

# Function to run the text-to-speech engine
def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

def analyze_input():
    text = input_text.get("1.0", tk.END).strip()  # Get text from input area

    summarize(text, 3)
    sent(text)
def append_and_after_every_2(text, n=5):
  """Adds a newline character after every n words in a given text.

  Args:
    text: The input text.
    n: The number of words after which to add a newline.

  Returns:
    The modified text with newlines.
  """

  words = text.split()
  new_text = []
  for i, word in enumerate(words):
    new_text.append(word)
    if (i + 1) % n == 0:
      new_text.append('\n')
  return ' '.join(new_text)

# Example usage:


def sentence_similarity(sent1, sent2):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([sent1, sent2])
    similarity_score = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
    return similarity_score


def summarize(text, num_sentences):

    sentences = sent_tokenize(text)

    # Create a similarity matrix
    similarity_matrix = np.zeros([len(sentences), len(sentences)])
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                similarity_matrix[i][j] = sentence_similarity(sentences[i], sentences[j])

    # Rank sentences based on similarity scores
    sentence_scores = np.sum(similarity_matrix, axis=1)
    top_sentences = np.argsort(sentence_scores)[::-1][:num_sentences]

    summary = ' '.join([sentences[i] for i in top_sentences])
    summary = append_and_after_every_2(summary)
    speech_thread = threading.Thread(target=speak_text, args=(summary,))
    speech_thread.start()
    text_box.delete(1.0, tk.END)
    text_box.insert(tk.END, summary + "\n")
    # Example usage

def sent(text):
    sia = SentimentIntensityAnalyzer()
    d = sia.polarity_scores(text)
    e = (d['pos']) * 100
    f = (d['neg']) * 100
    g = (d['neu']) * 100
    h = (d['compound']) * 100
    final = f"\n {e:.2f}% positive,\n {f:.2f}% negative,\n {g:.2f}% neutral \n"
    font_style1 = tkFont.Font(family="Helvetica", size=15, weight="bold")
    l32  = tk.Label(window, text = final,font= font_style1, bg= '#910A67')
    l32.place(x=1106, y=318)

def select_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        try:
            with open(file_path, 'r') as file:
                textf = file.read()
                summarize(textf, 3)
                sent(textf)
        except Exception as e:
            print(f"An error occurred: {e}")

recognizer = sr.Recognizer()
def recognize_speech():
    with sr.Microphone() as source:
        print("Adjusting for ambient noise... Please wait.")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        print("Listening...")
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio)
        print("You said: " + text)
        summarize(text, 3)
        sent(text)
    except sr.UnknownValueError:
        print("Sorry, I could not understand the audio.")
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")

button_image_1 = PhotoImage(
    file=relative_to_assets("button_1.png"))
button_1 = Button(
    image=button_image_1,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: analyze_input(),
    relief="flat"
)
button_1.place(
    x=1368.0,
    y=683.0,
    width=118.0,
    height=85.0
)

button_image_2 = PhotoImage(
    file=relative_to_assets("button_2.png"))
button_2 = Button(
    image=button_image_2,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: select_file(),
    relief="flat"
)
button_2.place(
    x=1247.0,
    y=675.0,
    width=113.0,
    height=101.0
)

button_image_3 = PhotoImage(
    file=relative_to_assets("button_3.png"))
button_3 = Button(
    image=button_image_3,
    borderwidth=0,
    highlightthickness=0,
    command=lambda: recognize_speech(),
    relief="flat"
)
button_3.place(
    x=1106.0,
    y=679.0,
    width=105.0,
    height=89.0
)
lab = tk.Label(window,text="summary:-", bg="black",font=(100), fg="white")
lab.place(x=253, y=250)
lab1 = tk.Label(window,text="sentiment analysis:-", bg="orange",font=(100), fg="black")
lab1.place(x=1106, y=280)
  # Position it at the same coordinates as the rectangle


font_style2 = tkFont.Font(family="Helvetica", size=15, weight="bold")
# Create a Text widget for user input within the rectangle
input_text = Text(window, height=2, width=70, bg="#FFF9F9", borderwidth=0, font= font_style2)
input_text.place(x=220, y=680)  # Adjust the position to fit within the rectangle

window.resizable(False, False)
window.mainloop()
