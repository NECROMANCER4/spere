import threading
import tkinter as tk
import tkinter.font as tkFont
from pathlib import Path
from tkinter import Tk, Canvas, Text, Button, PhotoImage
from tkinter import filedialog
import pyttsx3
import speech_recognition as sr
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from ai2 import AI
OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path(r"C:\Users\Lenovo\PycharmProjects\AI project 1\build\assets\frame0")


def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)


# Initialize the main window
window = Tk()
window.geometry("1514x781")
window.configure(bg="#000000")
window.title("spere bot")
window.iconphoto(True, PhotoImage(file="(.png"))
# Create canvas for background
canvas = Canvas(
    window,
    bg="#000000",
    height=781,
    width=1514,
    bd=0,
    highlightthickness=0,
    relief="ridge"
)
canvas.place(x=0, y=0)


image_image_1 = PhotoImage(file=relative_to_assets("image_1.png"))
canvas.create_image(760.0, 392.0, image=image_image_1)


font_style = tkFont.Font(family="Helvetica", size=20, weight="bold")
text_box = Text(window, height=8, width=44, bg="#FFF9F9", fg="purple", borderwidth=0, font=font_style)
text_box.place(x=253, y=318)


# Function to run the text-to-speech engine
def speak_text(text):
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id)
    engine.setProperty("rate", 150)
    engine.setProperty('volume', 5.0)
    engine.say(text)
    engine.runAndWait()

def analyze_input():
    text = input_text.get("1.0", tk.END).strip()
    summarize_text(text)
    perform_sentiment_analysis(text)


def extract_keywords(text, top_n=5):

    sentences = sent_tokenize(text)


    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(sentences)


    feature_names = vectorizer.get_feature_names_out()


    tfidf_scores = tfidf_matrix.sum(axis=0).A1


    word_scores = {feature_names[i]: tfidf_scores[i] for i in range(len(feature_names))}


    sorted_keywords = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)

    return [keyword for keyword, score in sorted_keywords[:top_n]]


def summarize_text(text, num_sentences=3):
    sentences = sent_tokenize(text)
    if len(sentences) <= num_sentences:
        summary = text
    else:
        tfidf_vectorizer = TfidfVectorizer().fit_transform(sentences)
        tfidf_matrix = tfidf_vectorizer.toarray()
        cosine_sim = cosine_similarity(tfidf_matrix)


        scores = cosine_sim.sum(axis=1)


        indexed_scores = [(i, scores[i]) for i in range(len(scores))]
        indexed_scores.sort(key=lambda x: (-x[1], x[0]))


        important_keywords = extract_keywords(text, top_n=5)


        important_indices = set()
        for i, sentence in enumerate(sentences):
            if any(keyword in sentence.lower() for keyword in important_keywords):
                important_indices.add(i)


        selected_indices = set()
        for index, score in indexed_scores:
            selected_indices.add(index)
            if len(selected_indices) >= num_sentences:
                break


        for idx in important_indices:
            if len(selected_indices) < num_sentences:
                selected_indices.add(idx)


        sorted_indices = sorted(selected_indices)
        summary = ' '.join([sentences[i] for i in sorted_indices])
        summary = AI.wrap_text(summary,45)

    text_box.delete(1.0, tk.END)
    text_box.insert(tk.END, summary + "\n")


    speech_thread = threading.Thread(target=speak_text, args=(summary,))
    speech_thread.start()

sia = SentimentIntensityAnalyzer()

def perform_sentiment_analysis(text):
    sentiment_scores = sia.polarity_scores(text)
    display_sentiment_results(sentiment_scores, text)


def display_sentiment_results(scores, text):
    x = AI.analyze_sentiment(text)
    e = (scores['pos']) * 100
    f = (scores['neg']) * 100
    g = (scores['neu']) * 100
    final = f"\n {e:.2f}% positive,\n {f:.2f}% negative,\n {g:.2f}% neutral \n negative words: \n {x}"
    text_box = tk.Text(window, wrap='word', font=("Helvetica", 20, "bold"), bg='#910A67', fg='white')
    text_box.place(x=1106, y=360, width=300, height=200)
    text_box.insert(tk.END, final)
    text_box.config(state=tk.DISABLED)

def select_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        try:
            with open(file_path, 'r') as file:
                textf = file.read()
                summarize_text(textf)
                perform_sentiment_analysis(textf)
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
        summarize_text(text)
        perform_sentiment_analysis(text)
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
font_style3 = tkFont.Font(family="Helvetica", size=25, weight="bold")
lab = tk.Label(window,text="SUMMARY", bg="orange",font= font_style3, fg="black")
lab.place(x=253, y=250)
lab1 = tk.Label(window,text="SENTIMENT ANALYSIS", bg="orange",font=font_style3, fg="black")
lab1.place(x=1106, y=250)

font_style2 = tkFont.Font(family="Helvetica", size=15, weight="bold")
input_text = Text(window, height=2, width=70, bg="#FFF9F9", borderwidth=0, font= font_style2)
input_text.place(x=220, y=680)
window.resizable(True, True)
window.mainloop()