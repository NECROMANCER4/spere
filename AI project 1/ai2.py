import threading
import numpy as np
import pyttsx3
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import urllib.request
import bs4 as bs
import re
import heapq
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()
class AI:
    def append_and_after_every_5(text, n=5):
        words = text.split()
        new_text = []
        for i, word in enumerate(words):
            new_text.append(word)
            if (i + 1) % n == 0:
                new_text.append('\n')
        return ' '.join(new_text)

    def old_sentence_similarity(sent1, sent2):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([sent1, sent2])
        similarity_score = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
        return similarity_score

    def old_summarize(text, num_sentences):

        sentences = sent_tokenize(text)

        # Create a similarity matrix
        similarity_matrix = np.zeros([len(sentences), len(sentences)])
        for i in range(len(sentences)):
            for j in range(len(sentences)):
                if i != j:
                    similarity_matrix[i][j] = text.old_sentence_similarity(sentences[i], sentences[j])

        # Rank sentences based on similarity scores
        sentence_scores = np.sum(similarity_matrix, axis=1)
        top_sentences = np.argsort(sentence_scores)[::-1][:num_sentences]

        summary = ' '.join([sentences[i] for i in top_sentences])
        summary = text.append_and_after_every_5(summary)
        speech_thread = threading.Thread(target=speak_text, args=(summary,))
        speech_thread.start()

    def wrap_text(text, max_length):
        words = text.split()
        wrapped_lines = []
        current_line = ""

        for word in words:
            # Check if adding this word would exceed the max_length
            if len(current_line) + len(word) + 1 > max_length:
                # If it does, append the current line to the wrapped lines
                wrapped_lines.append(current_line)
                # Start a new line with the current word
                current_line = word
            else:
                # If it doesn't, add the word to the current line
                if current_line:  # If current_line is not empty, add a space
                    current_line += " "
                current_line += word

        # Don't forget to add the last line if it exists
        if current_line:
            wrapped_lines.append(current_line)

        return "\n".join(wrapped_lines)


    def analyze_sentiment(text):
        sia = SentimentIntensityAnalyzer()
        words = word_tokenize(text)
        negative_words = []
        for word in words:
            score = sia.polarity_scores(word)
            if score['compound'] < 0:
                negative_words.append(word)
                negative_words.append('\n')
        negative_words = "".join(negative_words)
        return negative_words


class net:
    def summarize_article(url):
        # Scrape data from the provided URL
        try:
            scraped_data = urllib.request.urlopen(url)
            article = scraped_data.read()
        except Exception as e:
            print(f"Error fetching the article: {e}")
            return

        # Parse the article using BeautifulSoup
        parsed_article = bs.BeautifulSoup(article, 'lxml')
        paragraphs = parsed_article.find_all('p')

        # Combine all paragraphs into a single string
        article_text = ""
        for p in paragraphs:
            article_text += p.text

        # Remove square brackets and extra spaces
        article_text = re.sub(r'\[[0-9]*\]', ' ', article_text)
        article_text = re.sub(r'\s+', ' ', article_text)

        # Remove special characters and digits
        formatted_article_text = re.sub('[^a-zA-Z]', ' ', article_text)
        formatted_article_text = re.sub(r'\s+', ' ', formatted_article_text)

        # Tokenize the text into words
        stop_words = set(stopwords.words('english'))
        word_frequencies = {}
        for word in word_tokenize(formatted_article_text.lower()):
            if word not in stop_words:
                if word not in word_frequencies:
                    word_frequencies[word] = 1
                else:
                    word_frequencies[word] += 1

        # Calculate the maximum frequency of any word
        maximum_frequency = max(word_frequencies.values())

        # Calculate weighted frequencies
        for word in word_frequencies.keys():
            word_frequencies[word] = (word_frequencies[word] / maximum_frequency)

        # Tokenize the article into sentences
        sentence_list = sent_tokenize(article_text)
        sentence_scores = {}

        for sent in sentence_list:
            for word in word_tokenize(sent.lower()):
                if word in word_frequencies.keys():
                    if sent not in sentence_scores:
                        sentence_scores[sent] = word_frequencies[word]
                    else:
                        sentence_scores[sent] += word_frequencies[word]

        # Get the top N sentences
        summary_sentences = heapq.nlargest(5, sentence_scores, key=sentence_scores.get)

        # Join the sentences to form the final summary
        summary = ' '.join(summary_sentences)