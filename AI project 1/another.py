import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import urllib.request
import bs4 as bs
import re
import heapq


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

    # Print the summary
    print("\nSummary:")
    print(summary)

if __name__ == "__main__":
    # Get the URL from the user
    url = input("Enter the URL of the article you want to summarize: ")
    summarize_article(url)