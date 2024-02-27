from flask import Flask, render_template, request
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import os

# Specify a writable directory for NLTK data in /tmp
nltk_data_dir = '/tmp/nltk_data'
nltk.data.path.append(nltk_data_dir)

# Download the vader_lexicon resource if not present
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon', download_dir=nltk_data_dir)
    
app = Flask(__name__)

@app.route("/", methods=["GET","POST"])
def main():
    if request.method == "POST":
        inp = request.form.get("inp")
        sid = SentimentIntensityAnalyzer()
        score = sid.polarity_scores(inp)
        if score["neg"] != 0:
            return render_template('home.html',message="Negative😡")
        else:
            return render_template('home.html',message="Positive😊")
    return render_template('home.html')
