from flask import Flask, request, url_for, redirect, render_template
import nltk
from nltk import word_tokenize
nltk.download('punkt')
import math
app = Flask(__name__, template_folder='templates')
from jinja2 import Environment

def enumerate_filter(iterable, start=0):
    return enumerate(iterable, start=start)


env = Environment()
env.filters['enumerate'] = enumerate_filter

@app.route('/')
def hello_world():
    return render_template("tfidf.html")

def termfreq(document, word):
    N = len(document)
    occurance = len([token for token in document if token == word])
    return occurance/N

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    int_features = ([str(x) for x in request.form.values()])
    s1 = int_features[0].lower()
    s2 = int_features[1].lower()
    e1 = []
    e2 = []
    e1 = word_tokenize(s1)
    e2 = word_tokenize(s2)
    c1 = len(e1)
    c2 = len(e2)
    epro=e1+e2
    tf1 = []
    tf2 = []
    for x1 in e1:
        tf1.append(termfreq(e1, x1))

    for x2 in e2:
        tf2.append(termfreq(e2, x2))
    set = zip(e1, e2)
    idf1 = []
    c11 = 0
    for x in e1:
        if x in e2:
            c11 = 2
        else:
            c11 = 1
        idf1.append(math.log10(2 / c11))

    c22 = 0
    idf2 = []
    for x in e2:
        if x in e1:
            c22 = 2
        else:
            c22 = 1
        idf2.append(math.log10(2 / c22))
    tfidf = []
    for i in range(0, len(tf1)):
        tfidf.append(tf1[i] * idf1[i])
    for i in range(0, len(tf2)):
        tfidf.append(tf2[i] * idf2[i])

    return render_template('tfidf.html',
                               pred=zip(tfidf, epro))


if __name__ == '__main__':
    app.run(debug=True)
