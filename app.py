import warnings
try:
    from urllib3.exceptions import NotOpenSSLWarning
    warnings.filterwarnings("ignore", category=NotOpenSSLWarning)
except Exception:
    # urllib3 may not be installed yet or the import may fail in some envs
    pass

import os
import io
import base64
from flask import Flask, render_template, request, send_file
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader
import docx
import re

app = Flask(__name__)
nlp = spacy.load("en_core_web_sm")

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text(file, ext):
    if ext == 'txt':
        return file.read().decode('utf-8')
    elif ext == 'pdf':
        reader = PdfReader(file)
        return ''.join([page.extract_text() or '' for page in reader.pages])
    elif ext == 'docx':
        doc = docx.Document(file)
        return '\n'.join([p.text for p in doc.paragraphs])
    return ''

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)         # Normalize whitespace
    text = re.sub(r'[^A-Za-z0-9 ]+', '', text)  # Remove punctuation/special chars
    return text.strip()

def spacy_tokenizer(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    return tokens

def preprocess(text):
    text = clean_text(text)
    return ' '.join(spacy_tokenizer(text))

def check_similarity(texts, names):
    vectorizer = TfidfVectorizer(tokenizer=spacy_tokenizer, ngram_range=(1, 3))
    tfidf_matrix = vectorizer.fit_transform(texts)
    sim = cosine_similarity(tfidf_matrix)
    results = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            score = sim[i][j] * 100
            results.append((names[i], names[j], f"{score:.2f}%"))
    results.sort(key=lambda x: float(x[2].strip('%')), reverse=True)
    return results


@app.route('/favicon.ico')
def favicon():
    # Return a tiny 1x1 transparent PNG so browsers don't 404 for /favicon.ico
    png_b64 = (
        'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAAWgmWQ0AAA' \
        'AASUVORK5CYII='
    )
    data = base64.b64decode(png_b64)
    return send_file(io.BytesIO(data), mimetype='image/png')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        files = request.files.getlist('files')
        docs = []
        names = []
        for f in files:
            if f and allowed_file(f.filename):
                ext = f.filename.rsplit('.', 1)[1].lower()
                try:
                    text = extract_text(f, ext)
                    pre = preprocess(text)
                    if pre:
                        docs.append(pre)
                        names.append(f.filename)
                except Exception as e:
                    return render_template('index.html', error=f"Error with {f.filename}: {e}")
        if len(docs) >= 2:
            results = check_similarity(docs, names)
            return render_template('index.html', results=results)
        else:
            return render_template('index.html', error="Upload at least 2 valid documents.")
    return render_template('index.html')

if __name__ == '__main__':
    # Enable debug/reloader only when FLASK_DEBUG=1 is set in the environment.
    # This prevents the auto-reloader from restarting the server when files
    # (for example tests) are changed during development runs started by tools.
    debug_mode = os.getenv('FLASK_DEBUG', '0') == '1'
    print("Starting Flask server... (debug={})".format(debug_mode))
    # Only enable the reloader when debug_mode is True.
    app.run(debug=debug_mode, use_reloader=debug_mode)
