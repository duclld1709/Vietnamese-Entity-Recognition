from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from predict import predict_demo
from front import render_html
import os

app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), 'templates'))
CORS(app)  # Enable CORS for all routes

@app.route('/')
def index():
    return render_template('demo.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '')
    if not text.strip():
        return jsonify({'error': 'No text provided.'}), 400
    tokens, labels = predict_demo(text)
    html_result = render_html(tokens, labels)
    return jsonify({'tokens': tokens, 'labels': labels, 'html_result': html_result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
