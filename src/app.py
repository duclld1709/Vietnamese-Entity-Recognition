from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from predict import predict_demo
from front import render_html
import os

app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), 'templates'))
CORS(app) 
@app.route('/')
def index():
    return render_template('demo.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print("Received data:", data)
        text = data.get('text', '')
        print("Text:", text)
        if not text.strip():
            return jsonify({'error': 'No text provided.'}), 400
        tokens, labels = predict_demo(text)
        print("Tokens:", tokens)
        print("Labels:", labels)
        html_result = render_html(tokens, labels)
        print("HTML Result:", html_result)
        return jsonify({'tokens': tokens, 'labels': labels, 'html_result': html_result})
    except Exception as e:
        print("Exception:", e)
        return jsonify({'error': str(e)}), 500
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)