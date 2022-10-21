from flask import Flask, request, jsonify
from build_model.inference import prediction

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def main():
    content_type = request.headers.get('Content-Type')
    if (content_type == 'application/json'):
        event = request.json
        return jsonify(prediction(event))


