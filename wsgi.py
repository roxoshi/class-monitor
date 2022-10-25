from flask import Flask, request, jsonify, render_template
from build_model.inference import prediction

app = Flask(__name__)

@app.route('/')
def my_form():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def make_predict():
    event = request.json
    if 'application/json' in request.content_type:
        return jsonify({
            "prediction": prediction(event)
        })
    return jsonify({
        "prediction": None
    })


