import pickle
import numpy as np
from build_model import config as cn
from flask import Flask, request, jsonify
from build_model.preprocessing import TransformText

app = Flask(__name__)

with open(cn.MODEL_FILE, 'rb') as f:
    classifier = pickle.load(f)

@app.route('/predict', methods=['POST'])
def prediction():
    if request.method == "POST":
        req = request.json
        tweet = req.get("tweet")
        tr_array = TransformText.transform(tweet)
        predict_payload = np.array(tr_array).reshape(1,-1)

        return jsonify({
            "tweet": tweet,
            "prediction": str(classifier.predict(predict_payload)[0])
        })
    

# @app.route("/")
# def predict_score()



# dummy_features = np.array([0]*1000).reshape(1,-1) #np.random.rand(1,1000)


# score = classifier.predict_proba(dummy_features)
# print(f"score is: {score}")

