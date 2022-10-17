import pickle
import numpy as np
from build_model import config as cn
from flask import Flask
from build_model.preprocessing import TransformText

def main():
    app = Flask(__name__)
    with open(cn.MODEL_FILE, 'rb') as f:
        classifier = pickle.load(f)
    tr_array = TransformText.transform("Hi I am a small person")
    predict_payload = np.array(tr_array).reshape(1,-1)
    print(classifier.predict(predict_payload))
    

# @app.route("/")
# def predict_score()



# dummy_features = np.array([0]*1000).reshape(1,-1) #np.random.rand(1,1000)


# score = classifier.predict_proba(dummy_features)
# print(f"score is: {score}")

