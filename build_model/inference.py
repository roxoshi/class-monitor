import pickle
from build_model.preprocessing import wordvecs
from build_model import config as cn


with open(cn.MODEL_FILE, 'rb') as f:
    classifier = pickle.load(f)

def prediction(event):
        tweet = event.get("tweet")
        predict_payload = wordvecs(tweet)

        return {
            "tweet": tweet,
            "prediction": str(classifier.predict(predict_payload))
        }