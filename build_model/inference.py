import pickle
from build_model import preprocessing
from build_model import config as cn


with open(cn.MODEL_FILE, 'rb') as f:
    classifier = pickle.load(f)

def prediction(event):
        tweet = event.get("tweet")
        predict_payload = preprocessing.wordvecs(tweet)

        return {
            "tweet": tweet,
            "prediction": str(classifier.predict_proba(predict_payload))
        }