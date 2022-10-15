"""
Here we will train an ML model on our 
corpus which we have converted to document vectors 
"""
import numpy as np
import logging
import pickle
import json
from preprocessing import TransformText
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, precision_score, accuracy_score
import config as cn
logging.basicConfig(level=logging.DEBUG)

def run_and_log_model(estimator, comments=None) -> None:
    train_path = cn.TRAIN_FILE
    t = TransformText
    t.readcsv = train_path
    output_dataset = t.run()
    logging.info("Input dataset loaded")
    labels = np.array([x[0] for x in output_dataset])
    word_vector = np.array([x[1] for x in output_dataset])

    logging.info(f"train data shape: {word_vector.shape}")
    X_train, X_test, y_train, y_test = train_test_split(
        word_vector, labels, test_size=0.33, random_state=42)
    logging.info("Training started...")
    classifier = estimator
    classifier.fit(X_train, y_train)
    logging.info("Training complete")
    y_test_pred = classifier.predict(X_test)
    f1_val = f1_score(y_test, y_test_pred)
    auc  = roc_auc_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred)
    accuracy = accuracy_score(y_test, y_test_pred)
    logging.info(f'f1 score is:{f1_val}')
    # Logging some key params and comments for the
    # model into a file
    output_dict= json.dumps({
        'model_algorithm' : classifier.__class__.__name__,
        'data_shape': str(word_vector.shape),
        'params': classifier.get_params(),
        'f1_score' : f1_val,
        'AUC' : auc,
        'precision': precision,
        'accuracy': accuracy,
        'random_state': 10,
        'comments':comments
    }, indent=4)
    with open (cn.MODEL_LOGS, 'a') as f:
        f.write(output_dict + "\n")
    with open(cn.MODEL_FILE,'wb') as fp:
        pickle.dump(classifier, fp)

if __name__ == '__main__':
    comments = '''running model with frequency BOW, default val 0'''
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegressionCV
    clf = RandomForestClassifier(n_estimators=500)
    run_and_log_model(clf,comments)
