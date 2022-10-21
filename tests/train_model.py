"""
Here we will train an ML model on our 
corpus which we have converted to document vectors 
"""
import numpy as np
import pandas as pd
import logging
import pickle
import json
from build_model.preprocessing import transform_dataframe
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, precision_score, accuracy_score
from build_model import config as cn
logging.basicConfig(level=logging.DEBUG)

def run_and_log_model(estimator, comments=None) -> None:
    train_path = cn.TRAIN_FILE
    df_train = pd.read_csv(train_path)
    logging.info("Input dataset loaded")
    labels = df_train['label']
    word_vector = transform_dataframe(df_train,colname='tweet')
    logging.info(f"X shape: {word_vector.shape}, Y shape: {labels.shape}")
    logging.info(f"train data shape: {word_vector.shape}")
    X_train, X_test, y_train, y_test = train_test_split(
        word_vector, labels, test_size=0.3)
    logging.info("Training started...")
    classifier = estimator
    classifier.fit(X_train, y_train)
    logging.info("Training complete")
    y_test_pred = classifier.predict(X_test)
    logging.info(f"Shape of y predicted, {y_test_pred.shape}")
    logging.info(f"Shape of x test: {X_test.shape}")
    f1_val = f1_score(y_test, y_test_pred)
    auc  = roc_auc_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred)
    accuracy = accuracy_score(y_test, y_test_pred)
    logging.info(f'f1 score is:{f1_val}')
    logging.info(f'precision score is:{precision}')
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
    comments = '''running model with word2vec embedding model'''
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegressionCV
    from xgboost import XGBClassifier
    clf = XGBClassifier(random_state=10)
    run_and_log_model(clf,comments)
