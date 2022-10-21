#from wsgi import main
from tests.train_model import run_and_log_model

if __name__ == '__main__':
    comments = '''running model with frequency BOW, default val 0'''
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegressionCV
    from xgboost import XGBClassifier
    clf = XGBClassifier()
    run_and_log_model(clf,comments)