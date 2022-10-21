#from wsgi import main
from tests.train_model import run_and_log_model

if __name__ == '__main__':
    comments = '''running model with glove-200'''
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegressionCV
    from xgboost import XGBClassifier
    clf = XGBClassifier(n_estimators=200)
    run_and_log_model(clf,comments)