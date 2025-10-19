from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import xgboost as xgb
import lightgbm as lgb

def get_logistic_model():
    return LogisticRegression(random_state=42, class_weight='balanced')

def get_random_forest():
    return RandomForestClassifier(n_estimators=200, max_depth=4, random_state=13)

def get_xgboost():
    return xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

def get_lightgbm():
    return lgb.LGBMClassifier(objective='binary', random_state=42)

def get_ensemble(models):
    return VotingClassifier(estimators=models, voting='soft')
