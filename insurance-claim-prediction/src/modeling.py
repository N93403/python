from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def get_logistic_model():
    return LogisticRegression(max_iter=10000)

def get_decision_tree():
    return DecisionTreeClassifier(random_state=1)

def get_random_forest():
    return RandomForestClassifier(random_state=1)
