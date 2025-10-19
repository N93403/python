from preprocessing import load_data, scale_data
from models import get_logistic_model
import pandas as pd

train, test = load_data()
X = train.drop("target", axis=1)
y = train["target"]
X_train_scaled, X_test_scaled, scaler = scale_data(X, test)

model = get_logistic_model()
model.fit(X_train_scaled, y)

preds = model.predict(X_test_scaled)
pd.DataFrame({"prediction": preds}).to_csv("outputs/prediction_results.csv", index=False)
